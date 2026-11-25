"""Byte-level BPE with lossless UTF-8 and explicit chat boundaries."""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


BASE = 256
SPECIAL = {"BOS": 256, "EOS": 257, "END": 258, "SYSTEM": 259,
           "USER": 260, "ASSISTANT": 261, "TOOL": 262, "META": 263}
START = 264
PART = re.compile(rb"\s+|[^\s]+")
ROLES = {"system": SPECIAL["SYSTEM"], "user": SPECIAL["USER"],
         "assistant": SPECIAL["ASSISTANT"], "tool": SPECIAL["TOOL"]}
NAMES = {value: key.lower() for key, value in ROLES.items()}


class Tokenizer:
    def __init__(self, merges: Iterable[tuple[int, int]] = (), target: int = 4096,
                 seen: int = 0):
        if not 264 <= target <= 65536:
            raise ValueError("target vocabulary must be between 264 and 65536")
        self.target = target
        self.seen = seen
        self.parts = {key: bytes((key,)) for key in range(BASE)}
        self.ranks = {}
        self.codes = {}
        for rank, pair in enumerate(merges):
            left, right = pair
            if left not in self.parts or right not in self.parts:
                raise ValueError("merge refers to a token that does not exist")
            code = START + rank
            if code >= target or pair in self.ranks:
                raise ValueError("merge table exceeds target or repeats a pair")
            self.ranks[pair] = rank
            self.codes[pair] = code
            self.parts[code] = self.parts[left] + self.parts[right]
        self.size = max(START, START + len(self.ranks))
        if self.size > target:
            raise ValueError("tokenizer vocabulary exceeds target")
        self._digest_cache: str | None = None

    @classmethod
    def train(cls, texts: Iterable[str], target: int = 4096,
              budget: int = 1048576) -> Tokenizer:
        if target <= START or budget < 1:
            raise ValueError("target must allow merges and budget must be positive")
        words = Counter()
        seen = 0
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("training text must be str")
            raw = text.encode("utf-8")
            if seen + len(raw) > budget:
                raw = raw[:budget - seen]
            if not raw:
                break
            seen += len(raw)
            for match in PART.finditer(raw):
                words[tuple(match.group())] += 1
            if seen >= budget:
                break
        if not words:
            raise ValueError("no training bytes were supplied")

        pairs = Counter()
        members = defaultdict(set)
        heap = []

        def push(pair: tuple[int, int], amount: int) -> None:
            value = pairs.get(pair, 0) + amount
            if value <= 0:
                pairs.pop(pair, None)
                return
            pairs[pair] = value
            heapq.heappush(heap, (-value, pair[0], pair[1]))

        def add(word: tuple[int, ...], count: int) -> None:
            if len(word) < 2:
                return
            seq = list(zip(word, word[1:]))
            for pair in seq:
                push(pair, count)
            for pair in set(seq):
                members[pair].add(word)

        for word, count in words.items():
            add(word, count)

        merges = []
        limit = target - START
        while len(merges) < limit and heap:
            while heap:
                neg, left, right = heapq.heappop(heap)
                pair = (left, right)
                if pairs.get(pair) == -neg:
                    break
            else:
                break
            if -neg < 2:
                break
            affected = tuple(members.get(pair, ()))
            if not affected:
                continue
            code = START + len(merges)
            merges.append(pair)
            for word in affected:
                count = words.pop(word, 0)
                if not count:
                    continue
                seq = list(zip(word, word[1:]))
                for old in seq:
                    push(old, -count)
                for old in set(seq):
                    members[old].discard(word)
                out = []
                pos = 0
                while pos < len(word):
                    if pos + 1 < len(word) and (word[pos], word[pos + 1]) == pair:
                        out.append(code)
                        pos += 2
                    else:
                        out.append(word[pos])
                        pos += 1
                merged = tuple(out)
                words[merged] += count
                add(merged, count)
        return cls(merges, target, seen)

    @property
    def vocab(self) -> int:
        return self.target

    @property
    def digest(self) -> str:
        if self._digest_cache is None:
            raw = json.dumps(self._data(), sort_keys=True, separators=(",", ":")).encode()
            self._digest_cache = hashlib.sha256(raw).hexdigest()
        return self._digest_cache

    def _data(self) -> dict:
        return {"version": 1, "target": self.target, "seen": self.seen,
                "special": SPECIAL, "merges": [list(row) for row in self.ranks]}

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        out = [SPECIAL["BOS"]] if bos else []
        for match in PART.finditer(text.encode("utf-8")):
            word = list(match.group())
            while len(word) > 1:
                best = None
                for pos in range(len(word) - 1):
                    rank = self.ranks.get((word[pos], word[pos + 1]))
                    if rank is not None and (best is None or rank < best[0]):
                        best = (rank, pos)
                if best is None:
                    break
                pair = (word[best[1]], word[best[1] + 1])
                code = self.codes[pair]
                merged = []
                pos = 0
                while pos < len(word):
                    if pos + 1 < len(word) and (word[pos], word[pos + 1]) == pair:
                        merged.append(code)
                        pos += 2
                    else:
                        merged.append(word[pos])
                        pos += 1
                word = merged
            out.extend(word)
        if eos:
            out.append(SPECIAL["EOS"])
        return out

    def encode_with_offsets(self, text: str) -> dict[str, list]:
        """Encode text and return exact UTF-8 byte and overlapping character spans.

        Byte offsets are half-open offsets into text.encode('utf-8'). Character
        offsets are half-open Unicode-code-point spans. A BPE token containing
        only part of a multibyte code point maps to that full code point.
        """
        if not isinstance(text, str):
            raise TypeError("text must be str")
        raw = text.encode("utf-8")
        ids: list[int] = []
        byte_spans: list[tuple[int, int]] = []
        cursor = 0
        for match in PART.finditer(raw):
            word = list(match.group())
            spans = [(cursor + pos, cursor + pos + 1) for pos in range(len(word))]
            while len(word) > 1:
                best = None
                for pos in range(len(word) - 1):
                    rank = self.ranks.get((word[pos], word[pos + 1]))
                    if rank is not None and (best is None or rank < best[0]):
                        best = (rank, pos)
                if best is None:
                    break
                pair = (word[best[1]], word[best[1] + 1])
                code = self.codes[pair]
                merged_tokens = []
                merged_spans = []
                pos = 0
                while pos < len(word):
                    if pos + 1 < len(word) and (word[pos], word[pos + 1]) == pair:
                        merged_tokens.append(code)
                        merged_spans.append((spans[pos][0], spans[pos + 1][1]))
                        pos += 2
                    else:
                        merged_tokens.append(word[pos])
                        merged_spans.append(spans[pos])
                        pos += 1
                word, spans = merged_tokens, merged_spans
            ids.extend(word)
            byte_spans.extend(spans)
            cursor = match.end()

        byte_to_char = [0] * len(raw)
        byte_cursor = 0
        for char_index, character in enumerate(text):
            encoded_length = len(character.encode("utf-8"))
            byte_to_char[byte_cursor:byte_cursor + encoded_length] = [char_index] * encoded_length
            byte_cursor += encoded_length
        char_spans = [(byte_to_char[start], byte_to_char[end - 1] + 1)
                      for start, end in byte_spans]
        if self.decode(ids) != text:
            raise AssertionError("offset encoding failed to preserve the source text")
        return {"ids": ids, "byte_offsets": byte_spans, "char_offsets": char_spans}

    def decode(self, ids: Iterable[int]) -> str:
        raw = bytearray()
        for item in ids:
            if item not in self.parts:
                raise ValueError(f"special or invalid token in plain-text decode: {item}")
            raw.extend(self.parts[item])
        return raw.decode("utf-8", errors="strict")

    def chat(self, rows: Iterable[dict[str, str]], bos: bool = True,
             eos: bool = True) -> list[int]:
        out = [SPECIAL["BOS"]] if bos else []
        for row in rows:
            role = row["role"]
            if role not in ROLES or not isinstance(row.get("text"), str):
                raise ValueError("chat rows require system/user/assistant/tool role and text")
            out.append(ROLES[role])
            out.extend(self.encode(row["text"]))
            out.append(SPECIAL["END"])
        if eos:
            out.append(SPECIAL["EOS"])
        return out

    def parse(self, ids: Iterable[int]) -> list[dict[str, str]]:
        rows = []
        role = None
        body = []
        for item in ids:
            if item in (SPECIAL["BOS"], SPECIAL["EOS"]):
                if role is not None or body:
                    raise ValueError("BOS/EOS appeared inside an open message")
            elif item in NAMES:
                if role is not None:
                    raise ValueError("nested chat role before message end")
                role = NAMES[item]
                body = []
            elif item == SPECIAL["END"]:
                if role is None:
                    raise ValueError("message end without a role")
                rows.append({"role": role, "text": self.decode(body)})
                role = None
                body = []
            elif role is None:
                raise ValueError("text token outside a chat message")
            else:
                body.append(item)
        if role is not None:
            raise ValueError("chat ended before message end")
        return rows

    def save(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self._data()
        data["sha256"] = self.digest
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("x", encoding="utf-8") as file:
            json.dump(data, file, sort_keys=True, separators=(",", ":"))
            file.write("\n")
        os.replace(temp, path)

    @classmethod
    def load(cls, path: Path) -> Tokenizer:
        data = json.loads(path.read_text(encoding="utf-8"))
        digest = data.pop("sha256", None)
        obj = cls((tuple(row) for row in data["merges"]), data["target"], data["seen"])
        if data.get("version") != 1 or data.get("special") != SPECIAL or digest != obj.digest:
            raise ValueError("tokenizer version, special IDs or hash mismatch")
        return obj
