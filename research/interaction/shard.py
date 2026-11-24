"""Immutable uint16 token shards with document target spans and exact cursors."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Iterator

from research.interaction.tokenizer import SPECIAL, Tokenizer


def _hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            block = file.read(1048576)
            if not block:
                break
            value.update(block)
    return value.hexdigest()


def _pack(token: Tokenizer, rows: list[dict]) -> tuple[list[int], list[list[int]]]:
    ids = [SPECIAL["BOS"]]
    spans = []
    for row in rows:
        ids.append({"system": SPECIAL["SYSTEM"], "user": SPECIAL["USER"],
                    "assistant": SPECIAL["ASSISTANT"], "tool": SPECIAL["TOOL"]}[row["role"]])
        start = len(ids)
        ids.extend(token.encode(row["text"]))
        if row["role"] == "assistant":
            spans.append([start, len(ids)])
        ids.append(SPECIAL["END"])
    ids.append(SPECIAL["EOS"])
    return ids, spans


def write(corpus: Path, token: Tokenizer, root: Path,
          limit: int = 1000000) -> dict:
    if root.exists():
        raise FileExistsError(root)
    if not 1 <= limit <= 100000000:
        raise ValueError("shard token limit is out of range")
    root.mkdir(parents=True)
    states = {}
    files = []
    total = {"documents": 0, "raw": 0, "tokens": 0}

    def openone(split: str) -> dict:
        index = len(states.get(split, {}).get("files", []))
        name = split.replace("validation", "valid") + str(index)
        data = root / (name + ".bin")
        side = root / (name + ".jsonl")
        row = {"name": name, "data": data, "side": side, "count": 0,
               "file": data.open("xb"), "index": side.open("x", encoding="utf-8"),
               "documents": 0, "raw": 0}
        states.setdefault(split, {"files": []})["files"].append(row)
        return row

    try:
        with corpus.open("r", encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                split = row["split"]
                if split not in {"train", "validation", "test"}:
                    raise ValueError("invalid split name")
                ids, spans = _pack(token, row["messages"])
                if len(ids) > limit:
                    raise ValueError("episode exceeds shard token limit")
                current = states.get(split, {}).get("files", [])
                shard = current[-1] if current else None
                if shard is None or shard["count"] + len(ids) > limit:
                    shard = openone(split)
                start = shard["count"]
                blob = struct.pack("<" + "H" * len(ids), *ids)
                shard["file"].write(blob)
                info = {"id": row["id"], "source": row["source"], "doc": row["doc"],
                        "family": row["family"], "template": row["template"], "split": split,
                        "synthetic": row["synthetic"], "kind": row["kind"], "digest": row["digest"],
                        "start": start, "tokens": len(ids), "raw": row["raw"],
                        "targets": [[start + left, start + right] for left, right in spans]}
                shard["index"].write(json.dumps(info, sort_keys=True, separators=(",", ":")) + "\n")
                shard["count"] += len(ids)
                shard["documents"] += 1
                shard["raw"] += row["raw"]
                total["documents"] += 1
                total["raw"] += row["raw"]
                total["tokens"] += len(ids)
    except BaseException:
        for state in states.values():
            for shard in state["files"]:
                shard["file"].close()
                shard["index"].close()
        raise

    for split, state in states.items():
        for shard in state["files"]:
            shard["file"].flush()
            shard["index"].flush()
            shard["file"].close()
            shard["index"].close()
            files.append({"name": shard["name"], "split": split,
                          "tokens": shard["count"], "documents": shard["documents"],
                          "raw": shard["raw"], "data": shard["data"].name,
                          "index": shard["side"].name,
                          "sha256": _hash(shard["data"]),
                          "proof": _hash(shard["side"]),
                          "bytes": shard["data"].stat().st_size})
    manifest = {"version": 1, "tokenizer": token.digest, "vocab": token.vocab,
                "format": "little-endian-uint16", "files": files, "total": total}
    path = root / "manifest.json"
    with path.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, sort_keys=True, separators=(",", ":"))
        file.write("\n")
    return manifest


def stream(root: Path, split: str, cursor: dict | None = None,
           size: int = 4096) -> Iterator[dict]:
    if not 1 <= size <= 65536:
        raise ValueError("read size must be between 1 and 65536 tokens")
    data = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    files = [row for row in data["files"] if row["split"] == split]
    begin = cursor or {"shard": 0, "token": 0}
    if set(begin) != {"shard", "token"}:
        raise ValueError("cursor must contain shard and token")
    if not isinstance(begin["shard"], int) or not isinstance(begin["token"], int):
        raise ValueError("cursor coordinates must be integers")
    if begin["shard"] < 0 or begin["token"] < 0 or begin["shard"] > len(files):
        raise ValueError("cursor is out of range")
    for pos in range(begin["shard"], len(files)):
        row = files[pos]
        offset = begin["token"] if pos == begin["shard"] else 0
        if offset > row["tokens"]:
            raise ValueError("token cursor is out of range")
        with (root / row["index"]).open("r", encoding="utf-8") as file:
            index = [json.loads(line) for line in file]
        with (root / row["data"]).open("rb") as file:
            while offset < row["tokens"]:
                count = min(size, row["tokens"] - offset)
                file.seek(offset * 2)
                raw = file.read(count * 2)
                if len(raw) != count * 2:
                    raise ValueError("truncated token shard")
                ids = list(struct.unpack("<" + "H" * count, raw))
                mask = [False] * count
                end = offset + count
                for doc in index:
                    for left, right in doc["targets"]:
                        first = max(left, offset)
                        last = min(right, end)
                        if first < last:
                            mask[first - offset:last - offset] = [True] * (last - first)
                after = {"shard": pos, "token": end}
                if end == row["tokens"]:
                    after = {"shard": pos + 1, "token": 0}
                yield {"ids": ids, "mask": mask, "cursor": after}
                offset = end
    return


def check(root: Path, digest: str | None = None) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if digest is not None and manifest["tokenizer"] != digest:
        raise ValueError("tokenizer hash mismatch")
    totals = {name: {"tokens": 0, "documents": 0, "raw": 0}
              for name in ("train", "validation", "test")}
    ids = set()
    for row in manifest["files"]:
        data = root / row["data"]
        index = root / row["index"]
        if data.stat().st_size != row["bytes"] or row["bytes"] != row["tokens"] * 2:
            raise ValueError("token shard size mismatch")
        if _hash(data) != row["sha256"] or _hash(index) != row["proof"]:
            raise ValueError("shard or index hash mismatch")
        end = 0
        count = 0
        raw = 0
        with index.open("r", encoding="utf-8") as file:
            for line in file:
                doc = json.loads(line)
                if doc["split"] != row["split"] or doc["start"] != end or doc["tokens"] < 1:
                    raise ValueError("shard document offsets or split mismatch")
                if doc["id"] in ids:
                    raise ValueError("duplicate shard document ID")
                ids.add(doc["id"])
                last = doc["start"]
                for left, right in doc["targets"]:
                    if left < last or right < left or right > doc["start"] + doc["tokens"]:
                        raise ValueError("invalid answer target span")
                    last = right
                end += doc["tokens"]
                count += 1
                raw += doc["raw"]
        if end != row["tokens"] or count != row["documents"] or raw != row["raw"]:
            raise ValueError("shard index totals mismatch")
        total = totals[row["split"]]
        total["tokens"] += end
        total["documents"] += count
        total["raw"] += raw
    alltotal = {key: sum(row[key] for row in totals.values())
                for key in ("tokens", "documents", "raw")}
    if alltotal != manifest["total"]:
        raise ValueError("shard totals mismatch")
    return {"documents": len(ids), "files": len(manifest["files"]),
            "tokens": manifest["total"]["tokens"], "hash": manifest["tokenizer"]}
