"""Streaming source curation, provenance, deduplication and family splits."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator


MAIL = re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b")
PHONE = re.compile(r"(?<!\w)(?:\+?\d[ .()/-]*){9,15}(?!\w)")
CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
ADDRESS = re.compile(r"(?i)\b\d{1,5}\s+(?:[a-z0-9.'-]+\s+){0,4}(?:street|st\.?|road|rd\.?|avenue|ave\.?|lane|ln\.?|drive|dr\.?|boulevard|blvd\.?|court|ct\.?|circle|cir\.?)\b")
NUMBER = re.compile(r"\d+")
SPACE = re.compile(r"\s+")
WORD = re.compile(r"[\w']+", re.UNICODE)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def norm(text: str) -> str:
    return SPACE.sub(" ", text.casefold()).strip()


def template(text: str) -> str:
    return NUMBER.sub("#", norm(text))


def sensitive(text: str) -> bool:
    return bool(MAIL.search(text) or PHONE.search(text) or CARD.search(text) or
                SSN.search(text) or IPV4.search(text) or ADDRESS.search(text))


def split(family: str) -> str:
    value = int(hashlib.sha256(family.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if value < 80 else "validation" if value < 90 else "test"


def array(path: Path) -> Iterator[dict]:
    """Yield objects from a top-level JSON array without retaining the array."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as file:
        data = ""
        pos = 0
        opened = False
        ended = False
        while True:
            chunk = file.read(65536)
            if chunk:
                data += chunk
            elif pos >= len(data):
                break
            while True:
                while pos < len(data) and (data[pos].isspace() or data[pos] == ","):
                    pos += 1
                if not opened:
                    if pos == len(data):
                        break
                    if data[pos] != "[":
                        raise ValueError("expected a top-level JSON array")
                    opened = True
                    pos += 1
                    continue
                if pos == len(data):
                    break
                if data[pos] == "]":
                    pos += 1
                    ended = True
                    if data[pos:].strip():
                        raise ValueError("trailing data after JSON array")
                    break
                try:
                    row, end = decoder.raw_decode(data, pos)
                except json.JSONDecodeError:
                    if len(data) - pos > 16 * 1024 * 1024:
                        raise ValueError("JSON array item exceeds 16 MiB")
                    break
                if not isinstance(row, dict):
                    raise ValueError("JSON array entries must be objects")
                yield row
                pos = end
                if pos > 131072:
                    data = data[pos:]
                    pos = 0
            if ended:
                break
            if not chunk:
                break
            if pos > 0:
                data = data[pos:]
                pos = 0
        if not opened or not ended:
            raise ValueError("truncated JSON array")


def _safe(row: dict) -> bool:
    if row.get("lang") != "en" or row.get("review_result") is not True:
        return False
    if row.get("deleted") is not False or row.get("synthetic") is not False:
        return False
    labels = row.get("labels") or {}
    for key in ("pii", "spam"):
        value = labels.get(key)
        if isinstance(value, dict) and value.get("value", 0) > 0:
            return False
    return not sensitive(str(row.get("text", "")))


def _oasst(node: dict, path: tuple[dict, ...] = ()) -> Iterator[list[dict]]:
    current = path + (node,)
    replies = node.get("replies") or []
    if replies:
        for child in replies:
            yield from _oasst(child, current)
    elif node.get("role") == "assistant":
        yield list(current)


def _sim(text: str) -> int:
    words = WORD.findall(norm(text))
    if len(words) < 20:
        return -1
    votes = [0] * 63
    for word in words:
        gram = word.encode("utf-8")
        value = int.from_bytes(hashlib.blake2b(gram, digest_size=8).digest(), "big")
        for bit in range(63):
            votes[bit] += 1 if value >> bit & 1 else -1
    return sum((1 << bit) for bit, value in enumerate(votes) if value >= 0)


def _near(db: sqlite3.Connection, value: int, doc: str, family: str,
          text: str) -> tuple[set[str], set[tuple[str, str]]]:
    if value < 0:
        return set(), set()
    found = set()
    for band in range(4):
        part = value >> (band * 16) & 65535
        found.update(db.execute(
            "select distinct sig,doc,family from near where band=? and val=? limit 128", (band, part)))
    ids = set()
    links = set()
    words = set(WORD.findall(norm(text)))
    for other, owner, group in found:
        if (value ^ other).bit_count() <= 3:
            prior = db.execute("select text from neartext where doc=?", (owner,)).fetchone()
            if not prior:
                continue
            old = set(WORD.findall(norm(prior[0])))
            if not old or len(words & old) / len(words | old) < 0.9:
                continue
            ids.add(owner)
            if group != family:
                links.add(tuple(sorted((group, family))))
    for band in range(4):
        part = value >> (band * 16) & 65535
        db.execute("insert into near values(?,?,?,?,?)", (band, part, value, doc, family))
    db.execute("insert into neartext values(?,?)", (doc, text))
    return ids, links


def _valid(rows: list[dict]) -> bool:
    if len(rows) < 2 or len(rows) > 64:
        return False
    if any(row.get("role") not in {"prompter", "user", "assistant"} or
           not str(row.get("text", "")).strip() or sensitive(str(row.get("text", "")))
           for row in rows):
        return False
    if not any(row.get("role") == "assistant" for row in rows):
        return False
    return True


def _episode(name: str, source: str, doc: str, family: str, kind: str,
             rows: list[dict], license: str, synthetic: bool = False,
             templ: str | None = None) -> dict:
    messages = [{"role": row["role"], "text": row["text"]} for row in rows]
    text = "\n".join(row["role"] + ": " + row["text"] for row in messages)
    first = next((row["text"] for row in messages if row["role"] == "user"), text)
    fam = family or source + ":" + digest(template(first))
    review = "oasst-filtered" if source == "oasst1" else "taskmaster-source" if source.startswith("taskmaster") else "project-authored"
    return {"id": name, "source": source, "doc": doc, "family": fam,
            "template": digest(templ if templ is not None else template(first)),
            "split": split(fam), "kind": kind,
            "synthetic": synthetic, "review": review, "license": license, "digest": digest(text),
            "raw": sum(len(row["text"].encode("utf-8")) for row in messages),
            "turns": len(messages), "messages": messages}


def _clean(rows: list[dict]) -> list[dict] | None:
    if not _valid(rows):
        return None
    out = []
    last = None
    for row in rows:
        role = "user" if row.get("role") == "prompter" else "assistant"
        if role == last:
            return None
        last = role
        out.append({"role": role, "text": row["text"]})
    return out


def _assign(families: Counter, sources: dict[str, Counter], links: set[tuple[str, str]]) -> dict[str, str]:
    names = ("train", "validation", "test")
    goals = {"train": 0.8, "validation": 0.1, "test": 0.1}
    parents = {family: family for family in families}

    def find(family: str) -> str:
        while parents[family] != family:
            parents[family] = parents[parents[family]]
            family = parents[family]
        return family

    for left, right in links:
        one, two = find(left), find(right)
        if one != two:
            top, child = sorted((one, two))
            parents[child] = top
    weights = Counter()
    kinds = defaultdict(Counter)
    totals = Counter()
    for family, count in families.items():
        root = find(family)
        weights[root] += count
        kinds[root].update(sources[family])
        totals.update(sources[family])
    loads = {name: Counter() for name in names}
    roots = {}
    ordered = sorted(weights, key=lambda key: (-weights[key], hashlib.sha256(key.encode()).digest()))
    for root in ordered:
        def cost(name: str) -> float:
            return max((loads[name][source] / max(totals[source] * goals[name], 1)
                        for source in kinds[root]), default=0.0)
        place = min(enumerate(names), key=lambda pair: (cost(pair[1]), pair[0]))[1]
        roots[root] = place
        loads[place].update(kinds[root])
    return {family: roots[find(family)] for family in families}


def build(tree: Path, selfdata: Path, woz: Path, seed: Path,
          output: Path, pattern: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, scratch = tempfile.mkstemp(prefix="dedupe-", suffix=".sqlite", dir=output.parent)
    os.close(fd)
    fd, spool = tempfile.mkstemp(prefix="corpus-", suffix=".jsonl", dir=output.parent)
    os.close(fd)
    db = sqlite3.connect(scratch)
    db.execute("create table seen(kind text, value text, primary key(kind,value))")
    db.execute("create table near(band integer, val integer, sig integer, doc text, family text)")
    db.execute("create table neartext(doc text primary key,text text)")
    db.execute("create table sources(source text,doc text,primary key(source,doc))")
    db.execute("create index nearkey on near(band,val)")
    stat = Counter()
    lengths = []
    families = Counter()
    groups = defaultdict(Counter)
    templates = {}
    links = set()
    with Path(spool).open("w", encoding="utf-8") as sink:
        def put(row: dict) -> None:
            text = "\n".join(msg["role"] + ": " + msg["text"] for msg in row["messages"])
            raw = digest(text)
            flat = digest(norm(text))
            if db.execute("select 1 from seen where kind='raw' and value=?", (raw,)).fetchone():
                stat["exact"] += 1
                return
            if db.execute("select 1 from seen where kind='norm' and value=?", (flat,)).fetchone():
                stat["normalized"] += 1
                return
            db.execute("insert into seen values('raw',?)", (raw,))
            db.execute("insert into seen values('norm',?)", (flat,))
            family = row["family"]
            ids, edges = _near(db, _sim(text), raw, family, text)
            links.update(edges)
            row["near"] = bool(ids)
            row["matches"] = sorted(ids)
            stat["near"] += int(bool(ids))
            row["digest"] = raw
            row["raw"] = sum(len(msg["text"].encode("utf-8")) for msg in row["messages"])
            db.execute("insert or ignore into sources values(?,?)", (row["source"], row["doc"]))
            families[family] += 1
            groups[family][row["source"]] += 1
            prior = templates.setdefault(row["template"], family)
            if prior != family:
                raise RuntimeError("one template maps to multiple source families")
            stat[row["source"]] += 1
            stat["episodes"] += 1
            stat["turns"] += row["turns"]
            stat["bytes"] += row["raw"]
            stat["bytes:" + row["source"]] += row["raw"]
            lengths.append((row["turns"], row["raw"]))
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

        with gzip.open(tree, "rt", encoding="utf-8") as file:
            for line in file:
                root = json.loads(line)
                for path in _oasst(root["prompt"]):
                    if not _valid(path) or any(not _safe(row) for row in path):
                        stat["oasst_filtered"] += 1
                        continue
                    if any(row.get("role") == "assistant" and
                           (row.get("rank") != 0 or
                            ((row.get("labels") or {}).get("quality", {}).get("value", 0) < 0.75))
                           for row in path):
                        stat["oasst_quality"] += 1
                        continue
                    clean = _clean(path)
                    if clean is None:
                        stat["oasst_filtered"] += 1
                        continue
                    first = next(row["text"] for row in clean if row["role"] == "user")
                    fam = "oasst:" + digest(template(first))
                    row = _episode("oa:" + root["message_tree_id"] + ":" + path[-1]["message_id"],
                                   "oasst1", root["message_tree_id"], fam, "human", clean,
                                   "Apache-2.0")
                    put(row)

        for path, tag in ((selfdata, "self"), (woz, "woz")):
            for doc in array(path):
                rows = [{"role": turn["speaker"].lower(), "text": turn["text"]}
                        for turn in doc.get("utterances", [])]
                if not _valid(rows):
                    stat["taskmaster_filtered"] += 1
                    continue
                family = "tm1:" + tag + ":" + str(doc.get("instruction_id", "unknown"))
                row = _episode("tm:" + tag + ":" + str(doc["conversation_id"]),
                               "taskmaster1-" + tag, doc["conversation_id"], family,
                           "human", rows, "CC-BY-4.0", False, family)
                put(row)

        with seed.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                doc = json.loads(line)
                rows = doc["messages"]
                if len(rows) < 2 or len(rows) > 64 or any(
                        row.get("role") not in {"user", "assistant", "tool"} or
                        not isinstance(row.get("text"), str) or not row["text"].strip() or
                        sensitive(row["text"]) for row in rows):
                    stat["seed_filtered"] += 1
                    continue
                family = "project:" + doc["family"]
                row = _episode("seed:" + doc["id"], "project", doc["id"], family,
                               "authored", rows, "project-authored", True,
                               "authored:" + doc["family"])
                put(row)
        if pattern is not None:
            with pattern.open("r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    doc = json.loads(line)
                    for index, values in enumerate(doc["cases"]):
                        rows = [{"role": row["role"], "text": row["text"].format_map(values)}
                                for row in doc["messages"]]
                        if any(row["role"] not in {"user", "assistant"} or
                               not row["text"].strip() or sensitive(row["text"]) for row in rows):
                            stat["pattern_filtered"] += 1
                            continue
                        family = "project:controlled:" + doc["family"]
                        name = "pattern:" + doc["id"] + ":" + str(index)
                        row = _episode(name, "project", name, family, "controlled", rows,
                                       "project-authored", True, "controlled:" + doc["family"])
                        put(row)
        db.commit()
        stat["sourcecount"] = db.execute("select count(*) from sources").fetchone()[0]
    db.close()
    Path(scratch).unlink(missing_ok=True)
    assigns = _assign(families, groups, links)
    stat["nearlinks"] = len(links)
    fd, final = tempfile.mkstemp(prefix="split-", suffix=".jsonl", dir=output.parent)
    os.close(fd)
    with Path(spool).open("r", encoding="utf-8") as source, Path(final).open("w", encoding="utf-8") as sink:
        for line in source:
            row = json.loads(line)
            row["split"] = assigns[row["family"]]
            stat[row["split"]] += 1
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    Path(spool).unlink(missing_ok=True)
    os.replace(final, output)
    lengths.sort()
    turns = [row[0] for row in lengths]
    raw = [row[1] for row in lengths]
    stat["families"] = len(families)
    stat["templates"] = len(templates)
    turns.sort()
    raw.sort()
    stat["turnmin"] = min(turns, default=0)
    stat["turn50"] = turns[len(turns) // 2] if turns else 0
    stat["turn90"] = turns[int((len(turns) - 1) * 0.9)] if turns else 0
    stat["turnmax"] = max(turns, default=0)
    stat["bytemin"] = min(raw, default=0)
    stat["byte50"] = raw[len(raw) // 2] if raw else 0
    stat["byte90"] = raw[int((len(raw) - 1) * 0.9)] if raw else 0
    stat["bytemax"] = max(raw, default=0)
    return dict(stat)


def audit(path: Path) -> dict:
    fd, scratch = tempfile.mkstemp(prefix="audit-", suffix=".sqlite", dir=path.parent)
    os.close(fd)
    db = sqlite3.connect(scratch)
    db.execute("create table seen(kind text,value text,split text,primary key(kind,value))")
    db.execute("create table ids(value text primary key)")
    db.execute("create table families(value text primary key,split text)")
    db.execute("create table templates(value text primary key,split text)")
    db.execute("create table docs(value text primary key,split text)")
    groups = Counter()
    raw = turns = 0
    exact = normalized = nearcross = 0
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            try:
                db.execute("insert into ids values(?)", (row["id"],))
            except sqlite3.IntegrityError as exc:
                raise ValueError("duplicate episode ID") from exc
            prior = db.execute("select split from families where value=?", (row["family"],)).fetchone()
            if prior and prior[0] != row["split"]:
                raise ValueError("template family crossed split")
            db.execute("insert or ignore into families values(?,?)", (row["family"], row["split"]))
            prior = db.execute("select split from templates where value=?", (row["template"],)).fetchone()
            if prior and prior[0] != row["split"]:
                raise ValueError("template crossed split")
            db.execute("insert or ignore into templates values(?,?)", (row["template"], row["split"]))
            db.execute("insert into docs values(?,?)", (row["digest"], row["split"]))
            for other in row.get("matches", []):
                prior = db.execute("select split from docs where value=?", (other,)).fetchone()
                if prior and prior[0] != row["split"]:
                    nearcross += 1
            text = "\n".join(msg["role"] + ": " + msg["text"] for msg in row["messages"])
            for kind, value in (("raw", digest(text)), ("norm", digest(norm(text)))):
                prior = db.execute("select split from seen where kind=? and value=?", (kind, value)).fetchone()
                if prior and prior[0] != row["split"]:
                    if kind == "raw":
                        exact += 1
                    else:
                        normalized += 1
                else:
                    db.execute("insert or ignore into seen values(?,?,?)", (kind, value, row["split"]))
            groups[row["split"]] += 1
            raw += sum(len(msg["text"].encode("utf-8")) for msg in row["messages"])
            turns += row["turns"]
    count = db.execute("select count(*) from families").fetchone()[0]
    tcount = db.execute("select count(*) from templates").fetchone()[0]
    db.close()
    Path(scratch).unlink(missing_ok=True)
    return {"episodes": sum(groups.values()), "families": count, "templates": tcount,
            "splits": dict(groups),
            "raw": raw, "turns": turns, "exactleak": exact,
            "normleak": normalized, "nearleak": nearcross}
