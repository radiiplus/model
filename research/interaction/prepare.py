"""Build the reviewed interaction corpus and immutable tokenizer shards."""

from __future__ import annotations

import hashlib
import json
import resource
import shutil
import signal
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from research.interaction import corpus, shard
from research.interaction.tokenizer import Tokenizer
from research.prototype.trainer import MEMORY_GUARD_MIB, available_mib, configure_torch


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
READY = DATA / "ready7"
MANIFEST = HERE / "manifest7.json"
LIMIT = 120


def rows(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def texts(path: Path):
    for row in rows(path):
        if row["split"] == "train":
            for message in row["messages"]:
                yield message["text"]


def measure(path: Path, token: Tokenizer) -> dict:
    sizes = Counter()
    unique = set()
    hashes = {name: hashlib.sha256() for name in ("train", "validation", "test")}
    for row in rows(path):
        split = row["split"]
        canonical = (json.dumps(row, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")) + "\n").encode("utf-8")
        hashes[split].update(canonical)
        sizes["docs:" + split] += 1
        for message in row["messages"]:
            raw = message["text"].encode("utf-8")
            ids = token.encode(message["text"])
            sizes["bytes:" + split] += len(raw)
            sizes["tokens:" + split] += len(ids)
            unique.update(ids)
    return {"docs": {name: sizes["docs:" + name] for name in hashes},
            "bytes": {name: sizes["bytes:" + name] for name in hashes},
            "tokens": {name: sizes["tokens:" + name] for name in hashes},
            "hashes": {name: hashes[name].hexdigest() for name in hashes},
            "used": len(unique)}


def sources() -> dict:
    data = json.loads((HERE / "source.json").read_text(encoding="utf-8"))
    found = []
    for item in data["items"]:
        for file in item["files"]:
            path = DATA / file["path"].split("/", 1)[1] if file["path"].startswith("data/") else HERE / file["path"]
            found.append({"source": item["name"], "revision": item["revision"],
                          "path": file["path"], "sha256": file["sha256"],
                          "bytes": file["bytes"], "license": item["license"]})
            if not path.is_file() or path.stat().st_size != file["bytes"]:
                raise ValueError("source file missing or size changed: " + file["path"])
            if file.get("sha256") and _hash(path) != file["sha256"]:
                raise ValueError("source hash mismatch: " + file["path"])
    return {"retrieved": data.get("retrieved", "see source manifest"), "files": found}


def _hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1048576)
            if not chunk:
                break
            value.update(chunk)
    return value.hexdigest()


def run() -> dict:
    if READY.exists() or MANIFEST.exists():
        raise FileExistsError("corpus output already exists; choose a new immutable version")
    if available_mib() < MEMORY_GUARD_MIB:
        raise MemoryError("effective available memory is below the existing 2560 MiB guard")
    configure_torch()
    began = time.monotonic()
    signal.alarm(LIMIT)
    stage = Path(tempfile.mkdtemp(prefix="stage-", dir=DATA))
    try:
        cur = stage / "corpus.jsonl"
        stats = corpus.build(DATA / "oasst1-trees.gz", DATA / "taskmaster1-self.json",
                             DATA / "taskmaster1-woz.json", HERE / "seed.jsonl", cur,
                             HERE / "patterns.jsonl")
        check = corpus.audit(cur)
        if check["exactleak"] or check["normleak"] or check["nearleak"]:
            raise ValueError("content contamination crossed a split")
        token = Tokenizer.train(texts(cur), target=4096, budget=1048576)
        token.save(stage / "tokenizer.json")
        measured = measure(cur, token)
        packed = shard.write(cur, token, stage / "shards")
        targets = {}
        for name in ("train", "validation", "test"):
            targets[name] = sum(sum(batch["mask"]) for batch in shard.stream(
                stage / "shards", name, size=65536))
        created = datetime.now(timezone.utc).isoformat()
        metadata = {"version": 1, "created": created, "sources": sources(),
                    "filters": {"oasst": "English; reviewed; live; nonsynthetic; excludes PII/spam flags and sensitive patterns; assistant rank 0 and quality >= 0.75",
                                "taskmaster": "whole dialogues; e-mail, phone, payment-number, SSN, IP and street-address patterns excluded",
                                "project": "authored seed scenarios; explicitly marked synthetic"},
                    "curation": stats, "audit": check, "measure": measured,
                    "tokenizer": {"version": 1, "hash": token.digest,
                                  "target": token.vocab, "active": token.size,
                                  "sample": token.seen, "split": "train",
                                  "special": 8, "bytes": 256,
                                  "compression": "content byte/token counts are reported separately"},
                    "shards": packed, "targets": targets,
                    "units": {"docs": check["episodes"],
                              "sources": stats["sourcecount"],
                              "bytes": check["raw"],
                              "used": measured["used"],
                              "stored": packed["total"]["tokens"],
                              "presented": 0, "epochs": 0},
                    "limits": {"cpu": "CPU only; one Torch intra/inter-op thread",
                               "memorymib": MEMORY_GUARD_MIB, "wallseconds": LIMIT,
                               "corpusram": "source and corpus records stream; tokenizer sample capped at 1 MiB"},
                    "seconds": round(time.monotonic() - began, 3),
                    "rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
        with (stage / "manifest.json").open("x", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            file.write("\n")
        READY.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(READY)
        with MANIFEST.open("x", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, sort_keys=True, indent=2)
            file.write("\n")
        return metadata
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True, indent=2))
