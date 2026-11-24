"""Source-balanced held-out assistant prediction measurement."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import resource
import struct
import time
from pathlib import Path
from typing import Any

import torch

from research.foundation.model import Foundation
from research.foundation import trainer
from research.prototype import trainer as guard


HERE = Path(__file__).resolve().parent
ROOT = trainer.DATA
SOURCES = ("oasst1", "taskmaster1-self", "taskmaster1-woz")


def digest(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def sample(root: Path, count: int, seed: int, context: int,
           split: str = "validation") -> list[dict]:
    if count < len(SOURCES) or count % len(SOURCES):
        raise ValueError("sample count must split evenly across sources")
    if split not in {"validation", "test"}:
        raise ValueError("response probe requires validation or test data")
    stem = "valid0" if split == "validation" else "test0"
    path = root / (stem + ".jsonl")
    groups = {name: [] for name in SOURCES}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            doc = json.loads(line)
            source = doc["source"]
            if source not in groups:
                continue
            spans = [span for span in doc["targets"]
                     if 1 <= span[1] - span[0] <= context // 2]
            if spans:
                groups[source].append({"source": source, "family": doc["family"],
                                       "doc": doc["doc"], "start": doc["start"],
                                       "tokens": doc["tokens"], "span": spans[0]})
    rng = random.Random(seed)
    result = []
    each = count // len(SOURCES)
    for source in SOURCES:
        rows = groups[source]
        rng.shuffle(rows)
        used = set()
        chosen = []
        for row in rows:
            if row["doc"] in used:
                continue
            used.add(row["doc"])
            chosen.append(row)
            if len(chosen) == each:
                break
        if len(chosen) != each:
            raise ValueError(f"not enough distinct validation episodes for {source}")
        result.extend(chosen)
    rng.shuffle(result)
    return result


def measure(model: Foundation, root: Path, rows: list[dict], context: int,
            deadline: float, split: str) -> dict:
    totals = {name: {"loss": 0.0, "correct": 0, "tokens": 0,
                     "exact": 0, "turns": 0} for name in (*SOURCES, "all")}
    started = time.monotonic()
    model.eval()
    stem = "valid0" if split == "validation" else "test0"
    with (root / (stem + ".bin")).open("rb") as file, torch.no_grad():
        for row in rows:
            if time.monotonic() >= deadline:
                raise TimeoutError("held-out response probe exceeded its wall-time budget")
            file.seek(row["start"] * 2)
            raw = file.read(row["tokens"] * 2)
            if len(raw) != row["tokens"] * 2:
                raise ValueError("validation shard ended inside a selected episode")
            ids = list(struct.unpack("<" + "H" * row["tokens"], raw))
            left, right = row["span"]
            size = right - left + 1
            begin = max(row["start"], left - (context - size))
            end = right + 1
            view = ids[begin - row["start"]:end - row["start"]]
            if len(view) < 2 or len(view) - 1 > context:
                raise ValueError("selected response does not fit the evaluation context")
            inputs = torch.tensor(view[:-1], dtype=torch.long)[None, :]
            target = torch.tensor(view[1:], dtype=torch.long)
            logits = model(inputs)[0]
            first = left - begin - 1
            last = right - begin - 1
            picks = logits[first:last + 1]
            labels = target[first:last + 1]
            loss = torch.nn.functional.cross_entropy(picks, labels, reduction="sum")
            guess = picks.argmax(dim=-1)
            correct = int((guess == labels).sum())
            exact = int(correct == len(labels))
            for name in (row["source"], "all"):
                totals[name]["loss"] += float(loss)
                totals[name]["correct"] += correct
                totals[name]["tokens"] += len(labels)
                totals[name]["exact"] += exact
                totals[name]["turns"] += 1
    result = {}
    for name, row in totals.items():
        result[name] = {"nll": row["loss"] / max(row["tokens"], 1),
                        "accuracy": row["correct"] / max(row["tokens"], 1),
                        "exact": row["exact"] / max(row["turns"], 1),
                        "tokens": row["tokens"], "turns": row["turns"]}
    result["seconds"] = time.monotonic() - started
    return result


def run(path: Path, count: int = 96, seed: int = 2041,
        budget: int = 60, split: str = "validation") -> dict:
    if guard.available_mib() < guard.MEMORY_GUARD_MIB:
        raise MemoryError("effective available memory below the existing 2560 MiB guard")
    guard.configure_torch()
    config = trainer.load()
    context = int(config["model"]["context_length"])
    rows = sample(ROOT, count, seed, context, split)
    selected = digest(rows)
    start = time.monotonic()
    deadline = start + budget
    torch.manual_seed(int(config["train"]["seed"]))
    base = Foundation(config["model"])
    before = measure(base, ROOT, rows, context, deadline, split)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != 1 or payload.get("meta", {}).get("split") != \
            trainer.pack(trainer.DATA, config)["split"]:
        raise ValueError("checkpoint does not match the frozen foundation training split")
    after = Foundation(config["model"])
    after.load_state_dict(payload["model"])
    final = measure(after, ROOT, rows, context, deadline, split)
    audit = json.loads((HERE.parent / "interaction" / "manifest7.json").read_text())
    data = json.loads((ROOT / "manifest.json").read_text())
    files = [row for row in data["files"] if row["split"] == split]
    return {"seed": seed, "count": count, "sources": list(SOURCES),
            "families": {name: len({row["family"] for row in rows
                                   if row["source"] == name}) for name in SOURCES},
            "dataset": digest({"split": audit["measure"]["hashes"][split],
                               "files": files, "manifest": trainer.file(ROOT / "manifest.json")}),
            "split": audit["measure"]["hashes"][split],
            "tokenizer": data["tokenizer"], "part": split,
            "selection": selected, "checkpoint": trainer.file(path),
            "updates": payload["step"], "before": before, "after": final,
            "timing": {"wall": time.monotonic() - start,
                       "rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                       "threads": torch.get_num_threads(), "device": "cpu"},
            "code": digest({"probe": trainer.file(HERE / "probe.py"),
                            "model": trainer.file(HERE / "model.py"),
                            "trainer": trainer.file(HERE / "trainer.py")}),
            "limits": "Teacher-forced next-token prediction over held-out assistant spans; exact turn match is not free generation or human-rated quality."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "proof.json")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run(args.checkpoint, split=args.split)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(result, file, sort_keys=True, indent=2)
        file.write("\n")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
