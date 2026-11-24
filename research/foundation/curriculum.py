"""Run one explicit, resumable curriculum stage on approved token shards."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from research.foundation import trainer


A = {"name": "language", "lang": 1.0, "sft": 0.0, "target": 60_000_000}
B = {"name": "interaction", "lang": 0.0, "sft": 1.0, "target": 16_200_000}
C = {"name": "memory", "lang": 0.0, "sft": 1.0, "target": None}


def plan(name: str, tokens: int, sequence: int = 125,
         segment: int = 90) -> dict[str, Any]:
    stages = {"a": A, "b": B, "c": C}
    if name not in stages:
        raise ValueError("stage must be a, b or c")
    if name == "c":
        raise ValueError("Stage C awaits a reviewed complete-episode memory dataset")
    if (not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 1
            or not isinstance(sequence, int) or isinstance(sequence, bool)
            or not 1 <= sequence < 1024
            or not isinstance(segment, int) or isinstance(segment, bool)
            or not 10 <= segment <= 110):
        raise ValueError("stage token target, sequence or segment wall is invalid")
    if tokens % sequence:
        raise ValueError("token target must divide evenly by sequence length")
    value = trainer.load()
    value["train"].update({"steps": tokens // sequence, "sequence": sequence,
                           "interval": 1000, "time": segment,
                           "lang": stages[name]["lang"], "sft": stages[name]["sft"]})
    return {"stage": name, "name": stages[name]["name"], "tokens": tokens,
            "config": value, "steps": tokens // sequence,
            "source": "rights-cleared Batch 1 shards"}


def write(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp, path)


def run(name: str, tokens: int, initial: Path, output: Path,
        report: Path, *, sequence: int = 125, segment: int = 90,
        wall: int = 850) -> dict[str, Any]:
    spec = plan(name, tokens, sequence, segment)
    config = spec["config"]
    if (not isinstance(wall, int) or isinstance(wall, bool) or not segment <= wall <= 3600
            or not initial.is_file() or output.resolve() == initial.resolve()):
        raise ValueError("curriculum wall, initial checkpoint or output path is invalid")
    root = trainer.DATA
    if not report.parent.exists() or not output.parent.exists():
        raise ValueError("checkpoint and report directories must already exist")
    meta = trainer.pack(root, config)
    source = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((trainer.HERE.parent / "interaction" / "manifest7.json").read_text())
    stage = {"schema": 1, "name": spec["name"], "stage": name,
             "target": tokens, "sequence": sequence, "initial": trainer.file(initial),
             "data": meta["dataset"], "split": meta["split"],
             "tokenizer": meta["tokenizer"], "config": trainer.digest(config),
             "source": trainer.file(root / "manifest.json"),
             "licenses": sorted({row["license"] for row in audit["sources"]["files"]}),
             "docs": audit["measure"]["docs"]["train"],
             "tokens": audit["measure"]["tokens"]["train"],
             "steps": config["train"]["steps"], "segments": []}
    if report.exists():
        saved = json.loads(report.read_text(encoding="utf-8"))
        check = {key: saved.get(key) for key in stage if key != "segments"}
        current = {key: stage[key] for key in stage if key != "segments"}
        if check != current or not output.exists():
            raise ValueError("existing curriculum report does not match this stage")
        stage = saved
    elif output.exists():
        stage["recovered"] = True
    deadline = time.monotonic() + wall
    start = time.monotonic()
    step = 0
    if output.exists():
        payload = torch.load(output, map_location="cpu", weights_only=True)
        step = int(payload["step"])
    while step < config["train"]["steps"] and time.monotonic() < deadline:
        result = trainer.run(config, root, output=output,
                             resume=output if output.exists() else None,
                             init=initial)
        current = result["steps"]
        if current <= step:
            raise RuntimeError("training segment made no progress")
        if not result["before"]["complete"] or not result["validation"]["complete"]:
            raise RuntimeError("segment evaluation was incomplete; refusing to report partial metrics")
        stage["segments"].append({"start": step, "end": current,
                                  "tokens": result["tokens"],
                                  "fit": result["fit"],
                                  "seconds": result["seconds"],
                                  "speed": result["tokens"] / max(0.001, result["fit"]),
                                  "before": result["before"],
                                  "after": result["validation"],
                                  "peak": result["peakrss"],
                                  "memory": result["memory"],
                                  "hashes": result["hashes"],
                                  "losses": result["train"]})
        step = current
        stage["complete"] = step >= config["train"]["steps"]
        stage["done"] = step * sequence
        stage["seconds"] = time.monotonic() - start
        stage["peak"] = max(item["peak"] for item in stage["segments"])
        stage["speed"] = sum(item["tokens"] for item in stage["segments"]) / max(
            0.001, sum(item["fit"] for item in stage["segments"]))
        stage["checkpoint"] = str(output)
        stage["weight"] = trainer.file(output)
        write(report, stage)
        print(json.dumps({"stage": name, "step": step,
                          "tokens": stage["done"], "speed": stage["speed"],
                          "segment": result["seconds"],
                          "complete": stage["complete"]}), flush=True)
        if not result["interrupted"]:
            break
    return stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("a", "b", "c"), required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--wall", type=int, default=850)
    args = parser.parse_args()
    result = run(args.stage, args.tokens, args.initial, args.output,
                 args.report, wall=args.wall)
    print(json.dumps({key: result[key] for key in
                      ("stage", "target", "done", "complete", "seconds", "speed",
                       "peak", "checkpoint", "weight")}, sort_keys=True))


if __name__ == "__main__":
    main()
