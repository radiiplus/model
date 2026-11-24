"""Deterministic offline fitting from independently approved experience data."""

from __future__ import annotations

import hashlib
import json
import os
import random
import resource
import time
from pathlib import Path
from typing import Any

import torch

from research.foundation import trainer
from research.foundation.model import Foundation
from research.interaction.tokenizer import SPECIAL, Tokenizer
from research.prototype import trainer as guard


def seal(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as file:
        while True:
            part = file.read(1_048_576)
            if not part:
                break
            value.update(part)
    return value.hexdigest()


def code() -> str:
    paths = [Path(__file__), Path(__file__).with_name("model.py"),
             Path(__file__).with_name("trainer.py"),
             Path("research/interaction/compact.py"),
             Path("research/interaction/tokenizer.py"), Path(guard.__file__)]
    return hashlib.sha256("".join(seal(path) for path in paths).encode()).hexdigest()


def rows(source: Path) -> tuple[dict[str, Any], list[tuple[list[int], list[bool]]]]:
    if not source.is_file() or source.stat().st_size > 1_048_576:
        raise ValueError("approved dataset must be a local file no larger than 1 MiB")
    value = json.loads(source.read_text(encoding="utf-8"))
    if (not isinstance(value, dict) or value.get("schema") != 1
            or not isinstance(value.get("hash"), str)):
        raise ValueError("dataset is not an exported approved version")
    data = {key: item for key, item in value.items() if key not in {"schema", "hash"}}
    if hashlib.sha256(encode(data)).hexdigest() != value["hash"]:
        raise ValueError("approved dataset hash mismatch")
    valid = data.get("validation")
    report = valid.get("report") if isinstance(valid, dict) else None
    if (data.get("status") != "approved" or not isinstance(report, dict)
            or report.get("gate") != "independent-transfer-v1"
            or report.get("passed") is not True or report.get("safe") is not True):
        raise ValueError("dataset has no passing independent-transfer approval")
    train = data.get("train")
    if not isinstance(train, list) or not 1 <= len(train) <= 128:
        raise ValueError("approved dataset must contain 1..128 training rows")
    path = Path("research/interaction/data/ready7/tokenizer.json")
    token = Tokenizer.load(path)
    audit = json.loads(Path("research/interaction/manifest7.json").read_text())
    if token.digest != audit["tokenizer"]["hash"]:
        raise ValueError("training tokenizer does not match the frozen tokenizer manifest")
    result = []
    for row in train:
        if (not isinstance(row, dict) or set(row) != {"episode", "messages", "target"}
                or not isinstance(row["episode"], str) or not row["episode"]
                or not isinstance(row["messages"], list) or not row["messages"]
                or not isinstance(row["target"], str) or not row["target"]):
            raise ValueError("approved training row has an invalid schema")
        prompt = token.chat(row["messages"], bos=True, eos=False)
        prompt.append(SPECIAL["ASSISTANT"])
        target = token.encode(row["target"])
        ids = prompt + target + [SPECIAL["END"]]
        mask = [False] * len(prompt) + [True] * (len(target) + 1)
        result.append((ids, mask))
    return {"data": data, "token": token, "rows": result}, result


def encode(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def store(path: Path, value: dict[str, Any]) -> None:
    trainer.save(path, value)


def run(base: Path, data: Path, output: Path, *, steps: int,
        rate: float = 0.00003, seed: int = 2026, wall: float = 90.0,
        stop: int | None = None, resume: Path | None = None) -> dict[str, Any]:
    if (not isinstance(steps, int) or isinstance(steps, bool) or steps < 1
            or not isinstance(seed, int) or isinstance(seed, bool)
            or not isinstance(rate, (int, float)) or isinstance(rate, bool) or rate <= 0
            or not isinstance(wall, (int, float)) or isinstance(wall, bool) or not 0 < wall <= 120):
        raise ValueError("fit steps, seed, learning rate or wall budget is invalid")
    if stop is not None and (not isinstance(stop, int) or isinstance(stop, bool)
                             or not 1 <= stop <= steps):
        raise ValueError("segment stop must be in 1..steps")
    if not base.is_file() or not output.parent.exists():
        raise ValueError("base checkpoint and existing output directory are required")
    if resume is None and output.exists():
        raise FileExistsError(output)
    if guard.available_mib() < guard.MEMORY_GUARD_MIB:
        raise MemoryError("effective available memory below the existing 2560 MiB guard")
    guard.configure_torch()
    random.seed(seed)
    torch.manual_seed(seed)
    loaded, train = rows(data)
    token = loaded["token"]
    mark = hashlib.sha256(encode(loaded["data"])).hexdigest()
    origin = seal(base)
    source = torch.load(base, map_location="cpu", weights_only=True)
    config = source.get("config")
    if not isinstance(config, dict) or "model" not in config or "model" not in source:
        raise ValueError("base checkpoint lacks a Foundation config or weights")
    model = Foundation(config["model"])
    model.load_state_dict(source["model"])
    if any(len(ids) > model.context for ids, _ in train):
        raise ValueError("an approved training example exceeds model context")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(rate), weight_decay=0.1)
    meta = {"base": origin, "data": mark, "token": token.digest,
            "code": code(), "shape": trainer.digest(config["model"]),
            "rate": float(rate), "seed": seed, "steps": steps}
    start = 0
    cursor = random.Random(seed + 1)
    if resume is not None:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state.get("meta") != meta:
            raise ValueError("resume base, data, tokenizer, code or settings hash mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        random.setstate(state["python"])
        torch.set_rng_state(state["torch"])
        cursor.setstate(state["cursor"])
        start = int(state["step"])
        if start > steps:
            raise ValueError("resume step exceeds requested training steps")
        output = resume
    elif output.exists():
        raise FileExistsError(output)
    deadline = time.monotonic() + float(wall)
    swap = guard.swap_counters()
    started = time.monotonic()
    logs = []
    end = steps if stop is None else stop
    for index in range(start, end):
        if time.monotonic() >= deadline:
            break
        if index % 10 == 0:
            if guard.available_mib() < guard.MEMORY_GUARD_MIB:
                raise MemoryError("effective available memory fell below the 2560 MiB guard")
            now = guard.swap_counters()
            if swap is not None and now is not None and any(a > b for a, b in zip(now, swap)):
                raise MemoryError("swap activity detected during offline fitting")
            swap = now
        pick = cursor.randrange(len(train))
        ids, mask = train[pick]
        tokens = torch.tensor(ids, dtype=torch.long)[None, :]
        targets = torch.tensor(mask, dtype=torch.bool)[None, :]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = model.losses(tokens, targets)["interaction"]
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        logs.append({"step": index + 1, "loss": float(loss.detach()),
                     "grad": float(norm), "row": pick})
        if time.monotonic() >= deadline:
            break
    done = start + len(logs)
    state = {"schema": 1, "arch": 1, "config": config, "meta": meta,
             "step": done, "model": model.state_dict(),
             "optimizer": optimizer.state_dict(), "python": random.getstate(),
             "torch": torch.get_rng_state(), "cursor": cursor.getstate(),
             "rows": len(train)}
    store(output, state)
    return {"step": done, "start": start, "interrupted": done < steps,
            "parameters": model.params, "trainable": sum(p.numel() for p in model.parameters()
                                                        if p.requires_grad),
            "seconds": time.monotonic() - started,
            "peak": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "device": "cpu", "threads": torch.get_num_threads(),
            "losses": logs, "hashes": meta, "checkpoint": str(output),
            "bytes": output.stat().st_size}


__all__ = ["run", "rows"]
