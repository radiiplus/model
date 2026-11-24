"""One-seed matched-token and matched-compute arm comparison."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import resource
import time
from pathlib import Path
from typing import Any

import torch

from research.foundation import trainer
from research.foundation.model import Foundation
from research.foundation.recur import Refiner, transfer
from research.prototype import trainer as guard


HERE = Path(__file__).resolve().parent
SET = HERE / "recur.json"


def digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def code() -> str:
    paths = (HERE / "model.py", HERE / "recur.py", HERE / "compare.py",
             HERE / "trainer.py")
    return digest({"files": {str(path): trainer.file(path) for path in paths},
                   "trunk": trainer.code()})


def score(rows: list[dict[str, torch.Tensor]], weights: list[float], train: dict) -> tuple[
        torch.Tensor, list[dict[str, float]]]:
    total = None
    report = []
    for row, weight in zip(rows, weights):
        loss = float(train["lang"]) * row["language"] + float(train["sft"]) * row["interaction"]
        total = loss * weight if total is None else total + loss * weight
        report.append({"language": float(row["language"].detach()),
                       "interaction": float(row["interaction"].detach()),
                       "answers": int(row["answers"])})
    return total, report


def rows(model: Foundation, tokens: torch.Tensor, mask: torch.Tensor,
         steps: int, deep: list[float]) -> list[dict[str, torch.Tensor]]:
    if isinstance(model, Refiner):
        return model.losses(tokens, mask, steps, deep)
    return [model.losses(tokens, mask)]


def stamp(value: Any, tokens: torch.Tensor, mask: torch.Tensor) -> None:
    value.update(json.dumps([tokens.tolist(), mask.tolist()], separators=(",", ":")).encode())


def fit(model: Foundation, name: str, updates: int, steps: int,
        deep: list[float], settings: dict, meta: dict, deadline: float,
        path: Path | None = None, resume: Path | None = None,
        stop: int | None = None) -> dict:
    train = {"batch": settings["batch"], "sequence": settings["sequence"]}
    opt = torch.optim.AdamW(model.parameters(), lr=float(settings["rate"]),
                            weight_decay=float(settings["penalty"]))
    feed = trainer.Sampler(trainer.DATA, "train", int(settings["seed"]) + 1)
    report = []
    marks = []
    apps = 0
    targets = 0
    start = 0
    key = digest({"name": name, "steps": steps, "deep": deep, "settings": settings,
                  "meta": meta, "model": model.params, "code": code()})
    if resume is not None:
        data = torch.load(resume, map_location="cpu", weights_only=True)
        if data.get("schema") != 1 or data.get("key") != key:
            raise ValueError("recurrent resume checkpoint identity mismatch")
        model.load_state_dict(data["model"])
        opt.load_state_dict(data["optimizer"])
        feed = trainer.Sampler(trainer.DATA, "train", int(settings["seed"]) + 1,
                               data["sampler"])
        random.setstate(data["python"])
        torch.set_rng_state(data["torch"])
        report = data["report"]
        marks = data["marks"]
        apps = int(data["apps"])
        targets = int(data["targets"])
        start = int(data["step"])
        if path is None:
            path = resume
    elif path is not None and path.exists():
        raise FileExistsError(path)
    end = updates if stop is None else stop
    if start > updates or not start < end <= updates:
        raise ValueError("invalid segment stop or total update count")
    started = time.monotonic()
    swap = guard.swap_counters()
    for index in range(start, end):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"experiment wall-time budget expired before update {index}")
        if guard.available_mib() < guard.MEMORY_GUARD_MIB:
            raise MemoryError(f"effective available memory fell below guard at update {index}")
        current = guard.swap_counters()
        if swap is not None and current is not None and any(
                now > prior for now, prior in zip(current, swap)):
            raise MemoryError(f"swap activity detected at update {index}")
        swap = current
        opt.zero_grad(set_to_none=True)
        stamps = hashlib.sha256()
        stages = []
        amount = 0
        for _ in range(int(settings["accum"])):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"experiment wall-time budget expired during update {index}")
            tokens, mask = trainer.batch(feed, int(train["batch"]), int(train["sequence"]))
            stamp(stamps, tokens, mask)
            values = rows(model, tokens, mask, steps, deep)
            weights = deep if isinstance(model, Refiner) else [1.0]
            objective, measure = score(values, weights, settings)
            (objective / int(settings["accum"])).backward()
            stages.append(measure)
            amount += int(values[-1]["tokens"])
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["clip"]))
        opt.step()
        targets += amount
        count = len(model.blocks) * steps * int(settings["batch"]) * int(settings["sequence"])
        apps += len(model.blocks) * steps * int(settings["accum"])
        marks.append(stamps.hexdigest())
        report.append({"step": index + 1, "stages": stages, "tokens": amount,
                       "grad": float(norm), "rate": float(settings["rate"]),
                       "calls": len(model.blocks) * steps * int(settings["accum"]),
                       "positions": count * int(settings["accum"])})
    if path is not None:
        trainer.save(path, {"schema": 1, "key": key, "step": end,
                            "model": model.state_dict(), "optimizer": opt.state_dict(),
                            "sampler": feed.state(), "python": random.getstate(),
                            "torch": torch.get_rng_state(), "report": report,
                            "marks": marks, "apps": apps, "targets": targets})
    elapsed = time.monotonic() - started
    return {"name": name, "updates": updates, "steps": steps,
            "parameters": model.params, "blocks": len(model.blocks),
            "calls": apps, "positions": apps * int(train["batch"]) * int(train["sequence"]),
            "tokens": targets, "seconds": elapsed,
            "speed": targets / max(elapsed, 1e-12),
            "blockrate": apps * int(train["batch"]) * int(train["sequence"]) / max(elapsed, 1e-12),
            "feed": marks, "prefix": digest(marks), "train": report,
            "identity": key, "resume": start, "interrupted": end < updates}


def assess(model: Foundation, name: str, steps: int, deep: list[float],
           settings: dict, deadline: float) -> dict:
    size = int(settings["batch"])
    sequence = int(settings["sequence"])
    sampler = trainer.Sampler(trainer.DATA, "validation", int(settings["seed"]) + 2)
    count = int(settings["eval"])
    sums = [{"language": 0.0, "interaction": 0.0, "tokens": 0, "answers": 0}
            for _ in range(steps)]
    seen = 0
    feed = hashlib.sha256()
    model.eval()
    with torch.no_grad():
        while seen < count:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"experiment wall-time budget expired during {name} evaluation")
            tokens, mask = trainer.batch(sampler, size, sequence)
            stamp(feed, tokens, mask)
            values = rows(model, tokens, mask, steps, deep)
            for slot, row in enumerate(values):
                length = int(row["tokens"])
                answers = int(row["answers"])
                sums[slot]["language"] += float(row["language"]) * length
                if answers:
                    sums[slot]["interaction"] += float(row["interaction"]) * answers
                    sums[slot]["answers"] += answers
                sums[slot]["tokens"] += length
            seen += int(values[0]["tokens"])
    for row in sums:
        row["language"] /= max(row["tokens"], 1)
        row["interaction"] = (row["interaction"] / row["answers"]
                               if row["answers"] else None)
    return {"name": name, "steps": steps, "stages": sums, "feed": feed.hexdigest()}


def run() -> dict:
    settings = json.loads(SET.read_text(encoding="utf-8"))
    config = trainer.load()
    model = config["model"]
    seed = int(settings["seed"])
    updates = int(settings["updates"])
    steps = int(settings["steps"])
    deep = [float(value) for value in settings["deep"]]
    if len(deep) != steps or abs(sum(deep) - 1.0) > 1e-8:
        raise ValueError("deep-supervision weights must match steps and sum to one")
    if guard.available_mib() < guard.MEMORY_GUARD_MIB:
        raise MemoryError("effective available memory below the existing 2560 MiB guard")
    guard.configure_torch()
    started = time.monotonic()
    deadline = started + int(settings["time"])
    torch.manual_seed(seed)
    initial = Foundation(model)
    recur = Refiner(model, int(settings["limit"]))
    transfer(recur, initial)
    check = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    exact = torch.equal(initial(check), recur(check, steps=1))
    if not exact:
        raise AssertionError("K=1 transfer is not exactly equal to the plain foundation")
    before = assess(initial, "plain", 1, [1.0], settings, deadline)
    meta = trainer.pack(trainer.DATA, config)
    same = copy.deepcopy(initial)
    plain = fit(same, "plain-token", updates, 1, [1.0], settings, meta, deadline)
    loop = copy.deepcopy(recur)
    recurloss = fit(loop, "recur-token", updates, steps, deep, settings, meta, deadline)
    wide = copy.deepcopy(initial)
    broad = fit(wide, "plain-compute", updates * steps, 1, [1.0], settings, meta,
                deadline)
    if plain["feed"] != recurloss["feed"]:
        raise AssertionError("matched-token arms received different data presentations")
    if plain["prefix"] != digest(broad["feed"][:updates]):
        raise AssertionError("matched-compute arm did not share the identical ordered prefix")
    start = time.monotonic()
    results = [assess(same, "plain-token", 1, [1.0], settings, deadline),
               assess(loop, "recur-k1-after-k2", 1, [1.0], settings, deadline),
               assess(loop, "recur-k2", steps, deep, settings, deadline),
               assess(wide, "plain-compute", 1, [1.0], settings, deadline)]
    validation = {row["name"]: row for row in results}
    return {"seed": seed, "model": model, "recur": settings,
            "parameters": {"plain": initial.params, "recur": recur.params,
                           "shared": sum(value.numel() for value in recur.blocks.parameters()),
                           "gates": recur.params - initial.params},
            "transfer": {"exact": exact, "limit": recur.limit,
                         "latent": list(recur(check, steps=1, trace=True)["states"][0][0].shape),
                         "answer": list(recur(check, steps=1, trace=True)["states"][0][1].shape)},
            "data": {"hashes": meta, "prefix": plain["feed"],
                     "validation": before["feed"], "unit":
                     int(settings["batch"]) * int(settings["sequence"]) * int(settings["accum"])},
            "hashes": {"data": meta["dataset"], "split": meta["split"],
                       "tokenizer": meta["tokenizer"], "config": trainer.file(SET),
                       "settings": digest(settings), "code": code()},
            "match": {"plain": plain, "loop": recurloss,
                      "equal": plain["tokens"] == recurloss["tokens"],
                      "identical": plain["feed"] == recurloss["feed"]},
            "cost": {"plain": broad, "loop": recurloss,
                     "equal": broad["positions"] == recurloss["positions"],
                     "ratio": broad["tokens"] / max(recurloss["tokens"], 1)},
            "validation": {"before": before, "after": validation},
            "timing": {"wall": time.monotonic() - started,
                       "eval": time.monotonic() - start,
                       "rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                       "threads": torch.get_num_threads(), "device": "cpu"},
            "code": code(), "limitation": "One seed and a short pilot; no capability or superiority claim."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "pilot.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(result, file, sort_keys=True, indent=2)
        file.write("\n")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
