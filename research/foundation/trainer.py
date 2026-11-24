"""Streaming AdamW training with exact shard cursors and resumable segments."""

from __future__ import annotations

import hashlib
import json
import os
import random
import resource
import signal
import time
from pathlib import Path
from typing import Any

import torch

from research.experiments import tiny_gpt
from research.interaction import shard
from research.prototype import trainer as guard
from research.foundation.model import Foundation


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "interaction" / "data" / "ready7" / "shards"
ARCH = 1


def file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            part = file.read(1048576)
            if not part:
                break
            value.update(part)
    return value.hexdigest()


def digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def code() -> str:
    paths = (HERE / "model.py", HERE / "trainer.py", Path(tiny_gpt.__file__).resolve(),
             Path(shard.__file__).resolve(), HERE.parent / "interaction" / "tokenizer.py",
             Path(guard.__file__).resolve())
    return digest({str(path): file(path) for path in paths})


class Sampler:
    def __init__(self, root: Path, split: str, seed: int, state: dict | None = None):
        self.root = root
        self.split = split
        self.cursor = {"shard": 0, "token": 0}
        self.ids: list[int] = []
        self.mask: list[bool] = []
        self.done = False
        self.rng = random.Random(seed)
        if state is not None:
            if state["split"] != split:
                raise ValueError("sampler split mismatch")
            self.cursor = dict(state["cursor"])
            self.ids = list(state["ids"])
            self.mask = list(state["mask"])
            self.done = bool(state["done"])
            self.rng.setstate(state["rng"])
        self.feed = shard.stream(self.root, self.split, self.cursor, size=4096)

    def state(self) -> dict:
        return {"split": self.split, "cursor": self.cursor, "ids": self.ids,
                "mask": self.mask, "done": self.done, "rng": self.rng.getstate()}

    def take(self, count: int) -> tuple[list[int], list[bool]]:
        while len(self.ids) < count and not self.done:
            try:
                row = next(self.feed)
            except StopIteration:
                self.done = True
                break
            self.ids.extend(row["ids"])
            self.mask.extend(row["mask"])
            self.cursor = row["cursor"]
        if len(self.ids) < count:
            raise ValueError("requested more tokens than remain in the shard")
        ids = self.ids[:count]
        mask = self.mask[:count]
        del self.ids[:count]
        del self.mask[:count]
        return ids, mask


def pack(root: Path, config: dict) -> dict:
    source = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = [row for row in source["files"] if row["split"] == "train"]
    if not rows:
        raise ValueError("training shards are empty")
    audit = json.loads((HERE.parent / "interaction" / "manifest7.json").read_text(encoding="utf-8"))
    if source["tokenizer"] != audit["tokenizer"]["hash"]:
        raise ValueError("dataset shard and frozen tokenizer hashes differ")
    verified = shard.check(root, source["tokenizer"])
    if verified["documents"] != source["total"]["documents"]:
        raise ValueError("dataset shard document count differs from its manifest")
    return {"dataset": digest({"split": audit["measure"]["hashes"]["train"],
                                "files": rows, "manifest": file(root / "manifest.json")}),
            "split": audit["measure"]["hashes"]["train"],
            "tokenizer": source["tokenizer"], "tokens": sum(row["tokens"] for row in rows),
            "model": digest(config["model"]), "run": digest(config),
            "config": file(HERE / "config.json"), "code": code()}


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("xb") as file:
        torch.save(data, file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def rate(step: int, train: dict) -> float:
    warm = int(train["warm"])
    decay = int(train["decay"])
    floor = float(train["floor"])
    if warm and step < warm:
        return max(floor, (step + 1) / warm)
    if step >= decay:
        return floor
    return max(floor, 1.0 - (step - warm) / max(1, decay - warm))


def batch(sampler: Sampler, size: int, sequence: int) -> tuple[torch.Tensor, torch.Tensor]:
    values, masks = sampler.take(size * (sequence + 1))
    tokens = torch.tensor(values, dtype=torch.long).reshape(size, sequence + 1)
    mask = torch.tensor(masks, dtype=torch.bool).reshape(size, sequence + 1)
    return tokens, mask


def evaluate(model: Foundation, root: Path, seed: int, train: dict,
             deadline: float) -> dict:
    count = int(train["eval"])
    size = int(train["batch"])
    sequence = int(train["sequence"])
    sampler = Sampler(root, "validation", seed, None)
    language = interaction = total = answers = 0.0
    seen = 0
    model.eval()
    with torch.no_grad():
        while seen < count and time.monotonic() < deadline:
            tokens, mask = batch(sampler, size, sequence)
            values = model.losses(tokens, mask)
            length = int(values["tokens"])
            selected = int(values["answers"])
            language += float(values["language"]) * length
            if selected:
                interaction += float(values["interaction"]) * selected
                answers += selected
            total += length
            seen += length
    return {"language": language / total if total else None,
            "interaction": interaction / answers if answers else None,
            "tokens": total, "answers": int(answers)}


def checkpoint(model: Foundation, opt: torch.optim.Optimizer, schedule: Any,
               sampler: Sampler, meta: dict, config: dict, step: int, seed: int) -> dict:
    return {"schema": 1, "arch": ARCH, "meta": meta, "config": config, "seed": seed,
            "step": step, "model": model.state_dict(), "optimizer": opt.state_dict(),
            "scheduler": schedule.state_dict(), "python": random.getstate(),
            "torch": torch.get_rng_state(), "sampler": sampler.state()}


def run(config: dict, root: Path = DATA, output: Path | None = None,
        resume: Path | None = None, stop: int | None = None,
        init: Path | None = None) -> dict:
    model = config["model"]
    train = config["train"]
    seed = int(train["seed"])
    steps = int(train["steps"])
    sequence = int(train["sequence"])
    if not 1 <= sequence < int(model["context_length"]):
        raise ValueError("training sequence must fit the configured model context")
    if steps < 1 or int(train["batch"]) < 1 or int(train["accum"]) < 1:
        raise ValueError("steps, batch and accumulation must be positive")
    if int(train["interval"]) < 2 or int(train["time"]) < 1:
        raise ValueError("checkpoint interval must be at least two and wall budget positive")
    if int(train["warm"]) < 0 or int(train["decay"]) <= int(train["warm"]):
        raise ValueError("decay must be greater than nonnegative warmup")
    if not 0.0 <= float(train["floor"]) <= 1.0:
        raise ValueError("minimum learning-rate ratio must be in 0..1")
    if float(train["rate"]) <= 0 or float(train["clip"]) <= 0:
        raise ValueError("learning rate and gradient clipping must be positive")
    if int(train["eval"]) < 1 or float(train["lang"]) < 0 or float(train["sft"]) < 0:
        raise ValueError("evaluation token count must be positive and loss weights nonnegative")
    if guard.available_mib() < guard.MEMORY_GUARD_MIB:
        raise MemoryError("effective available memory below the existing 2560 MiB guard")
    guard.configure_torch()
    torch.manual_seed(seed)
    random.seed(seed)
    meta = pack(root, config)
    if init is not None:
        init = Path(init)
        if not init.is_file():
            raise FileNotFoundError(init)
        meta["initial"] = file(init)
    else:
        meta["initial"] = None
    sampler = Sampler(root, "train", seed + 1)
    model = Foundation(model)
    if init is not None:
        source = torch.load(init, map_location="cpu", weights_only=True)
        shape = source.get("config")
        shape = shape.get("model") if isinstance(shape, dict) else None
        origin = source.get("meta")
        origin = origin if isinstance(origin, dict) else {}
        if (shape != config["model"] or "model" not in source
                or origin.get("dataset") != meta["dataset"]
                or origin.get("split") != meta["split"]
                or origin.get("tokenizer") != meta["tokenizer"]):
            raise ValueError("initial checkpoint architecture or approved data hashes differ")
        if init.resolve() == (output.resolve() if output is not None else (HERE / "data" / "foundation.pt").resolve()):
            raise ValueError("initial checkpoint and new output must be separate files")
        model.load_state_dict(source["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(train["rate"]),
                            weight_decay=float(train["penalty"]))
    schedule = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: rate(step, train))
    start = 0
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=True)
        if (payload.get("schema") != 1 or payload.get("arch") != ARCH
                or payload.get("meta") != meta or payload.get("config") != config
                or payload.get("seed") != seed):
            raise ValueError("resume checkpoint configuration, code, data or tokenizer hash mismatch")
        model.load_state_dict(payload["model"])
        opt.load_state_dict(payload["optimizer"])
        schedule.load_state_dict(payload["scheduler"])
        sampler = Sampler(root, "train", seed + 1, payload["sampler"])
        random.setstate(payload["python"])
        torch.set_rng_state(payload["torch"])
        start = int(payload["step"])
    if start > steps or (stop is not None and not start < stop <= steps):
        raise ValueError("invalid segment stop or total step count")
    path = output or HERE / "data" / "foundation.pt"
    if resume is None and path.exists():
        raise FileExistsError(path)
    started = time.monotonic()
    before = evaluate(model, root, seed + 2, train, started + int(train["time"]))
    stamp = time.monotonic()
    swap = guard.swap_counters()
    report = []
    interrupted = False
    oldint = signal.getsignal(signal.SIGINT)
    oldterm = signal.getsignal(signal.SIGTERM)
    request = {"stop": False}

    def halt(signum: int, frame: Any) -> None:
        request["stop"] = True

    signal.signal(signal.SIGINT, halt)
    signal.signal(signal.SIGTERM, halt)
    end = steps if stop is None else stop
    try:
        for index in range(start, end):
            elapsed = time.monotonic() - started
            if elapsed >= int(train["time"]):
                interrupted = True
                if index > start:
                    save(path, checkpoint(model, opt, schedule, sampler, meta,
                                          config, index, seed))
                break
            if index % 10 == 0:
                if guard.available_mib() < guard.MEMORY_GUARD_MIB:
                    raise MemoryError(f"effective available memory fell below guard at step {index}")
                current = guard.swap_counters()
                if swap is not None and current is not None and any(
                        now > prior for now, prior in zip(current, swap)):
                    raise MemoryError(f"swap activity detected at step {index}")
                swap = current
            model.train()
            opt.zero_grad(set_to_none=True)
            lang = sft = updates = answers = 0.0
            for _ in range(int(train["accum"])):
                tokens, mask = batch(sampler, int(train["batch"]), sequence)
                values = model.losses(tokens, mask)
                lang += float(values["language"].detach()) / int(train["accum"])
                amount = int(values["answers"])
                if amount:
                    sft += float(values["interaction"].detach()) / int(train["accum"])
                    answers += amount
                objective = (float(train["lang"]) * values["language"] +
                             float(train["sft"]) * values["interaction"])
                (objective / int(train["accum"])).backward()
                updates += int(values["tokens"])
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(train["clip"]))
            opt.step()
            schedule.step()
            current = index + 1
            row = {"step": current, "language": lang, "interaction": sft if answers else None,
                   "combined": float(train["lang"]) * lang + float(train["sft"]) * sft,
                   "answers": answers, "tokens": updates,
                   "grad": float(norm), "rate": opt.param_groups[0]["lr"]}
            report.append(row)
            if current % int(train["interval"]) == 0 or current == end or request["stop"]:
                save(path, checkpoint(model, opt, schedule, sampler, meta,
                                      config, current, seed))
            if request["stop"]:
                interrupted = True
                break
    finally:
        signal.signal(signal.SIGINT, oldint)
        signal.signal(signal.SIGTERM, oldterm)
    spent = time.monotonic() - stamp
    model.eval()
    duration = time.monotonic() - started
    valid = evaluate(model, root, seed + 2, train, started + int(train["time"]))
    return {"architecture": config["model"], "training": train, "parameters": model.params,
            "blocks": len(model.blocks), "unique": sum(
                value.numel() for value in model.blocks.parameters()),
            "steps": (report[-1]["step"] if report else start), "resume": start,
            "interrupted": interrupted or (stop is not None and end < steps),
            "device": "cpu", "dtype": "float32", "threads": torch.get_num_threads(),
            "train": report, "before": before, "validation": valid, "seconds": duration,
            "fit": spent,
            "perstep": duration / max(1, len(report)),
            "tokens": sum(row["tokens"] for row in report),
            "peakrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "memory": guard.snapshot(), "hashes": meta,
            "checkpoint": str(path),
            "bytes": path.stat().st_size if path.exists() else 0}


def load(path: Path = HERE / "config.json") -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
