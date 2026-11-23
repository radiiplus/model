"""Bounded deterministic trainer for dual-state refinement."""

from __future__ import annotations

import hashlib
import json
import random
import resource
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from research.prototype.data import ByteWindowLoader
from research.prototype.trainer import (
    MEMORY_GUARD_MIB,
    available_mib,
    save_checkpoint,
    snapshot,
    swap_counters,
)
from research.recursive.dual import Refiner
from research.recursive.model import Decoder
from research.recursive.stack import Stack
from research.experiments.tiny_gpt import TinyGPT
from research.recursive.tasks import VERSION, build


ROOT = Path(__file__).resolve().parent
ARCH = ("plain-v1", "loop-v1", "dual-v1", "stack-v1")


def encode(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(encode(value)).hexdigest()


def check(config: dict[str, Any]) -> tuple[int, list[float]]:
    if config.get("architecture") not in ARCH:
        raise ValueError("unsupported architecture version")
    if config.get("mode") not in {"language", "task"}:
        raise ValueError("mode must be language or task")
    count = config.get("steps")
    model = config.get("model")
    if not isinstance(model, dict) or count not in Refiner.choices:
        raise ValueError("model and supported refinement steps are required")
    if config["architecture"] in {"plain-v1", "stack-v1"} and count != 1:
        raise ValueError("plain and stack controls use one configured pass")
    if count > int(model.get("limit", 8)):
        raise ValueError("training steps exceed the model limit")
    if int(model.get("steps", count)) != count:
        raise ValueError("model default steps must match training steps")
    if config.get("loss") not in {"deep", "final"}:
        raise ValueError("loss must be deep or final")
    weights = config.get("weights")
    if (not isinstance(weights, list) or len(weights) != count
            or any(not isinstance(value, (int, float)) or value < 0 for value in weights)
            or abs(sum(weights) - 1.0) > 1e-8):
        raise ValueError("stage weights must be nonnegative, match steps and sum to one")
    if not isinstance(config.get("seed"), int) or isinstance(config["seed"], bool):
        raise ValueError("seed must be an integer")
    if int(config.get("batch", 0)) < 1:
        raise ValueError("batch must be positive")
    wall = float(config.get("wall", 0))
    if not 0 < wall <= 120:
        raise ValueError("wall must be in (0, 120]")
    for key in ("rate", "decay", "clip"):
        if float(config.get(key, -1)) < 0:
            raise ValueError(f"{key} must be nonnegative")
    if float(config["rate"]) == 0 or float(config["clip"]) == 0:
        raise ValueError("rate and clip must be positive")
    return count, [float(value) for value in weights]


def tasks(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    data = build()
    family = config.get("family")
    if family not in data["families"]:
        raise ValueError("unknown generated task family")
    parts = data["families"][family]
    train = list(parts["train"])
    validation = list(parts["validation"])
    hashes = {
        "train": digest(train),
        "validation": digest(validation),
    }
    longest = max(len((row["prompt"] + row["target"]).encode("utf-8")) - 1
                  for row in train + validation)
    if longest > int(config["model"]["context_length"]):
        raise ValueError("model context cannot hold generated prompt and answer")
    return train, validation, {"hashes": hashes, "version": VERSION, "family": family}


def pack(rows: list[dict[str, Any]], indices: torch.Tensor,
         context: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batches = []
    for index in indices.tolist():
        row = rows[index]
        prompt = row["prompt"].encode("utf-8")
        sequence = list(prompt + row["target"].encode("utf-8"))
        if len(sequence) < 2 or len(sequence) - 1 > context:
            raise ValueError("task example does not fit configured context")
        batches.append((sequence[:-1], sequence[1:], len(prompt) - 1))
    width = max(len(source) for source, target, start in batches)
    ids = torch.zeros((len(batches), width), dtype=torch.long)
    targets = torch.zeros_like(ids)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for index, (source, target, start) in enumerate(batches):
        length = len(source)
        ids[index, :length] = torch.tensor(source, dtype=torch.long)
        targets[index, :length] = torch.tensor(target, dtype=torch.long)
        mask[index, start:length] = True
    return ids, targets, mask


def state(model: torch.nn.Module) -> str:
    rows = [(key, value.detach().cpu().contiguous().view(torch.uint8).tolist())
            for key, value in sorted(model.state_dict().items())]
    return digest(rows)


def guard() -> None:
    if available_mib() < MEMORY_GUARD_MIB:
        raise MemoryError(f"effective available memory below {MEMORY_GUARD_MIB} MiB")


def threads() -> None:
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            raise RuntimeError("configure one Torch interop thread before training") from exc
    torch.use_deterministic_algorithms(True)


def tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(tuples(item) for item in value)
    return value


def run(config: dict[str, Any], total: int, path: Path, *, docs: list[list[int]] | None = None,
        hold: list[list[int]] | None = None, stop: int | None = None,
        resume: Path | None = None) -> dict[str, Any]:
    count, weights = check(config)
    if total < 1 or (stop is not None and not 1 <= stop <= total):
        raise ValueError("total and stop steps are invalid")
    whole = time.perf_counter()
    deadline = whole + float(config["wall"])
    if config["mode"] == "language" and (not docs or not hold):
        raise ValueError("language mode requires train and validation documents")
    if config["mode"] == "task" and (docs is not None or hold is not None):
        raise ValueError("task mode constructs its own generated split")
    guard()
    threads()
    seed = int(config["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    rows = holdout = None
    meta = {"version": None, "family": None}
    if config["mode"] == "task":
        rows, holdout, meta = tasks(config)
        hashes = meta["hashes"]
        source, target = hashes["train"], hashes["validation"]
        loader = None
    else:
        loader = ByteWindowLoader(docs, int(config["model"]["context_length"]))
        source, target = digest(docs), digest(hold)
        if {digest(row) for row in docs} & {digest(row) for row in hold}:
            raise ValueError("language train and validation documents overlap")
    signature = {
        "config": config,
        "architecture": config["architecture"],
        "train": source,
        "validation": target,
        "generator": meta["version"],
        "refinement": {"steps": count, "loss": config["loss"], "weights": weights},
    }
    arch = config["architecture"]
    if arch == "plain-v1":
        model = TinyGPT(config["model"])
    elif arch == "stack-v1":
        model = Stack(config["model"])
    elif arch == "dual-v1":
        model = Refiner(config["model"])
    else:
        model = Decoder(config["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["rate"]),
        weight_decay=float(config["decay"]),
    )
    sampler = torch.Generator(device="cpu").manual_seed(seed + 1)
    cursor = 0
    first = 0
    history = []
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=True)
        if payload.get("signature") != signature:
            raise ValueError("checkpoint metadata does not match this run")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        sampler.set_state(payload["sampler"])
        torch.set_rng_state(payload["torch"])
        random.setstate(tuples(payload["python"]))
        cursor = int(payload["cursor"])
        history = payload["history"]
        first = int(payload["step"])
    if total < first or (stop is not None and not first < stop <= total):
        raise ValueError("requested stop must advance beyond checkpoint step")
    actual = total if stop is None else stop
    if actual <= first:
        raise ValueError("run has no remaining training steps")

    history = list(history)
    elapsed = 0.0
    before = swap_counters()
    for step in range(first, actual):
        tick = time.perf_counter()
        if tick >= deadline:
            raise TimeoutError("training wall-time limit reached")
        guard()
        after = swap_counters()
        if before is not None and after is not None and any(
            current > previous for current, previous in zip(after, before)
        ):
            raise MemoryError("swap activity detected; stopping training")
        before = after
        model.train()
        if config["mode"] == "language":
            sample = loader.sample(int(config["batch"]), sampler)
            ids, targets = sample[:, :-1], sample[:, 1:]
            mask = torch.ones_like(targets, dtype=torch.bool)
        else:
            choices = torch.randint(len(rows), (int(config["batch"]),), generator=sampler)
            ids, targets, mask = pack(
                rows, choices, int(config["model"]["context_length"])
            )
        if arch in {"plain-v1", "stack-v1"}:
            logits = model(ids)
            values = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                reduction="none",
            ).reshape_as(targets)
            stages = values.masked_select(mask).mean().reshape(1)
        else:
            _, trace = model(ids, steps=count, trace=True, targets=targets, mask=mask)
            stages = trace["losses"]
        used = weights if config["loss"] == "deep" else [0.0] * (count - 1) + [1.0]
        loss = sum(weight * value for weight, value in zip(used, stages))
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite refinement loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["clip"]))
        optimizer.step()
        history.append([float(value.detach()) for value in stages])
        if config["mode"] == "language":
            cursor += int(config["batch"]) * int(config["model"]["context_length"])
        else:
            cursor += int(config["batch"])
        elapsed += time.perf_counter() - tick
        save_checkpoint(path, {
            "schema": 1, "signature": signature, "step": step + 1,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "sampler": sampler.get_state(), "cursor": cursor,
            "torch": torch.get_rng_state(), "python": list(random.getstate()),
            "history": history,
        })
    if time.perf_counter() > deadline:
        raise TimeoutError("training wall-time limit reached before validation")
    metrics = evaluate(model, config, holdout, hold, count, config["mode"])
    if time.perf_counter() > deadline:
        raise TimeoutError("training wall-time limit reached during validation")
    high = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return {
        "updates": actual,
        "interrupted": actual < total,
        "mode": config["mode"],
        "family": meta["family"],
        "params": sum(value.numel() for value in model.parameters()),
        "blocks": int(config["model"].get("depth", 1)) if arch == "stack-v1" else 1,
        "calls": actual * (int(config["model"].get("depth", 1)) if arch == "stack-v1"
                           else count * (2 if arch == "dual-v1" else 1)),
        "count": cursor,
        "speed": cursor / max(elapsed, 1e-9),
        "stages": history,
        "loss": config["loss"],
        "weights": used,
        "time": elapsed,
        "elapsed": time.perf_counter() - whole,
        "perstep": elapsed / max(actual - first, 1),
        "rss": high,
        "memory": snapshot(),
        "validation": metrics,
        "state": state(model),
        "checkpoint": path.stat().st_size,
        "hash": hashlib.sha256(path.read_bytes()).hexdigest(),
        "train": source,
        "valid": target,
        "unit": "tokens" if config["mode"] == "language" else "examples",
        "generator": meta["version"],
        "steps": count,
        "architecture": config["architecture"],
        "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": digest(config),
        "seed": seed,
        "device": "cpu",
        "dtype": "float32",
        "threads": torch.get_num_threads(),
        "guard": MEMORY_GUARD_MIB,
        "wall": config["wall"],
    }


def evaluate(model: torch.nn.Module, config: dict[str, Any], rows: list[dict[str, Any]] | None,
             docs: list[list[int]] | None, steps: int, mode: str) -> dict[str, float | int]:
    model.eval()
    vocab = model.vocab if isinstance(model, Refiner) else model.vocab_size
    total = correct = count = 0
    nll = 0.0
    with torch.inference_mode():
        if mode == "task":
            for start in range(0, len(rows), int(config["batch"])):
                ids, targets, mask = pack(
                    rows[start:start + int(config["batch"])],
                    torch.arange(min(int(config["batch"]), len(rows) - start)),
                    int(config["model"]["context_length"]),
                )
                if isinstance(model, (TinyGPT, Stack)):
                    logits = model(ids)
                else:
                    logits = model(ids, steps=steps)
                values = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, vocab), targets.reshape(-1), reduction="none"
                ).reshape_as(targets)
                nll += float(values.masked_select(mask).sum())
                correct += int(((logits.argmax(-1) == targets) & mask).sum())
                count += int(mask.sum())
        else:
            loader = ByteWindowLoader(docs, int(config["model"]["context_length"]))
            picks = torch.arange(min(len(loader.windows), 32))
            for start in range(0, len(picks), int(config["batch"])):
                batch = torch.stack([loader.window(int(index))
                                     for index in picks[start:start + int(config["batch"])]] )
                ids, targets = batch[:, :-1], batch[:, 1:]
                logits = model(ids) if isinstance(model, (TinyGPT, Stack)) else model(ids, steps=steps)
                values = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, vocab), targets.reshape(-1), reduction="none"
                )
                nll += float(values.sum())
                correct += int((logits.argmax(-1) == targets).sum())
                count += targets.numel()
    return {"tokens": count, "nll": nll / max(count, 1), "accuracy": correct / max(count, 1)}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    config = load(ROOT / "train.json")
    text = (ROOT.parent / "experiments" / "fixtures" / "tiny_corpus.jsonl").read_text(
        encoding="utf-8"
    )
    data = [[byte for byte in row.encode("utf-8")] for row in text.splitlines() if row]
    docs, hold = data[:-4], data[-4:]
    with tempfile.TemporaryDirectory() as folder:
        result = run(config, 2, Path(folder) / "train.pt", docs=docs, hold=hold)
    (ROOT / "training.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
