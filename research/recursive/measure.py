"""Bounded CPU profile for the dual-state refiner."""

from __future__ import annotations

import hashlib
import json
import resource
import statistics
import time
from pathlib import Path

import torch

from research.prototype.trainer import MEMORY_GUARD_MIB, available_mib, configure_torch, snapshot
from research.recursive.dual import Refiner, transfer
from research.recursive.model import Decoder, transfer as load


ROOT = Path(__file__).resolve().parents[1]
HOME = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure() -> dict[str, object]:
    if available_mib() < MEMORY_GUARD_MIB:
        raise MemoryError(f"effective available memory below {MEMORY_GUARD_MIB} MiB")
    configure_torch()
    torch.manual_seed(43)
    torch.use_deterministic_algorithms(True)
    config = json.loads((HOME / "dual.json").read_text(encoding="utf-8"))
    path = ROOT / "experiments" / "results" / "batch12_byte_seed_11.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    base = Decoder({**payload["config"], "limit": 8, "steps": 4}).eval()
    load(base, payload)
    model = Refiner(config).eval()
    transfer(model, base)
    ids = torch.arange(config["context_length"], dtype=torch.long)
    ids = ids.remainder(config["vocab_size"])[None, :]
    rows = []
    started = time.perf_counter()
    for steps in Refiner.choices:
        for _ in range(3):
            with torch.inference_mode():
                model(ids, steps=steps)
        times = []
        for _ in range(20):
            tick = time.perf_counter()
            with torch.inference_mode():
                model(ids, steps=steps)
            times.append(time.perf_counter() - tick)
        values = sorted(times)
        rows.append({
            "steps": steps,
            "loops": steps,
            "applications": 2 * steps,
            "params": model.params,
            "runs": len(times),
            "latency": {
                "mean": statistics.mean(times),
                "p50": statistics.median(times),
                "p95": values[18],
            },
        })
    return {
        "version": 1,
        "seed": 43,
        "device": "cpu",
        "dtype": "float32",
        "context": config["context_length"],
        "batch": 1,
        "blocks": 1,
        "params": model.params,
        "budget": config["budget"],
        "rows": rows,
        "memory": snapshot(),
        "runtime": {
            "wall": time.perf_counter() - started,
            "rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "threads": torch.get_num_threads(),
            "guard": MEMORY_GUARD_MIB,
            "timeout": 120,
        },
        "hashes": {
            "model": digest(HOME / "dual.py"),
            "measure": digest(HOME / "measure.py"),
            "config": digest(HOME / "dual.json"),
            "baseline": digest(ROOT / "experiments" / "tiny_gpt.py"),
            "checkpoint": digest(path),
        },
    }


def main() -> None:
    result = measure()
    (HOME / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
