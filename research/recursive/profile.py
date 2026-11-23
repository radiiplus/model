"""Measure shared-decoder forward cost at each supported recurrence depth."""

from __future__ import annotations

import hashlib
import json
import resource
import statistics
import time
from pathlib import Path

import torch

from research.prototype.trainer import MEMORY_GUARD_MIB, available_mib, configure_torch, snapshot
from research.recursive.model import Decoder, transfer


ROOT = Path(__file__).resolve().parents[1]
HOME = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> dict[str, object]:
    if available_mib() < MEMORY_GUARD_MIB:
        raise MemoryError(f"effective available memory below {MEMORY_GUARD_MIB} MiB")
    configure_torch()
    torch.manual_seed(29)
    torch.use_deterministic_algorithms(True)
    config = json.loads((HOME / "config.json").read_text(encoding="utf-8"))
    payload = torch.load(ROOT / "experiments" / "results" / "batch12_byte_seed_11.pt",
                         map_location="cpu", weights_only=True)
    model = Decoder(config).eval()
    transfer(model, payload)
    ids = torch.arange(config["context_length"], dtype=torch.long).remainder(config["vocab_size"])[None, :]
    output = []
    started = time.perf_counter()
    for steps in Decoder.choices:
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
        output.append({"steps": steps, "loops": steps, "applications": steps,
                       "params": model.params, "runs": len(times),
                       "latency": {"mean": statistics.mean(times), "p50": statistics.median(times),
                                   "p95": values[18]}})
    return {"version": 1, "seed": 29, "device": "cpu", "dtype": "float32",
            "context": config["context_length"], "batch": 1,
            "blocks": sum(1 for item in model.modules() if item is model.block),
            "params": model.params, "rows": output,
            "memory": snapshot(),
            "runtime": {"wall": time.perf_counter() - started,
                        "rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                        "threads": torch.get_num_threads(), "guard": MEMORY_GUARD_MIB,
                        "timeout": 120},
            "hashes": {"model": digest(HOME / "model.py"),
                       "profile": digest(HOME / "profile.py"),
                       "config": digest(HOME / "config.json"),
                       "baseline": digest(ROOT / "experiments" / "tiny_gpt.py"),
                       "checkpoint": digest(ROOT / "experiments" / "results" / "batch12_byte_seed_11.pt")}}


def main() -> None:
    result = run()
    (HOME / "profile.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
