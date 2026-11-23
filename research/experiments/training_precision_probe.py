#!/usr/bin/env python3
"""Check CPU autocast forward/backward support and latency on the tiny baseline."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import torch

from tiny_gpt import TinyGPT
from training_efficiency import available_mib, fixed_batch


ROOT = Path(__file__).resolve().parents[1]


def probe(config: dict, steps: int) -> dict:
    if available_mib() < 2560:
        raise MemoryError("MemAvailable below 2560 MiB guard")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rows = []
    modes = [("float32", None), ("bfloat16_autocast", torch.bfloat16), ("float16_autocast", torch.float16)]
    tokens = fixed_batch()
    for name, dtype in modes:
        torch.manual_seed(301)
        model = TinyGPT(config).cpu()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
        started = time.perf_counter()
        losses = []
        error = None
        try:
            for index in range(steps + 2):
                if index and index % 5 == 0 and available_mib() < 2560:
                    raise MemoryError("MemAvailable below 2560 MiB guard during precision probe")
                optimizer.zero_grad(set_to_none=True)
                if dtype is None:
                    loss = model.loss(tokens)
                else:
                    with torch.autocast(device_type="cpu", dtype=dtype):
                        loss = model.loss(tokens)
                loss.backward()
                optimizer.step()
                if index >= 2:
                    losses.append(float(loss.detach()))
            support = True
        except (RuntimeError, MemoryError) as exc:
            support = False
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        rows.append({"mode": name, "supported": support, "error": error,
                     "timed_steps": len(losses), "seconds_including_two_warmups": elapsed,
                     "seconds_per_step_including_warmups": elapsed / max(len(losses) + 2, 1),
                     "last_loss_nats": losses[-1] if losses else None,
                     "finite_loss": all(torch.isfinite(torch.tensor(v)) for v in losses)})
    return {"schema_version": 1, "device": "cpu", "model_parameters": sum(p.numel() for p in TinyGPT(config).parameters()),
            "sequence_length": int(tokens.shape[1]), "batch_size": int(tokens.shape[0]),
            "modes": rows, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                            "threads": torch.get_num_threads(), "mem_available_mib": available_mib()},
            "limitations": ["tiny one-layer model only", "CPU autocast speed is kernel and shape dependent",
                            "peak RSS is process-wide across sequential modes"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/baseline_config.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.steps <= 100:
        raise SystemExit("--steps must be in 1..100")
    result = probe(json.loads(args.config.read_text(encoding="utf-8")), args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
