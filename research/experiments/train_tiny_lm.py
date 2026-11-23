#!/usr/bin/env python3
"""Short, CPU-only overfit run on one synthetic periodic token sequence."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import torch

from tiny_gpt import TinyGPT, checkpoint_payload, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments" / "baseline_config.json"
DEFAULT_RESULT = ROOT / "experiments" / "results" / "baseline_training.json"


def available_mib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable unavailable; stop rather than train without a memory guard")


def train(config_path: Path = DEFAULT_CONFIG, result_path: Path = DEFAULT_RESULT,
          steps: int = 250, min_available_mib: int = 2560) -> dict:
    if available_mib() < min_available_mib:
        raise MemoryError(f"MemAvailable below guard of {min_available_mib} MiB")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(int(config["seed"]))
    torch.use_deterministic_algorithms(True)
    model = TinyGPT(config).cpu()
    tokens = torch.tensor([[1, 4, 7, 2, 5, 3] * 2], dtype=torch.long)
    if int(tokens.max()) >= model.vocab_size:
        raise ValueError("synthetic training tokens exceed vocabulary")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    record_steps = sorted(set([0, 1, 2, 5, 10, 20, 40, 80, 120, steps]))
    curve = []

    model.eval()
    with torch.no_grad():
        initial_loss = float(model.loss(tokens))
    curve.append({"step": 0, "loss_nats": initial_loss})
    train_start = time.perf_counter()
    for step in range(1, steps + 1):
        if step % 25 == 0 and available_mib() < min_available_mib:
            raise MemoryError(f"MemAvailable below guard during training at step {step}")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(tokens)
        loss.backward()
        optimizer.step()
        if step in record_steps:
            model.eval()
            with torch.no_grad():
                curve.append({"step": step, "loss_nats": float(model.loss(tokens))})
    train_seconds = time.perf_counter() - train_start
    model.eval()
    with torch.no_grad():
        final_loss = float(model.loss(tokens))
    generated = model.generate(tokens[:, :3], max_new_tokens=12)[0].tolist()

    result_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = result_path.with_name("baseline_tiny_gpt.pt")
    torch.save(checkpoint_payload(model, optimizer), checkpoint_path)
    restored, _ = load_checkpoint(checkpoint_path)
    restored.eval()
    with torch.no_grad():
        roundtrip_loss = float(restored.loss(tokens))
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    result = {
        "schema_version": 1, "config": config, "optimizer": "AdamW", "learning_rate": 0.01,
        "weight_decay": 0.0, "steps": steps, "seed": int(config["seed"]),
        "synthetic_training_tokens": tokens[0].tolist(), "batch_size": 1,
        "sequence_length": int(tokens.shape[1]), "parameter_count": sum(p.numel() for p in model.parameters()),
        "initial_loss_nats": initial_loss, "final_loss_nats": final_loss,
        "training_curve": curve, "train_seconds": train_seconds,
        "seconds_per_step": train_seconds / steps,
        "tokens_per_second": int(tokens[:, 1:].numel()) * steps / train_seconds,
        "tokens_per_second_definition": "next-token targets processed divided by training wall time",
        "peak_rss_bytes": peak_rss, "mem_available_mib_at_start": available_mib(),
        "checkpoint_path": str(checkpoint_path), "checkpoint_roundtrip_loss_nats": roundtrip_loss,
        "checkpoint_roundtrip_exact": final_loss == roundtrip_loss,
        "generated_ids": generated,
        "generated_ids_valid": all(0 <= token_id < model.vocab_size for token_id in generated),
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "cuda_available": torch.cuda.is_available(),
                        "cpu_threads": torch.get_num_threads(),
                        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None},
        "limitations": ["single synthetic periodic sequence; memorization demonstration only", "one seed", "tiny model and CPU-only run do not establish language competence"],
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--min-available-mib", type=int, default=2560)
    args = parser.parse_args()
    if not 1 <= args.steps <= 2000:
        raise SystemExit("--steps must be in 1..2000")
    print(json.dumps(train(args.config, args.output, args.steps, args.min_available_mib), indent=2))


if __name__ == "__main__":
    main()
