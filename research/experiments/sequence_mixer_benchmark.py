#!/usr/bin/env python3
"""Small same-budget toy comparison: full causal attention vs recurrent retention."""

from __future__ import annotations

import argparse
import json
import os
import random
import resource
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_gpt import TinyGPT
from sequence_mixers import RetentionGPT

DEFAULT_CONFIG = ROOT / "experiments" / "baseline_config.json"
DEFAULT_OUTPUT = ROOT / "experiments" / "results" / "sequence_mixer_tiny.json"
TRAIN_LENGTH = 32
SEEDS = (101, 202, 303)
TASKS = ("periodic_next_token", "parity_state", "copy_exact_token")


def available_mib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable unavailable; stop rather than run without memory guard")


def make_periodic_batch(batch: int, length: int) -> torch.Tensor:
    pattern = (1, 4, 7, 2, 5, 3)
    row = [pattern[i % len(pattern)] for i in range(length)]
    return torch.tensor([row[:] for _ in range(batch)], dtype=torch.long)


def make_parity_batch(batch: int, length: int, rng: random.Random):
    rows, labels = [], []
    for _ in range(batch):
        bits = [rng.randrange(2) for _ in range(length - 1)]
        rows.append([1 + bit for bit in bits] + [11])
        labels.append(8 + (sum(bits) % 2))
    return torch.tensor(rows, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def make_copy_batch(batch: int, length: int, rng: random.Random):
    rows, labels = [], []
    for _ in range(batch):
        delay = rng.randint(1, length - 3)
        payload = rng.randint(1, 7)
        row = [0] * length
        row[0], row[length - 1], row[length - delay - 2] = 10, 11, payload
        rows.append(row)
        labels.append(payload)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def task_batch(task: str, rng: random.Random, batch: int = 4):
    if task == "periodic_next_token":
        return make_periodic_batch(batch, TRAIN_LENGTH), None
    if task == "parity_state":
        return make_parity_batch(batch, TRAIN_LENGTH, rng)
    return make_copy_batch(batch, TRAIN_LENGTH, rng)


def task_loss(model, task: str, ids: torch.Tensor, labels: torch.Tensor | None):
    logits = model(ids)
    if task == "periodic_next_token":
        return F.cross_entropy(logits[:, :-1].reshape(-1, model.vocab_size), ids[:, 1:].reshape(-1))
    return F.cross_entropy(logits[:, -1], labels)


def task_accuracy(model, task: str, seed: int) -> dict:
    model.eval()
    with torch.no_grad():
        if task == "periodic_next_token":
            ids = make_periodic_batch(1, TRAIN_LENGTH)
            pred = model(ids)[:, :-1].argmax(dim=-1)
            correct = int((pred == ids[:, 1:]).sum())
            total = ids.shape[1] - 1
            return {"accuracy": correct / total, "correct": correct, "total": total}
        if task == "parity_state":
            metrics = {}
            for length in (8, 32, 64, 128):
                rng = random.Random(seed * 1000 + length)
                ids, labels = make_parity_batch(32, length, rng)
                correct = int((model(ids)[:, -1].argmax(dim=-1) == labels).sum())
                metrics[str(length)] = {"accuracy": correct / len(labels), "correct": correct, "total": len(labels)}
            return metrics
        metrics = {}
        for delay in (1, 4, 8, 16, 29, 32, 64, 125):
            length = delay + 3
            ids, labels = [], []
            for i in range(28):
                row = [0] * length
                payload = 1 + (i % 7)
                row[0], row[1], row[-1] = 10, payload, 11
                ids.append(row)
                labels.append(payload)
            inputs = torch.tensor(ids, dtype=torch.long)
            targets = torch.tensor(labels, dtype=torch.long)
            correct = int((model(inputs)[:, -1].argmax(dim=-1) == targets).sum())
            metrics[str(delay)] = {"accuracy": correct / len(targets), "correct": correct, "total": len(targets)}
        return metrics


def train_one(model_name: str, task: str, seed: int, config: dict, steps: int) -> dict:
    torch.manual_seed(seed)
    model_config = dict(config)
    model = TinyGPT(model_config) if model_name == "transformer" else RetentionGPT(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    rng = random.Random(seed * 1009 + (TASKS.index(task) + 1) * 97)
    model.eval()
    with torch.no_grad():
        first_ids, first_labels = task_batch(task, rng)
        initial_loss = float(task_loss(model, task, first_ids, first_labels))
    curve = [{"step": 0, "validation_loss": initial_loss}]
    validation_rng = random.Random(seed * 1013 + (TASKS.index(task) + 1) * 89)
    validation_ids, validation_labels = task_batch(task, validation_rng)
    rng = random.Random(seed * 1009 + (TASKS.index(task) + 1) * 97)
    train_start = time.perf_counter()
    for step in range(1, steps + 1):
        if step % 20 == 0 and available_mib() < 2560:
            raise MemoryError(f"MemAvailable below 2560 MiB in {model_name}/{task} at step {step}")
        model.train()
        ids, labels = task_batch(task, rng)
        optimizer.zero_grad(set_to_none=True)
        loss = task_loss(model, task, ids, labels)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss in {model_name}/{task} at step {step}")
        loss.backward()
        optimizer.step()
        if step in (1, 10, steps):
            model.eval()
            with torch.no_grad():
                curve.append({"step": step, "validation_loss": float(task_loss(model, task, validation_ids, validation_labels))})
    train_seconds = time.perf_counter() - train_start
    return {
        "model": model_name, "task": task, "seed": seed, "steps": steps,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "initial_loss": initial_loss, "training_curve": curve,
        "train_seconds": train_seconds,
        "accuracy": task_accuracy(model, task, seed),
    }


def state_bytes(state) -> int:
    if isinstance(state, torch.Tensor):
        return state.numel() * state.element_size()
    if isinstance(state, (tuple, list)):
        return sum(state_bytes(item) for item in state)
    return 0


def latency_benchmark(config: dict, repeats: int) -> dict:
    torch.manual_seed(9001)
    models = {"transformer": TinyGPT(config), "retention": RetentionGPT(config)}
    result = {}
    with torch.no_grad():
        for name, model in models.items():
            model.eval()
            lengths = {}
            for length in (32, 64, 128):
                ids = torch.randint(0, config["vocab_size"], (1, length))
                model(ids)
                start = time.perf_counter()
                for _ in range(repeats):
                    model(ids)
                prefill = (time.perf_counter() - start) / repeats
                entry = {"prefill_seconds": prefill}
                prefix = ids[:, :-1]
                candidate = ids[:, -1]
                if name == "transformer":
                    prefix = torch.cat((ids[:, :-1], candidate[:, None]), dim=1)
                    start = time.perf_counter()
                    for _ in range(repeats):
                        model(prefix)[:, -1]
                    decode = (time.perf_counter() - start) / repeats
                    entry["one_token_decode_seconds_no_kv_cache"] = decode
                    entry["kv_cache_bytes_if_enabled"] = 2 * length * config["d_model"] * 4
                    entry["actual_persistent_state_bytes"] = 0
                else:
                    _, state = model.forward_with_state(prefix)
                    start = time.perf_counter()
                    for _ in range(repeats):
                        model.step(candidate, length - 1, state)
                    decode = (time.perf_counter() - start) / repeats
                    entry["one_token_decode_seconds_recurrent"] = decode
                    entry["actual_persistent_state_bytes"] = state_bytes(state)
                    entry["state_shapes"] = [list(value.shape) for value in state]
                lengths[str(length)] = entry
            result[name] = lengths
    return result


def run(config_path: Path = DEFAULT_CONFIG, output_path: Path = DEFAULT_OUTPUT,
        steps: int = 80, seeds: tuple[int, ...] = SEEDS, repeats: int = 5,
        tasks: tuple[str, ...] = TASKS) -> dict:
    if available_mib() < 2560:
        raise MemoryError("MemAvailable below 2560 MiB before benchmark")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["context_length"] = 128
    results = []
    run_start = time.perf_counter()
    for model_name in ("transformer", "retention"):
        for task in tasks:
            for seed in seeds:
                results.append(train_one(model_name, task, seed, config, steps))
    latencies = latency_benchmark(config, repeats)
    elapsed = time.perf_counter() - run_start
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return {
        "schema_version": 1, "config": config, "train_sequence_length": TRAIN_LENGTH,
        "batch_size": 4, "steps_per_model_task_seed": steps, "seeds": list(seeds),
        "tasks": list(tasks), "runs": results, "latency": latencies,
        "total_wall_seconds": elapsed, "peak_rss_bytes": peak_rss,
        "mem_available_mib_at_start": available_mib(),
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "cpu_threads": torch.get_num_threads(),
                        "cuda_available": torch.cuda.is_available(),
                        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None},
        "limitations": ["RetNet-like prototype is normalized simple retention, not canonical RetNet", "one layer and tiny synthetic probes", "three seeds per task but short synthetic training and no held-out natural text", "no models or external data downloaded", "latency excludes tokenizer and is not a production decode benchmark"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tasks", choices=TASKS, nargs="+", default=list(TASKS))
    args = parser.parse_args()
    if not 1 <= args.steps <= 400 or not 1 <= len(args.seeds) <= 3:
        raise SystemExit("steps must be 1..400 and provide 1..3 seeds")
    output = run(args.config, args.output, args.steps, tuple(args.seeds), args.repeats, tuple(args.tasks))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(output["runs"]), "wall_seconds": output["total_wall_seconds"],
                      "peak_rss_bytes": output["peak_rss_bytes"], "result": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
