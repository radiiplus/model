"""Short deterministic AdamW trainer with interruption-safe checkpoint resume."""

from __future__ import annotations

import json
import io
import os
import resource
import time
from pathlib import Path
from typing import Any

import torch

from research.prototype.data import ByteWindowLoader, evaluate_bytes
from research.prototype.model import ByteLanguageModel

MEMORY_GUARD_MIB = 2560
_TORCH_CONFIGURED = False


def available_mib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable unavailable; refuse to train without a memory guard")


def swap_counters() -> tuple[int, int] | None:
    values = {}
    for line in Path("/proc/vmstat").read_text(encoding="ascii").splitlines():
        key, value = line.split()[:2]
        if key in {"pswpin", "pswpout"}:
            values[key] = int(value)
    return (values["pswpin"], values["pswpout"]) if len(values) == 2 else None


def configure_torch() -> None:
    global _TORCH_CONFIGURED
    if not _TORCH_CONFIGURED:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        _TORCH_CONFIGURED = True


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def train_to(config: dict[str, Any], train_docs: list[list[int]], validation_docs: list[list[int]],
             *, seed: int, total_steps: int, checkpoint_path: Path,
             resume_from: Path | None = None, stop_after_step: int | None = None) -> dict[str, Any]:
    if available_mib() < MEMORY_GUARD_MIB:
        raise MemoryError(f"MemAvailable below {MEMORY_GUARD_MIB} MiB")
    configure_torch()
    loader = ByteWindowLoader(train_docs, int(config["context_length"]))
    if resume_from is None:
        model_adapter = ByteLanguageModel(config, seed)
        model = model_adapter.module
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                      weight_decay=float(config["weight_decay"]))
        generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        start_step = 0
    else:
        payload = torch.load(resume_from, map_location="cpu", weights_only=True)
        if payload["config"] != config or payload["seed"] != seed:
            raise ValueError("resume checkpoint config/seed mismatch")
        model_adapter = ByteLanguageModel(config, seed)
        model = model_adapter.module
        model.load_state_dict(payload["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                      weight_decay=float(config["weight_decay"]))
        optimizer.load_state_dict(payload["optimizer"])
        generator = torch.Generator(device="cpu")
        generator.set_state(payload["batch_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        start_step = int(payload["step"])
    if total_steps < start_step or (stop_after_step is not None and not start_step < stop_after_step <= total_steps):
        raise ValueError("invalid total_steps or stop_after_step")

    model.eval()
    validation_before = evaluate_bytes(model, validation_docs, int(config["context_length"]))
    started = time.perf_counter()
    actual_stop = total_steps if stop_after_step is None else stop_after_step
    last_swap = swap_counters()
    for step in range(start_step, actual_stop):
        if step % 10 == 0 and available_mib() < MEMORY_GUARD_MIB:
            raise MemoryError(f"MemAvailable fell below guard at step {step}")
        if step % 10 == 0:
            current_swap = swap_counters()
            if last_swap is not None and current_swap is not None and any(
                current > previous for current, previous in zip(current_swap, last_swap)
            ):
                raise MemoryError(f"swap activity detected at step {step}; stop this run")
            last_swap = current_swap
        model.train()
        batch = loader.sample(int(config["batch_size"]), generator)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        next_step = step + 1
        if next_step == actual_stop or next_step % 10 == 0:
            save_checkpoint(checkpoint_path, {
                "schema_version": 1, "config": config, "seed": seed, "step": next_step,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "batch_rng_state": generator.get_state(), "torch_rng_state": torch.get_rng_state(),
            })
    train_seconds = time.perf_counter() - started
    model.eval()
    validation_after = evaluate_bytes(model, validation_docs, int(config["context_length"]))
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    export_buffer = io.BytesIO()
    torch.save({"config": config, "model": model.state_dict()}, export_buffer)
    return {
        "seed": seed, "steps_completed": actual_stop, "interrupted": actual_stop < total_steps,
        "train_seconds": train_seconds,
        "seconds_per_step": train_seconds / max(actual_stop - start_step, 1),
        "target_bytes_per_step": int(config["batch_size"]) * int(config["context_length"]),
        "target_bytes_per_second": (actual_stop - start_step) * int(config["batch_size"]) * int(config["context_length"]) / max(train_seconds, 1e-12),
        "train_windows": len(loader.windows), "validation_before": validation_before,
        "validation_after": validation_after, "parameter_count": sum(p.numel() for p in model.parameters()),
        "peak_rss_bytes_process_highwater": peak_rss,
        "model_tensor_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "inference_export_bytes": len(export_buffer.getvalue()),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_path": str(checkpoint_path), "resume_step": start_step,
        "generation_repeatable": model_adapter.generate([32, 65, 66], 8) == model_adapter.generate([32, 65, 66], 8),
    }


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
