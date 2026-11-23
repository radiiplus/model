#!/usr/bin/env python3
"""Small, guarded optimizer and training-loop experiments on the fixed TinyGPT."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from tiny_gpt import TinyGPT


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "baseline_config.json"


def available_mib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable unavailable; refusing to run without a memory guard")


def fixed_batch() -> torch.Tensor:
    cycle = torch.tensor([1, 4, 7, 2, 5, 3], dtype=torch.long)
    rows = [torch.roll(cycle.repeat(4), shifts=phase) for phase in range(4)]
    return torch.stack(rows)


def make_optimizers(model: TinyGPT, kind: str, lr_adam: float, lr_muon: float):
    if kind == "adamw":
        return [torch.optim.AdamW(model.parameters(), lr=lr_adam, weight_decay=0.0)]
    if kind != "muon_hybrid":
        raise ValueError(kind)
    muon_names = ("attn.qkv.weight", "attn.proj.weight", "mlp.fc_in.weight", "mlp.fc_out.weight")
    muon_params = [p for name, p in model.named_parameters() if p.ndim == 2 and any(key in name for key in muon_names)]
    muon_ids = {id(p) for p in muon_params}
    adam_params = [p for p in model.parameters() if id(p) not in muon_ids]
    if not muon_params or not adam_params or not hasattr(torch.optim, "Muon"):
        raise RuntimeError("Muon hybrid is unavailable or its parameter partition is empty")
    return [
        torch.optim.Muon(muon_params, lr=lr_muon, weight_decay=0.0, momentum=0.95,
                         nesterov=True, ns_steps=5, adjust_lr_fn="original"),
        torch.optim.AdamW(adam_params, lr=lr_adam, weight_decay=0.0),
    ]


def state_bytes(optimizers) -> int:
    return sum(t.numel() * t.element_size()
               for opt in optimizers for state in opt.state.values() for t in state.values()
               if isinstance(t, torch.Tensor))


def zero_grad(optimizers) -> None:
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)


def step(optimizers) -> None:
    for opt in optimizers:
        opt.step()


def snapshot(model: TinyGPT, optimizers, step_no: int, torch_rng, py_rng) -> dict[str, Any]:
    return {"model": copy.deepcopy(model.state_dict()),
            "optimizers": [copy.deepcopy(opt.state_dict()) for opt in optimizers],
            "step": step_no, "torch_rng": torch_rng.get_state(), "python_rng": py_rng.getstate()}


def load_snapshot(model: TinyGPT, optimizers, state: dict[str, Any], torch_rng, py_rng) -> int:
    model.load_state_dict(state["model"])
    for opt, opt_state in zip(optimizers, state["optimizers"], strict=True):
        opt.load_state_dict(opt_state)
    torch_rng.set_state(state["torch_rng"])
    py_rng.setstate(state["python_rng"])
    return int(state["step"])


def run_one(config: dict[str, Any], optimizer_kind: str, seed: int, steps: int,
            lr_adam: float, lr_muon: float, *, stop_after: int | None = None,
            resume: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if available_mib() < 2560:
        raise MemoryError("MemAvailable below 2560 MiB guard")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = TinyGPT(config).cpu()
    optimizers = make_optimizers(model, optimizer_kind, lr_adam, lr_muon)
    torch_rng, py_rng = torch.Generator().manual_seed(seed + 33), random.Random(seed + 77)
    tokens = fixed_batch()
    start_step = 0
    if resume is not None:
        start_step = load_snapshot(model, optimizers, resume, torch_rng, py_rng)
    with torch.no_grad():
        initial_loss = float(model.loss(tokens))
    curve = []
    if start_step == 0:
        curve.append({"step": 0, "loss_nats": initial_loss})
    begin = time.perf_counter()
    run_to = stop_after if stop_after is not None else steps
    for step_no in range(start_step + 1, run_to + 1):
        if step_no % 20 == 0 and available_mib() < 2560:
            raise MemoryError(f"MemAvailable below guard at step {step_no}")
        # The fixed batch and RNG state are checkpointed, so all methods see identical tokens.
        torch.rand(1, generator=torch_rng)
        py_rng.random()
        zero_grad(optimizers)
        loss = model.loss(tokens)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at step {step_no}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        step(optimizers)
        if step_no in {1, 2, 5, 10, 20, 50, 100, steps} or step_no % 50 == 0:
            with torch.no_grad():
                curve.append({"step": step_no, "loss_nats": float(model.loss(tokens))})
    elapsed = time.perf_counter() - begin
    with torch.no_grad():
        final_loss = float(model.loss(tokens))
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode("ascii"))
        digest.update(bytes(tensor.untyped_storage()))
    state_digest = digest.hexdigest()
    result = {
        "optimizer": optimizer_kind, "seed": seed, "steps": run_to,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "initial_loss_nats": initial_loss, "final_loss_nats": final_loss,
        "training_curve": curve, "training_seconds": elapsed,
        "tokens_per_second": (tokens.shape[0] * (tokens.shape[1] - 1) * (run_to - start_step)) / max(elapsed, 1e-12),
        "optimizer_state_bytes": state_bytes(optimizers),
        "model_state_sha256": state_digest,
        "mem_available_mib_at_end": available_mib(),
    }
    mid_state = snapshot(model, optimizers, run_to, torch_rng, py_rng) if stop_after is not None else None
    return result, mid_state


def accumulation_check(config: dict[str, Any], seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    initial = TinyGPT(config).cpu().state_dict()
    tokens = fixed_batch()
    full = TinyGPT(config).cpu()
    full.load_state_dict(initial)
    micro = TinyGPT(config).cpu()
    micro.load_state_dict(initial)
    opt_full = torch.optim.AdamW(full.parameters(), lr=0.001, weight_decay=0.0)
    opt_micro = torch.optim.AdamW(micro.parameters(), lr=0.001, weight_decay=0.0)
    opt_full.zero_grad(set_to_none=True)
    full.loss(tokens).backward()
    opt_full.step()
    opt_micro.zero_grad(set_to_none=True)
    for row in tokens.split(1):
        (micro.loss(row) / tokens.shape[0]).backward()
    opt_micro.step()
    max_abs = max(float((a - b).abs().max().detach()) for a, b in zip(full.parameters(), micro.parameters(), strict=True))
    return {"effective_batch_size": int(tokens.shape[0]), "micro_batch_size": 1,
            "max_parameter_abs_difference_after_one_step": max_abs, "equivalent_within_1e-6": max_abs < 1e-6}


def resume_check(config: dict[str, Any], kind: str, seed: int, steps: int,
                 lr_adam: float, lr_muon: float) -> dict[str, Any]:
    full_result, _ = run_one(config, kind, seed, steps, lr_adam, lr_muon)
    _, state = run_one(config, kind, seed, steps, lr_adam, lr_muon, stop_after=steps // 2)
    assert state is not None
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "resume.pt"
        torch.save(state, path)
        restored = torch.load(path, map_location="cpu", weights_only=False)
    resumed_result, _ = run_one(config, kind, seed, steps, lr_adam, lr_muon, resume=restored)
    return {"full_final_loss_nats": full_result["final_loss_nats"],
            "resumed_final_loss_nats": resumed_result["final_loss_nats"],
            "full_model_state_sha256": full_result["model_state_sha256"],
            "resumed_model_state_sha256": resumed_result["model_state_sha256"],
            "loss_exact": full_result["final_loss_nats"] == resumed_result["final_loss_nats"],
            "model_state_exact": full_result["model_state_sha256"] == resumed_result["model_state_sha256"],
            "scope": "model, optimizer, and RNG state serialized to a temporary file and resumed"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--lr-adam", type=float, default=0.003)
    parser.add_argument("--lr-muon", type=float, default=0.02)
    parser.add_argument("--include-muon", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000:
        raise SystemExit("--steps must be in 1..1000")
    if available_mib() < 2560:
        raise SystemExit("MemAvailable below 2560 MiB guard")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    methods = ["adamw"] + (["muon_hybrid"] if args.include_muon else [])
    runs = []
    for kind in methods:
        for seed in args.seeds:
            record, _ = run_one(config, kind, seed, args.steps, args.lr_adam, args.lr_muon)
            runs.append(record)
    result = {
        "schema_version": 1,
        "experiment": "fixed_tiny_gpt_optimizer_and_training_efficiency",
        "config": config,
        "batch": {"batch_size": 4, "sequence_length": int(fixed_batch().shape[1]),
                  "same_fixed_batch_each_update": True},
        "hyperparameters": {"adamw_lr": args.lr_adam, "muon_lr": args.lr_muon,
                             "weight_decay": 0.0, "global_grad_clip": 1.0,
                             "muon_momentum": 0.95, "muon_nesterov": True,
                             "muon_newton_schulz_steps": 5},
        "runs": runs,
        "accumulation_check": accumulation_check(config, args.seeds[0]),
        "resume_checks": {kind: resume_check(config, kind, args.seeds[0], args.steps, args.lr_adam, args.lr_muon)
                          for kind in methods},
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "cpu_threads": torch.get_num_threads(), "cuda_available": torch.cuda.is_available(),
                        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                        "mem_available_mib_at_start": available_mib()},
        "limitations": ["tiny repeated synthetic pattern only", "optimizer learning rates are not tuned or matched by an extensive sweep",
                        "three seeds are preliminary", "no large-scale capability or eventual pretraining throughput inference"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
