"""Transparent planning calculator for the proposed dense decoder configs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

MIB = 1024**2

CONFIGS = {
    "pilot": {"vocab": 16384, "context": 1024, "layers": 4, "width": 384, "heads": 6, "tokens": 100_000_000},
    "target": {"vocab": 16384, "context": 2048, "layers": 8, "width": 640, "heads": 10, "tokens": 1_000_000_000},
    "stretch": {"vocab": 16384, "context": 2048, "layers": 12, "width": 768, "heads": 12, "tokens": 1_000_000_000},
}

PROFILES = {
    "conservative": 0.25e12,
    "moderate": 1.5e12,
    "ambitious": 8e12,
}


def estimate(name: str, tokens: int, sustained_flops_s: float, profile: str) -> dict[str, str | int | float]:
    c = CONFIGS[name]
    v, ctx, layers, d, heads = (c[k] for k in ("vocab", "context", "layers", "width", "heads"))
    dff = 4 * d
    if d % heads:
        raise ValueError(f"{name}: width must divide evenly across attention heads")

    # Tied output projection is counted once; this matches tiny_gpt.py.
    params = v * d + ctx * d + layers * (4 * d * d + 2 * d * dff + dff + 9 * d) + 2 * d
    fp32_weights = 4 * params
    fp32_gradients = 4 * params
    adam_states = 8 * params
    bf16_with_fp32_state_floor = 16 * params
    bf16_with_fp32_gradients = 18 * params

    # One sequence per microbatch. Saved-activation factor is an editable proxy,
    # not a framework allocator measurement. Full FP32 logits are included.
    microbatch = 1
    activation_proxy = 12 * layers * microbatch * ctx * d * 2
    logits_fp32 = microbatch * ctx * v * 4
    persistent = bf16_with_fp32_gradients
    working_estimate = persistent + activation_proxy + logits_fp32

    # Dense matmuls plus a conservative full-context attention estimate.
    training_flops = 6 * params * tokens + 12 * layers * ctx * d * tokens
    wall_seconds = training_flops / sustained_flops_s * 1.25
    return {
        "model": name,
        "profile": profile,
        "vocab": v,
        "context": ctx,
        "layers": layers,
        "width": d,
        "heads": heads,
        "mlp_width": dff,
        "parameters": params,
        "tokens": tokens,
        "tokens_per_parameter": tokens / params,
        "fp32_weights_MiB": fp32_weights / MIB,
        "fp32_gradients_MiB": fp32_gradients / MIB,
        "adam_moments_MiB": adam_states / MIB,
        "fp32_persistent_16N_MiB": 16 * params / MIB,
        "bf16_fp32_state_floor_16N_MiB": bf16_with_fp32_state_floor / MIB,
        "bf16_fp32_gradient_floor_18N_MiB": bf16_with_fp32_gradients / MIB,
        "activation_proxy_MiB": activation_proxy / MIB,
        "fp32_logits_proxy_MiB": logits_fp32 / MIB,
        "estimated_working_bytes_MiB": working_estimate / MIB,
        "fp32_training_checkpoint_floor_12N_MiB": 12 * params / MIB,
        "kv_cache_fp16_batch1_MiB": 2 * layers * ctx * d * 2 / MIB,
        "estimated_training_FLOPs": training_flops,
        "assumed_sustained_TFLOP_s": sustained_flops_s / 1e12,
        "wall_hours_with_25pct_overhead": wall_seconds / 3600,
        "assumption_note": "Scenario extrapolation; sustained FLOP/s is an input, not a hardware benchmark",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*CONFIGS, "all"], default="all")
    parser.add_argument("--tokens", type=int, default=None, help="Override token target for selected model(s)")
    parser.add_argument("--sustained-tflops", type=float, default=None,
                        help="Use a custom sustained useful FLOP/s scenario for selected model(s)")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path")
    args = parser.parse_args()

    names = list(CONFIGS) if args.model == "all" else [args.model]
    rows = []
    for name in names:
        token_count = args.tokens if args.tokens is not None else CONFIGS[name]["tokens"]
        if token_count <= 0:
            parser.error("--tokens must be positive")
        rates = {"custom": args.sustained_tflops * 1e12} if args.sustained_tflops is not None else PROFILES
        if any(rate <= 0 for rate in rates.values()):
            parser.error("--sustained-tflops must be positive")
        for profile, rate in rates.items():
            rows.append(estimate(name, token_count, rate, profile))

    fields = list(rows[0])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
