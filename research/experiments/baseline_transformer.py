#!/usr/bin/env python3
"""Dependency-free tiny decoder-only Transformer forward-pass reference."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import resource
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments" / "baseline_config.json"
DEFAULT_OUTPUT = ROOT / "experiments" / "results" / "baseline_forward.json"


def available_mib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable unavailable; stop rather than run without the memory guard")


def matrix(rng: random.Random, rows: int, cols: int, scale: float) -> list[list[float]]:
    return [[rng.gauss(0.0, scale) for _ in range(cols)] for _ in range(rows)]


def linear(values: list[float], weights: list[list[float]], bias: list[float] | None = None) -> list[float]:
    out = [sum(x * w for x, w in zip(values, row)) for row in weights]
    if bias is not None:
        out = [value + shift for value, shift in zip(out, bias)]
    return out


def layer_norm(values: list[float], gain: list[float], bias: list[float], eps: float = 1e-5) -> list[float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    scale = 1.0 / math.sqrt(variance + eps)
    return [(value - mean) * scale * g + b for value, g, b in zip(values, gain, bias)]


class TinyDecoder:
    def __init__(self, config: dict[str, Any], state: dict[str, Any] | None = None):
        self.config = config
        self.vocab = int(config["vocab_size"])
        self.context = int(config["context_length"])
        self.width = int(config["d_model"])
        self.heads = int(config["n_heads"])
        self.head_width = self.width // self.heads
        self.hidden = int(config["d_ff"])
        if self.width % self.heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.vocab < 2 or self.context < 2 or self.width < 2:
            raise ValueError("vocab_size, context_length and d_model must be >= 2")
        if state is not None:
            self.state = state
            return
        rng = random.Random(int(config["seed"]))
        w = self.width
        self.state = {
            "token_embedding": matrix(rng, self.vocab, w, 1 / math.sqrt(w)),
            "position_embedding": matrix(rng, self.context, w, 1 / math.sqrt(w)),
            "ln1_gain": [1.0] * w, "ln1_bias": [0.0] * w,
            "q": matrix(rng, w, w, 1 / math.sqrt(w)),
            "k": matrix(rng, w, w, 1 / math.sqrt(w)),
            "v": matrix(rng, w, w, 1 / math.sqrt(w)),
            "o": matrix(rng, w, w, 1 / math.sqrt(w)),
            "ln2_gain": [1.0] * w, "ln2_bias": [0.0] * w,
            "ff1": matrix(rng, self.hidden, w, 1 / math.sqrt(w)),
            "ff1_bias": [0.0] * self.hidden,
            "ff2": matrix(rng, w, self.hidden, 1 / math.sqrt(self.hidden)),
            "ff2_bias": [0.0] * w,
            "final_gain": [1.0] * w, "final_bias": [0.0] * w,
        }

    def parameter_count(self) -> int:
        def count(value: Any) -> int:
            if isinstance(value, dict):
                return sum(count(item) for item in value.values())
            if isinstance(value, list):
                return sum(count(item) for item in value)
            return 1
        return count(self.state)

    def forward(self, token_ids: list[int]) -> list[list[float]]:
        if not token_ids or len(token_ids) > self.context:
            raise ValueError("input length must be between 1 and context_length")
        if any(token_id < 0 or token_id >= self.vocab for token_id in token_ids):
            raise ValueError("input contains an out-of-vocabulary token ID")
        p = self.state
        w, head_width = self.width, self.head_width
        x = [[p["token_embedding"][idx][j] + p["position_embedding"][pos][j]
              for j in range(w)] for pos, idx in enumerate(token_ids)]

        # Pre-LayerNorm attention; key positions stop at the query position.
        normed = [layer_norm(row, p["ln1_gain"], p["ln1_bias"]) for row in x]
        q = [linear(row, p["q"]) for row in normed]
        k = [linear(row, p["k"]) for row in normed]
        v = [linear(row, p["v"]) for row in normed]
        attended = [[0.0] * w for _ in token_ids]
        for pos in range(len(token_ids)):
            for head in range(self.heads):
                lo, hi = head * head_width, (head + 1) * head_width
                scores = [sum(q[pos][c] * k[past][c] for c in range(lo, hi)) / math.sqrt(head_width)
                          for past in range(pos + 1)]
                peak = max(scores)
                probs = [math.exp(score - peak) for score in scores]
                total = sum(probs)
                probs = [prob / total for prob in probs]
                for channel in range(lo, hi):
                    attended[pos][channel] = sum(probs[past] * v[past][channel]
                                                 for past in range(pos + 1))
        x = [[a + b for a, b in zip(row, linear(attended[pos], p["o"]))]
             for pos, row in enumerate(x)]

        # Pre-LayerNorm feed-forward residual with exact GELU via erf.
        ff_input = [layer_norm(row, p["ln2_gain"], p["ln2_bias"]) for row in x]
        ff_output = []
        for row in ff_input:
            hidden = linear(row, p["ff1"], p["ff1_bias"])
            hidden = [0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0))) for value in hidden]
            ff_output.append(linear(hidden, p["ff2"], p["ff2_bias"]))
        x = [[a + b for a, b in zip(row, ff_output[pos])] for pos, row in enumerate(x)]
        x = [layer_norm(row, p["final_gain"], p["final_bias"]) for row in x]

        # Output projection is tied to the input token embedding.
        return [[sum(value * p["token_embedding"][token][j] for j, value in enumerate(row))
                 for token in range(self.vocab)] for row in x]

    def next_token_loss(self, token_ids: list[int]) -> float:
        if len(token_ids) < 2:
            raise ValueError("next-token loss needs at least two token IDs")
        logits = self.forward(token_ids[:-1])
        losses = []
        for row, target in zip(logits, token_ids[1:]):
            peak = max(row)
            log_partition = peak + math.log(sum(math.exp(value - peak) for value in row))
            losses.append(log_partition - row[target])
        return sum(losses) / len(losses)

    def generate(self, prefix: list[int], max_new_tokens: int = 8) -> list[int]:
        out = list(prefix)
        if not out:
            raise ValueError("prefix must contain at least one token ID")
        for _ in range(max_new_tokens):
            context = out[-self.context:]
            logits = self.forward(context)[-1]
            out.append(max(range(self.vocab), key=logits.__getitem__))
        return out

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"config": self.config, "state": self.state}, separators=(",", ":")) + "\n",
                         encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TinyDecoder":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload["config"], payload["state"])


def run(config_path: Path = DEFAULT_CONFIG, output_path: Path = DEFAULT_OUTPUT,
        min_available_mib: int = 2560, repeats: int = 5) -> dict[str, Any]:
    if available_mib() < min_available_mib:
        raise MemoryError(f"MemAvailable below guard of {min_available_mib} MiB")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = TinyDecoder(config)
    corpus = [1, 4, 7, 2, 5, 3, 1, 4, 7, 2]
    if max(corpus) >= config["vocab_size"]:
        raise ValueError("smoke corpus must fit model vocabulary")
    started = time.perf_counter()
    for _ in range(repeats):
        model.forward(corpus[:-1])
    elapsed = time.perf_counter() - started
    loss = model.next_token_loss(corpus)
    generated = model.generate(corpus[:3], max_new_tokens=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output_path.with_name("baseline_checkpoint.json")
    model.save(checkpoint)
    restored = TinyDecoder.load(checkpoint)
    roundtrip_loss = restored.next_token_loss(corpus)
    result = {
        "schema_version": 1, "config": config, "parameter_count": model.parameter_count(),
        "sequence_length": len(corpus) - 1, "batch_size": 1,
        "forward_repeats": repeats, "forward_seconds_total": elapsed,
        "forward_seconds_per_step": elapsed / repeats,
        "tokens_per_second": (len(corpus) - 1) * repeats / elapsed,
        "next_token_loss_nats": loss, "checkpoint_roundtrip_loss_nats": roundtrip_loss,
        "checkpoint_roundtrip_exact_loss": loss == roundtrip_loss,
        "generated_ids": generated, "generated_ids_valid": all(0 <= idx < model.vocab for idx in generated),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "mem_available_mib_at_start": available_mib(), "python": sys.version.split()[0],
        "implementation_dtype": "CPython float (binary64 on this platform)",
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "dependencies": "Python standard library only",
        "training_performed": False,
        "limitations": ["forward-only prototype; no autograd or optimizer installed", "not trained and not a capability result", "batch size 1; one synthetic token sequence", "pure Python timings are not comparable to optimized tensor kernels"],
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-available-mib", type=int, default=2560)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100:
        raise SystemExit("--repeats must be in 1..100")
    print(json.dumps(run(args.config, args.output, args.min_available_mib, args.repeats), indent=2))


if __name__ == "__main__":
    main()
