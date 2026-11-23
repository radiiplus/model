#!/usr/bin/env python3
"""CPU profile of the preserved TinyGPT, including a valid dynamic INT8 path."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from tiny_gpt import TinyGPT, load_checkpoint


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "experiments" / "results" / "baseline_tiny_gpt.pt"
DEFAULT_RESULT = ROOT / "experiments" / "results" / "inference_profile_tiny.json"
DEFAULT_FP32_FILE = ROOT / "experiments" / "results" / "inference_tiny_fp32.pt"
DEFAULT_INT8_FILE = ROOT / "experiments" / "results" / "inference_tiny_dynamic_int8.pt"
MIN_AVAILABLE_MIB = 2560


def require_memory() -> int:
    available = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:")) // 1024
    if available < MIN_AVAILABLE_MIB:
        raise MemoryError(f"MemAvailable below {MIN_AVAILABLE_MIB} MiB")
    return available


def cache_bytes(batch: int, length: int, model: TinyGPT, dtype: torch.dtype = torch.float32) -> int:
    # One Transformer block; store K and V as [batch, heads, time, head_dim].
    element_bytes = torch.empty((), dtype=dtype).element_size()
    return 2 * batch * model.block.attn.n_heads * length * model.block.attn.head_dim * element_bytes


@torch.inference_mode()
def prefill(model: TinyGPT, token_ids: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    batch, length = token_ids.shape
    positions = torch.arange(length, device=token_ids.device)
    x = model.token_embedding(token_ids) + model.position_embedding(positions)[None, :, :]
    z = model.block.ln_attn(x)
    qkv = model.block.attn.qkv(z).view(batch, length, 3, model.block.attn.n_heads,
                                       model.block.attn.head_dim)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(model.block.attn.head_dim)
    future = torch.triu(torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1)
    weights = F.softmax(scores.masked_fill(future, torch.finfo(scores.dtype).min), dim=-1)
    attended = (weights @ v).transpose(1, 2).contiguous().view(batch, length, model.config["d_model"])
    x = x + model.block.attn.proj(attended)
    x = x + model.block.mlp(model.block.ln_mlp(x))
    logits = F.linear(model.final_norm(x), model.token_embedding.weight)
    return logits, (k, v)


@torch.inference_mode()
def decode_step(model: TinyGPT, token: torch.Tensor, position: int,
                cache: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    k_cache, v_cache = cache
    x = model.token_embedding(token) + model.position_embedding(torch.tensor([position], device=token.device))[None, :, :]
    z = model.block.ln_attn(x)
    qkv = model.block.attn.qkv(z).view(token.shape[0], 1, 3, model.block.attn.n_heads,
                                       model.block.attn.head_dim)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    k_cache = torch.cat((k_cache, k), dim=2)
    v_cache = torch.cat((v_cache, v), dim=2)
    scores = (q @ k_cache.transpose(-2, -1)) / math.sqrt(model.block.attn.head_dim)
    weights = F.softmax(scores, dim=-1)
    attended = (weights @ v_cache).transpose(1, 2).contiguous().view(token.shape[0], 1, model.config["d_model"])
    x = x + model.block.attn.proj(attended)
    x = x + model.block.mlp(model.block.ln_mlp(x))
    logits = F.linear(model.final_norm(x), model.token_embedding.weight)
    return logits, (k_cache, v_cache)


def build_cached_model(model: TinyGPT) -> dict[str, Any]:
    return {"config": model.config, "state_dict": model.state_dict()}


def quantize_model(model: TinyGPT) -> TinyGPT:
    model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    model.eval()
    return model


def make_model(payload: dict[str, Any], quantized: bool) -> TinyGPT:
    model = TinyGPT(payload["config"]).cpu().eval()
    if quantized:
        model = quantize_model(model)
    model.load_state_dict(payload["state_dict"])
    return model


def save_state(path: Path, payload: dict[str, Any]) -> int:
    torch.save(payload, path)
    return path.stat().st_size


def load_export(path: Path, quantized: bool) -> tuple[TinyGPT, float]:
    start = time.perf_counter()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = make_model(payload, quantized)
    return model, time.perf_counter() - start


def timed(fn, repeats: int = 12, warmups: int = 3) -> dict[str, float]:
    for _ in range(warmups):
        fn()
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        values.append(time.perf_counter() - start)
    return {"mean_seconds": statistics.mean(values), "median_seconds": statistics.median(values),
            "p95_seconds": sorted(values)[math.ceil(.95 * len(values)) - 1]}


@torch.inference_mode()
def accuracy_and_logits(model: TinyGPT, batches: list[torch.Tensor]) -> tuple[int, int, list[torch.Tensor]]:
    correct = total = 0
    outputs = []
    for ids in batches:
        logits = model(ids[:, :-1])
        correct += int((logits.argmax(dim=-1) == ids[:, 1:]).sum())
        total += ids[:, 1:].numel()
        outputs.append(logits.float())
    return correct, total, outputs


def benchmark_model(model: TinyGPT, repeats: int) -> dict[str, Any]:
    result = {"prefill": {}, "decode": {}}
    for length in [8, 16, 24, 32]:
        ids = torch.tensor([[(i * 7 + 3) % model.vocab_size for i in range(length)]], dtype=torch.long)
        result["prefill"][str(length)] = timed(lambda: model(ids), repeats=repeats)
        if length < model.context_length:
            n_transitions = min(4, model.context_length - length)

            def prepare_cached():
                logits, cache = prefill(model, ids)
                token = logits[:, -1].argmax(dim=-1, keepdim=True)
                return token, cache

            def cached_transitions(token, cache):
                for offset in range(n_transitions):
                    logits, cache = decode_step(model, token, length + offset, cache)
                    token = logits[:, -1].argmax(dim=-1, keepdim=True)

            def recompute_transitions(token):
                current = torch.cat((ids, token), dim=1)
                for _ in range(n_transitions):
                    logits = model(current[:, -model.context_length:])
                    token = logits[:, -1].argmax(dim=-1, keepdim=True)
                    current = torch.cat((current, token), dim=1)

            cached_times, recompute_times = [], []
            for _ in range(2):
                token, cache = prepare_cached()
                cached_transitions(token, cache)
                recompute_transitions(token)
            for _ in range(repeats):
                token, cache = prepare_cached()
                start = time.perf_counter()
                cached_transitions(token, cache)
                cached_times.append(time.perf_counter() - start)
                token, _ = prepare_cached()
                start = time.perf_counter()
                recompute_transitions(token)
                recompute_times.append(time.perf_counter() - start)
            cache_timing = {"mean_seconds": statistics.mean(cached_times),
                            "median_seconds": statistics.median(cached_times),
                            "p95_seconds": sorted(cached_times)[math.ceil(.95 * len(cached_times)) - 1]}
            recompute_timing = {"mean_seconds": statistics.mean(recompute_times),
                                "median_seconds": statistics.median(recompute_times),
                                "p95_seconds": sorted(recompute_times)[math.ceil(.95 * len(recompute_times)) - 1]}
            result["decode"][str(length)] = {
                "decode_transitions_per_run": n_transitions,
                "cached_decode_only": cache_timing,
                "cached_decode_seconds_per_token": cache_timing["mean_seconds"] / n_transitions,
                "cached_decode_tokens_per_second": n_transitions / cache_timing["mean_seconds"],
                "full_prefix_recompute_decode_only": recompute_timing,
                "full_prefix_recompute_seconds_per_token": recompute_timing["mean_seconds"] / n_transitions,
                "full_prefix_recompute_tokens_per_second": n_transitions / recompute_timing["mean_seconds"],
                "kv_cache_bytes_at_prefix": cache_bytes(1, length, model),
            }

    # Reuse one prefilled prefix for two alternative one-token branches.
    prefix = torch.tensor([[1, 4, 7, 2, 5, 3, 1, 4]], dtype=torch.long)
    branch_tokens = [torch.tensor([[6]]), torch.tensor([[9]])]

    def reuse_prefix_cache():
        _, base_cache = prefill(model, prefix)
        for branch in branch_tokens:
            cloned = tuple(t.clone() for t in base_cache)
            decode_step(model, branch, prefix.shape[1], cloned)

    def recompute_branches():
        for branch in branch_tokens:
            model(torch.cat((prefix, branch), dim=1))

    result["shared_prefix_two_branches"] = {
        "reuse_one_prefill_then_clone_kv": timed(reuse_prefix_cache, repeats=repeats),
        "recompute_both_prefixes": timed(recompute_branches, repeats=repeats),
        "prefix_length": int(prefix.shape[1]), "branches": len(branch_tokens),
        "cache_bytes_per_branch": cache_bytes(1, prefix.shape[1], model),
    }
    return result


def run(checkpoint: Path, output: Path, fp32_file: Path, int8_file: Path, repeats: int) -> dict[str, Any]:
    run_started = time.perf_counter()
    available = require_memory()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    trained, _ = load_checkpoint(checkpoint)
    trained.eval()
    raw = build_cached_model(trained)
    fp32_bytes = save_state(fp32_file, raw)
    fp32, fp32_load_s = load_export(fp32_file, quantized=False)
    int8 = quantize_model(make_model(raw, quantized=False))
    buffer = io.BytesIO()
    torch.save({"config": raw["config"], "state_dict": int8.state_dict()}, buffer)
    int8_file.write_bytes(buffer.getvalue())
    int8_bytes = int8_file.stat().st_size
    int8_reloaded, int8_load_s = load_export(int8_file, quantized=True)

    train_tokens = torch.tensor([[1, 4, 7, 2, 5, 3] * 2], dtype=torch.long)
    batches = [train_tokens] + [torch.tensor([[((i + j) * 7 + 3) % fp32.vocab_size for i in range(9 + j)]], dtype=torch.long)
                                for j in range(5)]
    fp_correct, total, fp_logits = accuracy_and_logits(fp32, batches)
    q_correct, _, q_logits = accuracy_and_logits(int8_reloaded, batches)
    diffs = [torch.abs(a - b) for a, b in zip(fp_logits, q_logits)]
    logits = {"mean_absolute_error": statistics.mean(float(d.mean()) for d in diffs),
              "max_absolute_error": max(float(d.max()) for d in diffs),
              "argmax_agreement": sum(int((a.argmax(-1) == b.argmax(-1)).sum()) for a,b in zip(fp_logits,q_logits)) / total,
              "fp32_next_token_accuracy": fp_correct / total,
              "int8_next_token_accuracy": q_correct / total,
              "accuracy_delta": (q_correct - fp_correct) / total,
              "evaluated_target_tokens": total,
              "scope": "one memorized training sequence plus five deterministic token probes; not held-out language quality"}
    file_size = {"fp32_inference_checkpoint_bytes": fp32_bytes, "dynamic_int8_checkpoint_bytes": int8_bytes,
                 "checkpoint_size_ratio_int8_to_fp32": int8_bytes / fp32_bytes,
                 "checkpoint_size_reduction_fraction": 1 - int8_bytes / fp32_bytes,
                 "original_training_checkpoint_bytes": checkpoint.stat().st_size,
                 "fp32_state_dict_serialized_bytes": len((lambda b: (torch.save(raw["state_dict"], b), b.getvalue())[1])(io.BytesIO())),
                 "int8_state_dict_serialized_bytes": len((lambda b: (torch.save(int8.state_dict(), b), b.getvalue())[1])(io.BytesIO()))}

    # Cached/full-recompute numerical equivalence for one prefix and one next token.
    ids = torch.tensor([[1, 4, 7, 2, 5, 3, 1, 4]], dtype=torch.long)
    full = fp32(ids)
    prefilled, cache = prefill(fp32, ids)
    branch = torch.tensor([[6]])
    cached_next, _ = decode_step(fp32, branch, ids.shape[1], cache)
    full_next = fp32(torch.cat((ids, branch), dim=1))[:, -1:]
    cache_equivalence = {"prefill_max_abs_error": float((full - prefilled).abs().max().detach()),
                         "next_step_max_abs_error": float((cached_next - full_next).abs().max().detach()),
                         "cache_dtype": str(cache[0].dtype),
                         "cache_shapes": [list(cache[0].shape), list(cache[1].shape)]}

    result = {
        "schema_version": 1, "experiment_id": "batch10-tinygpt-cpu-inference",
        "model": {"checkpoint": str(checkpoint), "parameter_count": sum(p.numel() for p in fp32.parameters()),
                  "context_length": fp32.context_length, "batch_size": 1},
        "quantization": {"method": "PyTorch CPU dynamic qint8 for nn.Linear modules", "backend": torch.backends.quantized.engine,
                         "quantized_modules": ["attn.qkv", "attn.proj", "mlp.fc_in", "mlp.fc_out"],
                         "not_quantized": ["token/position embeddings", "tied output projection", "LayerNorm", "activations", "KV cache"],
                         "api_note": "torch.ao.quantization.quantize_dynamic emitted a deprecation warning in installed PyTorch; use only as a measured compatibility probe."},
        "file_sizes": file_size, "logit_and_task_comparison": logits,
        "cold_model_load_seconds": {"fp32_load_from_small_local_file": fp32_load_s,
                                    "dynamic_int8_load_including_requantize_wrapping": int8_load_s,
                                    "caveat": "First import/process cold start not measured; local files are small and OS-cache state is warm."},
        "fp32_runtime": benchmark_model(fp32, repeats),
        "dynamic_int8_runtime": benchmark_model(int8_reloaded, repeats),
        "cache_equivalence": cache_equivalence,
        "cache_estimates": {str(n): cache_bytes(1, n, fp32) for n in [8,16,24,32]},
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "device": "CPU",
                        "cuda_available": torch.cuda.is_available(), "torch_threads": torch.get_num_threads(),
                        "cpu_affinity": sorted(os.sched_getaffinity(0)), "mem_available_mib_at_start": available,
                        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                        "thread_caps": {key: os.environ.get(key) for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]}},
        "run_wall_seconds": time.perf_counter() - run_started,
        "limitations": ["Tiny one-layer model and synthetic token probes only", "Torch eager dynamic INT8 path is not GGUF/GGML and does not quantize embeddings, tied head, activations or KV cache", "No held-out natural-language quality measurement", "No serving concurrency, long-context, energy or device-portability claims"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--fp32-file", type=Path, default=DEFAULT_FP32_FILE)
    parser.add_argument("--int8-file", type=Path, default=DEFAULT_INT8_FILE)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    if not 3 <= args.repeats <= 30:
        raise SystemExit("--repeats must be in 3..30")
    result = run(args.checkpoint, args.output, args.fp32_file, args.int8_file, args.repeats)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
