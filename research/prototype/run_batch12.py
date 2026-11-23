#!/usr/bin/env python3
"""Run the tiny byte-LM baseline and separate retrieval/verifier ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import torch

from research.experiments.retrieval_controller import DOCS, QUERIES
from research.prototype.data import evaluate_bytes, load_splits
from research.prototype.interfaces import ByteTokenizer
from research.prototype.model import ByteLanguageModel
from research.prototype.system import compare_system_modes, load_fixture
from research.prototype.trainer import (
    MEMORY_GUARD_MIB,
    available_mib,
    configure_torch,
    load_config,
    snapshot,
    train_to,
)


HERE = Path(__file__).resolve().parent
TEXTS = HERE.parent / "experiments/fixtures/tokenizer_corpus.jsonl"
SPLITS = HERE / "splits.json"
CONFIG = HERE / "byte_config.json"
RESULT = HERE.parent / "experiments/results/batch12_tiny.json"


def model_hash(path: Path) -> str:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    h = hashlib.sha256()
    for name, tensor in sorted(payload["model"].items()):
        h.update(name.encode())
        h.update(bytes(tensor.contiguous().reshape(-1).view(torch.uint8).tolist()))
    return h.hexdigest()


def run(output: Path, steps: int = 60, seeds: tuple[int, ...] = (11, 23, 37), repeats: int = 5) -> dict[str, Any]:
    run_started = time.perf_counter()
    state = snapshot()
    start_available = available_mib()
    if start_available < MEMORY_GUARD_MIB:
        raise MemoryError(f"effective available memory below {MEMORY_GUARD_MIB} MiB")
    configure_torch()
    tokenizer = ByteTokenizer()
    encoded, audit = load_splits(TEXTS, SPLITS, tokenizer)
    config = load_config(CONFIG)
    if config["vocab_size"] != tokenizer.vocab_size:
        raise ValueError("model vocabulary must match UTF-8 byte IDs")
    model_runs = []
    for seed in seeds:
        run_config = {**config, "seed": seed}
        initial_model = ByteLanguageModel(run_config, seed).module
        initial_test = evaluate_bytes(initial_model, encoded["test"], int(run_config["context_length"]))
        del initial_model
        checkpoint = output.with_name(f"batch12_byte_seed_{seed}.pt")
        record = train_to(run_config, encoded["train"], encoded["validation"], seed=seed,
                          total_steps=steps, checkpoint_path=checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = ByteLanguageModel(run_config, seed).module
        model.load_state_dict(payload["model"])
        test = evaluate_bytes(model, encoded["test"], int(run_config["context_length"]))
        export_path = output.with_name(f"batch12_byte_seed_{seed}_inference.pt")
        torch.save({"config": run_config, "model": model.state_dict()}, export_path)
        exported = torch.load(export_path, map_location="cpu", weights_only=True)
        restored = ByteLanguageModel(run_config, seed).module
        restored.load_state_dict(exported["model"])
        with torch.inference_mode():
            probe = torch.tensor([tokenizer.encode("A ")], dtype=torch.long)
            export_roundtrip_exact = torch.equal(model(probe), restored(probe))
        record["test"] = test
        record["test_before"] = initial_test
        record["model_state_sha256"] = model_hash(checkpoint)
        record["inference_export_path"] = str(export_path)
        record["inference_export_bytes"] = export_path.stat().st_size
        record["export_roundtrip_exact"] = export_roundtrip_exact
        model_runs.append(record)
        del model, payload, exported, restored

    resume_root = output.with_name("batch12_resume")
    resume_root.mkdir(parents=True, exist_ok=True)
    resume_full = resume_root / "full.pt"
    resume_part = resume_root / "partial.pt"
    train_to({**config, "seed": 79}, encoded["train"], encoded["validation"], seed=79,
             total_steps=20, checkpoint_path=resume_full)
    interrupted = train_to({**config, "seed": 79}, encoded["train"], encoded["validation"], seed=79,
                           total_steps=20, checkpoint_path=resume_part, stop_after_step=10)
    resumed = train_to({**config, "seed": 79}, encoded["train"], encoded["validation"], seed=79,
                       total_steps=20, checkpoint_path=resume_part, resume_from=resume_part)
    resume_exact = model_hash(resume_full) == model_hash(resume_part)

    documents, queries = load_fixture(DOCS, QUERIES)
    system_start = time.perf_counter()
    system_results = compare_system_modes(documents, queries, repeats=repeats)
    system_wall = time.perf_counter() - system_start
    cache_bytes_per_token = (2 * 1 * int(config["n_heads"])
                             * (int(config["d_model"]) // int(config["n_heads"])) * 4)
    cache_bytes_at_context = cache_bytes_per_token * int(config["context_length"])
    result = {
        "schema_version": 1,
        "experiment_id": "batch12-byte-transformer-and-bounded-retrieval",
        "scope": "Byte-level text model training and structured retrieval system are measured separately.",
        "model": {
            "architecture": "preserved TinyGPT architecture with UTF-8 byte vocabulary; separate from Batch 4 vocab-32 checkpoint",
            "config": config, "parameter_count": model_runs[0]["parameter_count"],
            "parameter_count_per_run": [row["parameter_count"] for row in model_runs],
            "tokenizer": tokenizer.name, "kv_cache_bytes_per_token_estimate_fp32": cache_bytes_per_token,
            "kv_cache_bytes_at_context_estimate_fp32": cache_bytes_at_context,
            "kv_cache_implementation": "not implemented for this training prototype; formula estimate only",
            "runs": model_runs,
            "mean_validation_bits_per_byte_before": sum(r["validation_before"]["bits_per_byte"] for r in model_runs) / len(model_runs),
            "mean_validation_bits_per_byte_after": sum(r["validation_after"]["bits_per_byte"] for r in model_runs) / len(model_runs),
            "mean_test_bits_per_byte": sum(r["test"]["bits_per_byte"] for r in model_runs) / len(model_runs),
            "mean_test_next_byte_accuracy": sum(r["test"]["next_byte_accuracy"] for r in model_runs) / len(model_runs),
            "resume_test": {"interrupted_after_step": interrupted["steps_completed"],
                            "resumed_step": resumed["steps_completed"], "final_model_exact": resume_exact,
                            "full_state_sha256": model_hash(resume_full),
                            "resumed_state_sha256": model_hash(resume_part)},
            "data_audit": audit,
        },
        "system_ablation": {
            "fixture_license": "project-authored synthetic CC0-1.0",
            "document_count": len(documents), "query_count": len(queries), "repeats_per_mode": repeats,
            "modes": system_results, "system_ablation_wall_seconds": system_wall,
            "process_peak_rss_bytes_including_prior_model_training": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "system_memory_note": "RSS is the shared process high-water mark; standalone retrieval-only 23,330,816 B is preserved in Batch 8 result.",
            "semantics_note": "Structured resolver/verifier, not generated by or evaluated as a language model.",
            "budget": {"max_retrieval_calls": 2, "max_verifier_calls": 2,
                       "max_context_tokens": 64, "max_output_tokens": 16, "time_budget_seconds": 0.05},
        },
        "environment": {
            "python": sys.version.split()[0], "torch": torch.__version__, "device": "CPU",
            "cuda_available": torch.cuda.is_available(), "torch_threads": torch.get_num_threads(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)), "mem_available_mib_at_start": start_available,
            "memory_guard_mib": MEMORY_GUARD_MIB,
            "memory": state,
            "thread_caps": {key: os.environ.get(key) for key in
                            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        },
        "limitations": [
            "Synthetic corpus has 18 short records; no general language, reasoning or coding claim.",
            "Validation/test examples are held out by document ID and exact/normalized content hash, but the corpus is tiny and authored for representation coverage.",
            "BM25/verifier task answers use structured metadata and do not involve LM generation or natural-language entailment.",
            "No recurrent mixer, SFT, online RL, quantized runtime, or GGUF export was added.",
            "Three seeds are preliminary; model RSS includes PyTorch process overhead and peak is shared across sequential runs.",
        ],
        "run_wall_seconds": time.perf_counter() - run_started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = output.with_name("batch12_comparison.csv")
    fields = ["kind", "condition", "seed", "n", "accuracy", "bits_per_byte", "unsupported_answer_rate",
              "abstention_precision", "abstention_recall", "false_abstention_rate", "mean_context_tokens",
              "mean_latency_ms", "p95_latency_ms", "train_seconds", "seconds_per_step",
              "target_bytes_per_second", "peak_rss_bytes", "parameter_count", "inference_export_bytes",
              "kv_cache_bytes_per_token_estimate_fp32", "kv_cache_bytes_at_context_estimate_fp32"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in model_runs:
            writer.writerow({"kind": "byte_lm", "condition": "tiny_transformer", "seed": row["seed"],
                             "n": row["test"]["target_bytes"], "accuracy": row["test"]["next_byte_accuracy"],
                             "bits_per_byte": row["test"]["bits_per_byte"], "train_seconds": row["train_seconds"],
                             "seconds_per_step": row["seconds_per_step"],
                             "target_bytes_per_second": row["target_bytes_per_second"],
                             "peak_rss_bytes": row["peak_rss_bytes_process_highwater"],
                             "parameter_count": row["parameter_count"],
                             "inference_export_bytes": row["inference_export_bytes"],
                             "kv_cache_bytes_per_token_estimate_fp32": cache_bytes_per_token,
                             "kv_cache_bytes_at_context_estimate_fp32": cache_bytes_at_context})
        for mode, row in system_results.items():
            writer.writerow({"kind": "structured_system", "condition": mode, "n": row["n"],
                             "accuracy": row["task_accuracy"],
                             "unsupported_answer_rate": row["unsupported_answer_rate"],
                             "abstention_precision": row["abstention_precision"],
                             "abstention_recall": row["abstention_recall"],
                             "false_abstention_rate": row["false_abstention_rate"],
                             "mean_context_tokens": row["mean_context_tokens"],
                             "mean_latency_ms": row["mean_latency_ms"],
                             "p95_latency_ms": row["p95_latency_ms"],
                             "peak_rss_bytes": result["system_ablation"]["process_peak_rss_bytes_including_prior_model_training"],
                             "kv_cache_bytes_per_token_estimate_fp32": cache_bytes_per_token,
                             "kv_cache_bytes_at_context_estimate_fp32": cache_bytes_at_context})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.steps <= 120 or not 1 <= args.repeats <= 10 or not args.seeds:
        raise SystemExit("steps 1..120, repeats 1..10, and at least one seed required")
    print(json.dumps(run(args.output, args.steps, tuple(args.seeds), args.repeats), indent=2))


if __name__ == "__main__":
    main()
