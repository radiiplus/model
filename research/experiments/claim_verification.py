#!/usr/bin/env python3
"""Tiny deterministic evidence and abstention policy experiment.

The fixture encodes evidence relations explicitly. This checks policy mechanics,
not natural-language entailment or model factuality.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import statistics
import time
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "experiments" / "fixtures" / "claim_verification_tiny.json"
MEMORY_GUARD_MIB = 2560
MAX_REVIEW_STEPS = 1


def require_memory(minimum_mib: int = MEMORY_GUARD_MIB) -> int:
    available_kib = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if available_kib < minimum_mib * 1024:
        raise MemoryError(f"MemAvailable below {minimum_mib} MiB guard")
    return available_kib


def current(source: dict[str, Any], as_of: date) -> bool:
    start = date.fromisoformat(source["valid_from"])
    end = date.fromisoformat(source["valid_to"]) if source["valid_to"] else None
    return start <= as_of and (end is None or as_of <= end)


def classify(case: dict[str, Any], as_of: date) -> tuple[str, list[str]]:
    """Classify only from cited, typed relations and date/authority metadata."""
    cited = set(case["candidate_evidence_ids"])
    sources = [s for s in case["sources"] if s["id"] in cited]
    live = [s for s in sources if current(s, as_of)]
    supports = [s for s in live if case["claim_key"] in s["supports"]]
    contradicts = [s for s in live if case["claim_key"] in s["contradicts"]]
    if supports and contradicts:
        return "ambiguous", [s["id"] for s in supports + contradicts]
    if contradicts:
        return "contradicted", [s["id"] for s in contradicts]
    if supports:
        if max(s["authority"] for s in supports) < 0.5:
            return "ambiguous", [s["id"] for s in supports]
        return "supported", [s["id"] for s in supports]
    if any(case["claim_key"] in s["supports"] for s in sources):
        return "stale", [s["id"] for s in sources]
    return "unsupported", [s["id"] for s in sources]


def policy_decision(case: dict[str, Any], mode: str, as_of: date, threshold: float) -> tuple[bool, list[str]]:
    if mode == "single_pass":
        return case["confidence"] >= threshold, list(case["candidate_evidence_ids"])
    if mode in {"retrieval_only", "echo_review"}:
        # Retrieval acceptance checks only that some source was returned.
        # echo_review repeats that decision without independent evidence.
        return bool(case["candidate_evidence_ids"]), list(case["candidate_evidence_ids"])
    if mode == "deterministic_verifier":
        status, citations = classify(case, as_of)
        return status == "supported", citations if status == "supported" else []
    raise ValueError(mode)


def method_metrics(cases: list[dict[str, Any]], mode: str, as_of: date, threshold: float = 0.0) -> dict[str, Any]:
    rows = []
    latencies = []
    accepted = correct = supported = unsupported = citation_correct = total_tokens = 0
    gold_supported = sum(case["expected"] == "supported" for case in cases)
    brier = []
    for case in cases:
        start = time.perf_counter()
        emit, citations = policy_decision(case, mode, as_of, threshold)
        latency = time.perf_counter() - start
        latencies.append(latency)
        actual, valid_ids = classify(case, as_of)
        is_supported = emit and actual == "supported"
        accepted += int(emit)
        correct += int(is_supported or (not emit and case["expected"] != "supported"))
        supported += int(is_supported)
        unsupported += int(emit and actual != "supported")
        citation_correct += int(emit and bool(set(citations) & set(valid_ids)) and actual == "supported")
        tokens = len(case["claim"].split()) if emit else 0
        total_tokens += tokens
        brier.append((case["confidence"] - float(actual == "supported")) ** 2)
        rows.append({
            "id": case["id"], "gold_status": case["expected"], "evidence_status": actual,
            "emitted": emit, "supported_answer": is_supported, "citation_correct": bool(emit and bool(set(citations) & set(valid_ids)) and actual == "supported"),
            "confidence": case["confidence"], "tokens_proxy": tokens, "latency_seconds": latency,
        })
    emitted = accepted
    precision = supported / emitted if emitted else None
    return {
        "n": len(cases), "emitted": emitted, "coverage": emitted / len(cases),
        "supported_claims": supported, "unsupported_assertions": unsupported,
        "claim_precision": precision, "unsupported_claim_rate": unsupported / emitted if emitted else None,
        "supported_claim_recall": supported / gold_supported if gold_supported else None,
        "citation_correct_rate": citation_correct / emitted if emitted else None,
        "correct_abstentions": sum(not row["emitted"] and row["gold_status"] != "supported" for row in rows),
        "false_rejections": sum(not row["emitted"] and row["gold_status"] == "supported" for row in rows),
        "answer_behavior_accuracy": correct / len(cases),
        "brier_confidence_vs_supported": statistics.mean(brier),
        "tokens_proxy": total_tokens,
        "tokens_per_supported_claim": total_tokens / supported if supported else None,
        "latency_seconds_per_supported_claim": sum(row["latency_seconds"] for row in rows if row["supported_answer"]) / supported if supported else None,
        "latency_seconds": {"mean": statistics.mean(latencies), "p95": sorted(latencies)[math.ceil(.95 * len(latencies)) - 1]},
        "rows": rows,
    }


def risk_coverage(cases: list[dict[str, Any]], as_of: date) -> list[dict[str, Any]]:
    output = []
    for threshold in [0.0, 0.7, 0.8, 0.85, 0.9, 0.95]:
        result = method_metrics(cases, "single_pass", as_of, threshold)
        output.append({"confidence_threshold": threshold, "coverage": result["coverage"],
                       "risk_among_emitted": result["unsupported_claim_rate"],
                       "supported_claim_recall": result["supported_claim_recall"],
                       "false_rejections": result["false_rejections"]})
    return output


def run(fixture: dict[str, Any]) -> dict[str, Any]:
    cases = fixture["cases"]
    as_of = date.fromisoformat(fixture["as_of"])
    modes = ["single_pass", "retrieval_only", "echo_review", "deterministic_verifier"]
    results = {mode: method_metrics(cases, mode, as_of) for mode in modes}
    return {
        "schema_version": 1, "experiment_id": "batch9-claim-verification-tiny",
        "fixture_license": fixture["license"], "as_of": fixture["as_of"], "n": len(cases),
        "expected_status_counts": {s: sum(c["expected"] == s for c in cases) for s in sorted({c["expected"] for c in cases})},
        "methods": results, "risk_coverage": risk_coverage(cases, as_of),
        "review_step": {"kind": "echo_control_only", "max_steps": MAX_REVIEW_STEPS,
                        "language_model_backend_available": False,
                        "interpretation": "Not a model-review measurement. It quantifies the no-new-evidence case where a second pass repeats the first decision."},
        "limits": [fixture["description"], "The verifier reads structured gold relation annotations; it does not perform semantic entailment.",
                   "Latency and token counts are Python policy and whitespace-proxy measurements, not generation cost.",
                   "Ten synthetic claims cannot estimate population factuality or guarantee risk on future data."],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-available-mib", type=int, default=MEMORY_GUARD_MIB)
    args = parser.parse_args()
    available = require_memory(args.min_available_mib)
    started = time.perf_counter()
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    result = run(fixture)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result["environment"] = {
        "python": platform.python_version(), "device": "CPU", "logical_cpus_visible": os.cpu_count(),
        "thread_caps": {key: os.environ.get(key) for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]},
        "mem_available_mib_at_start": round(available / 1024), "peak_rss_bytes": usage.ru_maxrss * 1024,
        "run_wall_seconds": time.perf_counter() - started,
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
