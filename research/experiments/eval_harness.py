#!/usr/bin/env python3
"""Small offline evaluator for hand-authored cases and pluggable model outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "experiments" / "fixtures" / "tiny_eval.json"


class Backend(Protocol):
    name: str

    def generate(self, case: dict[str, Any]) -> dict[str, Any]: ...


class PredictionFileBackend:
    """Adapter for one JSON object per case ID; suitable for any model runner."""

    def __init__(self, predictions: dict[str, dict[str, Any]]):
        self.name = "prediction-file"
        self.predictions = predictions

    def generate(self, case: dict[str, Any]) -> dict[str, Any]:
        return self.predictions.get(case["id"], {"answer": "", "abstained": False})


class AbstainBackend:
    name = "deterministic-abstain"

    def generate(self, case: dict[str, Any]) -> dict[str, Any]:
        return {"answer": "", "abstained": True, "claims": [], "confidence": 0.0, "tokens_generated": 0}


class RandomBackend:
    """Seeded answer-pool baseline; deliberately does not inspect expected labels."""

    name = "seeded-random"

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.pool = ["unknown", "0", "1", "2", "yes", "no", "A", "B", "GREEN", "sources_conflict"]

    def generate(self, case: dict[str, Any]) -> dict[str, Any]:
        return {"answer": self.rng.choice(self.pool), "abstained": self.rng.random() < 0.1,
                "confidence": self.rng.random(), "tokens_generated": 1}


class FixtureOracle:
    """Scorer self-check only. Reads labels and must never be treated as a model."""

    name = "fixture-oracle-self-check"

    def generate(self, case: dict[str, Any]) -> dict[str, Any]:
        output = {"answer": case["gold_answer"], "abstained": not case["answerable"],
                  "confidence": 1.0}
        if "gold_claims" in case:
            output["claims"] = case["gold_claims"]
        if case["track"] == "coding":
            output["code"] = case["reference_code"]
        return output


def normalize(text: Any, case_sensitive: bool = False) -> str:
    value = str(text).strip()
    return " ".join((value if case_sensitive else value.casefold()).split())


def exact_match(prediction: Any, gold: Any, case_sensitive: bool = False) -> bool:
    return normalize(prediction, case_sensitive) == normalize(gold, case_sensitive)


def run_code(code: str, test_driver: str, timeout_s: float = 2.0) -> tuple[bool, str]:
    """Run a local fixture solution with a timeout; this is not a security sandbox."""
    with tempfile.TemporaryDirectory(prefix="tiny-eval-") as temp_dir:
        script = Path(temp_dir) / "candidate.py"
        script.write_text(code + "\n" + test_driver, encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"}
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", str(script)], cwd=temp_dir,
                capture_output=True, text=True, timeout=timeout_s, env=env,
                start_new_session=True,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        return completed.returncode == 0, completed.stderr[-1000:]


def score_case(case: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    abstained = bool(output.get("abstained", False))
    if case["track"] == "coding":
        correct, detail = run_code(str(output.get("code", "")), case["test_driver"])
        exact = None
    else:
        strict_format = case["task_type"] == "format_constraint"
        exact = exact_match(output.get("answer", ""), case["gold_answer"], case_sensitive=strict_format)
        correct = bool(abstained and not case["answerable"]) or bool(not abstained and exact)
        detail = ""

    claim_scores: list[bool] = []
    gold_claims = case.get("gold_claims", [])
    matched_gold: set[int] = set()
    for claim in output.get("claims", []):
        text = normalize(claim.get("text", ""))
        evidence_ids = set(claim.get("evidence_ids", []))
        matches = {i for i, gold in enumerate(gold_claims)
                   if text == normalize(gold["text"]) and evidence_ids & set(gold["evidence_ids"])}
        claim_scores.append(bool(matches))
        matched_gold.update(matches)
    gold_count = len(gold_claims)
    supported = sum(claim_scores)
    evidence_precision = supported / len(claim_scores) if claim_scores else (0.0 if gold_count else None)
    evidence_recall = len(matched_gold) / gold_count if gold_count else None

    confidence = output.get("confidence")
    if confidence is not None:
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1] for {case['id']}")
    return {
        "id": case["id"], "track": case["track"], "correct": bool(correct), "exact_match": exact,
        "abstained": abstained, "answerable": bool(case["answerable"]),
        "evidence_claim_precision": evidence_precision, "evidence_claim_recall": evidence_recall,
        "unsupported_claim_count": len(claim_scores) - supported,
        "supported_claim_count": supported, "recall_claim_count": len(matched_gold),
        "gold_claim_count": gold_count,
        "claim_count": len(claim_scores), "confidence": confidence, "detail": detail,
    }


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def require_available_memory(minimum_mib: int) -> None:
    values = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith(("MemAvailable:",)):
            values["available_kib"] = int(line.split()[1])
            break
    if values.get("available_kib", 0) < minimum_mib * 1024:
        raise MemoryError(f"MemAvailable is below configured guard of {minimum_mib} MiB")


def calibration(rows: list[dict[str, Any]], bins: int = 5) -> dict[str, float | int | None]:
    points = [(r["confidence"], float(r["correct"])) for r in rows if r["confidence"] is not None]
    if not points:
        return {"n": 0, "brier": None, "ece": None}
    brier = statistics.mean((p - y) ** 2 for p, y in points)
    weighted_gap = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        group = [(p, y) for p, y in points if lo <= p < hi or (i == bins - 1 and p == 1.0)]
        if group:
            weighted_gap += len(group) / len(points) * abs(statistics.mean(p for p, _ in group) - statistics.mean(y for _, y in group))
    return {"n": len(points), "brier": brier, "ece": weighted_gap}


def summarize(rows: list[dict[str, Any]], latencies: list[float], cpu_s: float, peak_rss_bytes: int) -> dict[str, Any]:
    total = len(rows)
    correct = sum(r["correct"] for r in rows)
    # Abstention is the positive class: it is warranted on unanswerable cases.
    tp = sum(r["abstained"] and not r["answerable"] for r in rows)
    fp = sum(r["abstained"] and r["answerable"] for r in rows)
    fn = sum((not r["abstained"]) and not r["answerable"] for r in rows)
    claims = sum(r["claim_count"] for r in rows)
    unsupported = sum(r["unsupported_claim_count"] for r in rows)
    by_track: dict[str, dict[str, float | int | None]] = {}
    for track in sorted({r["track"] for r in rows}):
        group = [r for r in rows if r["track"] == track]
        by_track[track] = {"n": len(group), "accuracy": ratio(sum(r["correct"] for r in group), len(group))}
    return {
        "n": total,
        "task_accuracy": ratio(correct, total),
        "exact_match_rate": ratio(sum(r["exact_match"] is True for r in rows), sum(r["exact_match"] is not None for r in rows)),
        "abstention_precision": ratio(tp, tp + fp),
        "abstention_recall": ratio(tp, tp + fn),
        "evidence_supported_claim_precision": ratio(sum(r["supported_claim_count"] for r in rows), claims),
        "evidence_supported_claim_recall": ratio(sum(r["recall_claim_count"] for r in rows), sum(r["gold_claim_count"] for r in rows)),
        "unsupported_claim_rate": ratio(unsupported, claims),
        "calibration": calibration(rows),
        "latency_seconds": {"mean": statistics.mean(latencies) if latencies else None,
                            "p50": statistics.median(latencies) if latencies else None,
                            "p95": sorted(latencies)[max(0, math.ceil(0.95 * len(latencies)) - 1)] if latencies else None},
        "tokens_generated": None,
        "peak_rss_bytes": peak_rss_bytes,
        "cpu_seconds": cpu_s,
        "cpu_seconds_per_correct": ratio(0, 0) if correct == 0 else cpu_s / correct,
        "tracks": by_track,
        "limitations": ["whitespace-token estimate is reported per example only when backend supplies no tokenizer count; aggregate token count is null in this first version",
                        "CPU time is a coarse proxy for compute, not energy or FLOPs", "single tiny fixture is not a benchmark score"],
    }


def evaluate(cases: list[dict[str, Any]], backend: Backend, min_available_mib: int = 2560) -> dict[str, Any]:
    rows, latencies = [], []
    tokens_generated = 0
    cpu_start = time.process_time()
    for case in cases:
        require_available_memory(min_available_mib)
        start = time.perf_counter()
        output = backend.generate(case)
        latency = time.perf_counter() - start
        latencies.append(latency)
        tokens = output.get("tokens_generated")
        if tokens is None:
            tokens = len(str(output.get("answer", "")).split()) + len(str(output.get("code", "")).split())
        tokens = int(tokens)
        tokens_generated += tokens
        row = score_case(case, output)
        row["prediction"] = output
        row["latency_seconds"] = latency
        row["tokens_generated"] = tokens
        rows.append(row)
    cpu_s = time.process_time() - cpu_start
    own_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    peak_rss = (own_peak + child_peak) * 1024
    summary = summarize(rows, latencies, cpu_s, peak_rss)
    summary["tokens_generated"] = tokens_generated
    return {"schema_version": 1, "backend": backend.name, "results": rows, "summary": summary}


def load_cases(path: Path = FIXTURES) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictions(path: Path) -> PredictionFileBackend:
    predictions = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            item = json.loads(line)
            if "id" not in item:
                raise ValueError(f"prediction line {line_no} lacks id")
            predictions[item.pop("id")] = item
    return PredictionFileBackend(predictions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=["abstain", "random", "oracle"], default="abstain")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-available-mib", type=int, default=2560)
    args = parser.parse_args()
    require_available_memory(args.min_available_mib)
    run_started = time.perf_counter()
    cases = load_cases()
    if args.predictions:
        backend = load_predictions(args.predictions)
    elif args.baseline == "random":
        backend = RandomBackend(args.seed)
    elif args.baseline == "oracle":
        backend = FixtureOracle()
    else:
        backend = AbstainBackend()
    result = evaluate(cases, backend, args.min_available_mib)
    result["seed"] = args.seed
    result["fixture"] = str(FIXTURES.relative_to(ROOT))
    result["python"] = sys.version.split()[0]
    result["environment"] = {
        "platform": platform.platform(), "logical_cpus_visible": os.cpu_count(),
        "mem_available_kib_at_start": int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))),
        "cgroup_memory_limit": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
        "thread_caps": {key: os.environ.get(key) for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]},
        "fixture_sha256": hashlib.sha256(FIXTURES.read_bytes()).hexdigest(),
    }
    result["run_wall_seconds"] = time.perf_counter() - run_started
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
