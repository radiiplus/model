"""Bounded retrieval-system adapter; it never treats search rank as evidence support."""

from __future__ import annotations

import json
import math
import resource
import statistics
import time
from pathlib import Path
from typing import Any

from research.experiments.retrieval_controller import (
    VersionedMemory,
    bounded_controller,
    response_correct,
    token_count,
    verify_context,
)


class StructuredEvidenceVerifier:
    name = "structured-fixture-verifier"

    def verify(self, query: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
        return verify_context(query, evidence, final=True)


class OfflineEvidenceSystem:
    """Evidence-only system path; answer resolution is structured, not LM generation."""

    def __init__(self, documents: list[dict[str, Any]], *, max_cycles: int = 2,
                 max_context_tokens: int = 64, max_output_tokens: int = 16,
                 time_budget_seconds: float = 0.05):
        self.documents = documents
        self.memory = VersionedMemory(documents)
        self.verifier = StructuredEvidenceVerifier()
        self.max_cycles = max_cycles
        self.max_context_tokens = max_context_tokens
        self.max_output_tokens = max_output_tokens
        self.time_budget_seconds = time_budget_seconds

    def answer(self, query: dict[str, Any], mode: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        start = time.perf_counter()
        if mode == "none":
            evidence: list[dict[str, Any]] = []
            result = self.verifier.verify(query, evidence)
            meta = {"retrieval_calls": 0, "verifier_calls": 1, "context_tokens": 0, "cycles": 1}
        elif mode == "full_context":
            evidence = list(self.documents)
            result = self.verifier.verify(query, evidence)
            meta = {"retrieval_calls": 0, "verifier_calls": 1,
                    "context_tokens": sum(token_count(d["text"]) for d in evidence), "cycles": 1}
        elif mode == "bm25_top3":
            evidence = self.memory.search(query["text"], method="bm25", k=3)
            result = self.verifier.verify(query, evidence)
            meta = {"retrieval_calls": 1, "verifier_calls": 1,
                    "context_tokens": sum(token_count(d["text"]) for d in evidence), "cycles": 1}
        elif mode == "bm25_unverified":
            evidence = self.memory.search(query["text"], method="bm25", k=1)
            result = ({"answer": evidence[0]["value"], "abstained": False,
                       "reason": "retrieved_without_verification", "citations": [evidence[0]["id"]],
                       "needs_more": False} if evidence else
                      {"answer": None, "abstained": True, "reason": "no_retrieval_result",
                       "citations": [], "needs_more": False})
            meta = {"retrieval_calls": 1, "verifier_calls": 0,
                    "context_tokens": sum(token_count(d["text"]) for d in evidence), "cycles": 1}
        elif mode == "bounded_bm25":
            result, evidence, meta = bounded_controller(
                query, self.memory, method="bm25", max_cycles=self.max_cycles,
                max_context_tokens=self.max_context_tokens,
                max_output_tokens=self.max_output_tokens,
                time_budget_seconds=self.time_budget_seconds,
            )
        else:
            raise ValueError(f"unknown system mode: {mode}")
        meta = {**meta, "elapsed_seconds": time.perf_counter() - start, "mode": mode}
        return result, evidence, meta


def load_fixture(doc_path: Path, query_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return json.loads(doc_path.read_text()), json.loads(query_path.read_text())


def compare_system_modes(documents: list[dict[str, Any]], queries: list[dict[str, Any]], repeats: int = 5) -> dict[str, Any]:
    system = OfflineEvidenceSystem(documents)
    modes = ("none", "full_context", "bm25_unverified", "bm25_top3", "bounded_bm25")
    comparison: dict[str, Any] = {}
    for mode in modes:
        rows = []
        latencies: list[float] = []
        for query in queries:
            outputs = []
            for _ in range(repeats):
                start = time.perf_counter()
                output, evidence, meta = system.answer(query, mode)
                latencies.append(time.perf_counter() - start)
                outputs.append((output, evidence, meta))
            output, evidence, meta = outputs[0]
            is_correct = response_correct(query, output, evidence)
            rows.append({
                "id": query["id"], "correct": is_correct, "answerable": query["answerable"],
                "abstained": output["abstained"], "answer": output["answer"], "reason": output["reason"],
                "citations": output["citations"], "context_tokens": meta["context_tokens"],
                "retrieval_calls": meta["retrieval_calls"], "verifier_calls": meta["verifier_calls"],
            })
        answerable = [row for row in rows if row["answerable"]]
        unanswerable = [row for row in rows if not row["answerable"]]
        emitted = [row for row in rows if not row["abstained"]]
        abstained = [row for row in rows if row["abstained"]]
        bad_emitted = [row for row in emitted if not row["correct"]]
        comparison[mode] = {
            "n": len(rows), "task_accuracy": sum(row["correct"] for row in rows) / len(rows),
            "answerable_accuracy": sum(row["correct"] for row in answerable) / max(len(answerable), 1),
            "abstention_precision": (sum(row["correct"] and row["abstained"] for row in rows) / len(abstained)) if abstained else None,
            "abstention_recall": sum(row["correct"] and row["abstained"] for row in unanswerable) / max(len(unanswerable), 1),
            "false_abstention_rate": sum(row["abstained"] for row in answerable) / max(len(answerable), 1),
            "unsupported_answer_rate": (len(bad_emitted) / len(emitted)) if emitted else None,
            "mean_context_tokens": statistics.mean(row["context_tokens"] for row in rows),
            "mean_latency_ms": 1000 * statistics.mean(latencies),
            "p95_latency_ms": 1000 * sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1],
            "max_retrieval_calls": max(row["retrieval_calls"] for row in rows),
            "max_verifier_calls": max(row["verifier_calls"] for row in rows),
            "correct": sum(row["correct"] for row in rows), "rows": rows,
        }
    return comparison
