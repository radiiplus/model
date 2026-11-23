#!/usr/bin/env python3
"""Offline lexical retrieval, provenance checks, versioned memory, and bounded control."""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "experiments/fixtures/tiny_knowledge_base.json"
QUERIES = ROOT / "experiments/fixtures/tiny_retrieval_queries.json"
STOP = {"a", "an", "the", "is", "of", "in", "on", "at", "to", "what", "when", "where", "does", "did", "which", "are", "now", "its", "it"}
REQUIRED_PROVENANCE = ("source_id", "source_name", "source_uri", "license", "acquired_at", "valid_from", "reliability")


def available_mib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable unavailable; refusing to run without a memory guard")


def tokenize(text: str) -> list[str]:
    tokens = []
    current = []
    for char in text.casefold():
        if char.isalnum() or char in "-_:":
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current.clear()
    if current:
        tokens.append("".join(current))
    return tokens


def valid_provenance(doc: dict[str, Any]) -> bool:
    return all(doc.get(key) not in (None, "") for key in REQUIRED_PROVENANCE)


class LexicalIndex:
    def __init__(self, documents: list[dict[str, Any]]):
        self.documents = {doc["id"]: dict(doc) for doc in documents}
        self.doc_tokens = {doc["id"]: Counter(tokenize(doc["text"])) for doc in documents}
        self.lengths = {doc_id: sum(counts.values()) for doc_id, counts in self.doc_tokens.items()}
        self.avg_len = sum(self.lengths.values()) / max(len(self.lengths), 1)
        self.df: Counter[str] = Counter()
        for counts in self.doc_tokens.values():
            self.df.update(counts.keys())
        self.postings: dict[str, list[str]] = defaultdict(list)
        for doc_id, counts in self.doc_tokens.items():
            for term in counts:
                self.postings[term].append(doc_id)

    def search(self, query: str, method: str = "bm25", k: int = 3, exclude: set[str] | None = None) -> list[dict[str, Any]]:
        terms = [term for term in tokenize(query) if term not in STOP]
        exclude = exclude or set()
        scores: dict[str, float] = defaultdict(float)
        n_docs = len(self.documents)
        for term in terms:
            for doc_id in self.postings.get(term, []):
                if doc_id in exclude:
                    continue
                tf = self.doc_tokens[doc_id][term]
                if method == "overlap":
                    scores[doc_id] += 1.0
                elif method == "bm25":
                    df = self.df[term]
                    idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                    k1, b = 1.2, 0.75
                    norm = tf + k1 * (1 - b + b * self.lengths[doc_id] / max(self.avg_len, 1))
                    scores[doc_id] += idf * tf * (k1 + 1) / norm
                else:
                    raise ValueError(f"unknown retrieval method {method}")
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [{**self.documents[doc_id], "retrieval_score": score} for doc_id, score in ranked]


class VersionedMemory:
    """Explicit, provenance-required writes with version-keyed retrieval cache."""

    def __init__(self, documents: list[dict[str, Any]]):
        self.documents = {doc["id"]: dict(doc) for doc in documents}
        self.version = 1
        self._index = LexicalIndex(list(self.documents.values()))
        self._cache: dict[tuple[int, str, str, int], list[dict[str, Any]]] = {}

    def _commit(self) -> None:
        self.version += 1
        self._index = LexicalIndex(list(self.documents.values()))
        self._cache.clear()

    def add(self, doc: dict[str, Any]) -> None:
        if not valid_provenance(doc):
            raise ValueError("memory write requires source, license, acquisition and validity provenance")
        if doc["id"] in self.documents:
            raise ValueError("document ID already exists; use update")
        self.documents[doc["id"]] = dict(doc)
        self._commit()

    def update(self, doc_id: str, doc: dict[str, Any]) -> None:
        if doc_id not in self.documents:
            raise KeyError(doc_id)
        if doc.get("id") != doc_id or not valid_provenance(doc):
            raise ValueError("update requires same ID and complete provenance")
        self.documents[doc_id] = dict(doc)
        self._commit()

    def delete(self, doc_id: str) -> None:
        del self.documents[doc_id]
        self._commit()

    def search(self, query: str, method: str = "bm25", k: int = 3, exclude: set[str] | None = None):
        key = (self.version, method, query.casefold(), k)
        if not exclude and key in self._cache:
            return [dict(row) for row in self._cache[key]]
        result = self._index.search(query, method, k, exclude)
        if not exclude:
            self._cache[key] = [dict(row) for row in result]
        return result


def canonical_id(doc_id: str, docs: dict[str, dict[str, Any]]) -> str:
    visited = set()
    while docs[doc_id].get("duplicate_of"):
        if doc_id in visited:
            raise ValueError("duplicate provenance cycle")
        visited.add(doc_id)
        parent_id = docs[doc_id]["duplicate_of"]
        if parent_id not in docs:
            return doc_id
        doc_id = parent_id
    return doc_id


def current_documents(query: dict[str, Any], context: list[dict[str, Any]]) -> list[dict[str, Any]]:
    as_of = date.fromisoformat(query["as_of"])
    docs = {doc["id"]: doc for doc in context}
    selected = []
    for doc in context:
        if not valid_provenance(doc) or float(doc["reliability"]) < 0.6:
            continue
        if doc["entity"] != query["entity"] or doc["attribute"] != query["attribute"]:
            continue
        start = date.fromisoformat(doc["valid_from"])
        end = date.fromisoformat(doc["valid_to"]) if doc.get("valid_to") else None
        if start <= as_of and (end is None or as_of <= end):
            selected.append({**doc, "canonical_id": canonical_id(doc["id"], docs)})
    return selected


def verify_context(query: dict[str, Any], context: list[dict[str, Any]], *, final: bool = True) -> dict[str, Any]:
    matches = current_documents(query, context)
    distinct_sources = {doc["source_id"] for doc in matches}
    values = {doc["value"] for doc in matches}
    if len(values) > 1:
        return {"answer": None, "abstained": True, "reason": "conflicting_current_sources",
                "citations": sorted({doc["canonical_id"] for doc in matches}), "needs_more": False}
    if not matches:
        return {"answer": None, "abstained": True, "reason": "no_valid_evidence",
                "citations": [], "needs_more": not final}
    if query.get("require_exhaustive") and len(distinct_sources) < 2 and not final:
        return {"answer": None, "abstained": True, "reason": "need_independent_source",
                "citations": sorted({doc["canonical_id"] for doc in matches}), "needs_more": True}
    value = next(iter(values))
    return {"answer": value, "abstained": False, "reason": "supported_by_current_source",
            "citations": sorted({doc["canonical_id"] for doc in matches}), "needs_more": False}


def token_count(text: str) -> int:
    return len(tokenize(text))


def response_correct(query: dict[str, Any], response: dict[str, Any], context: list[dict[str, Any]]) -> bool:
    if not query["answerable"]:
        return bool(response["abstained"] and response["answer"] is None)
    if response["abstained"] or response["answer"] != query["answer"]:
        return False
    evidence = current_documents(query, context)
    valid_ids = {doc["canonical_id"] for doc in evidence}
    values = {doc["value"] for doc in evidence}
    cited = set(response["citations"])
    return bool(cited) and cited.issubset(valid_ids) and len(values) == 1 and response["answer"] in values


def answer_mode(query: dict[str, Any], docs: list[dict[str, Any]], mode: str,
                index: LexicalIndex | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if mode == "none":
        context = []
    elif mode == "full_context":
        context = list(docs)
    elif mode.startswith("retrieved_"):
        method = mode.removeprefix("retrieved_")
        index = index or LexicalIndex(docs)
        context = index.search(query["text"], method=method, k=3)
    else:
        raise ValueError(mode)
    response = verify_context(query, context, final=True)
    return response, context, {"cycles": 1 if context else 0, "retrieval_calls": 1 if mode.startswith("retrieved_") else 0}


def bounded_controller(query: dict[str, Any], memory: VersionedMemory, *, method: str = "bm25",
                       max_cycles: int = 2, max_context_tokens: int = 64,
                       max_output_tokens: int = 16, time_budget_seconds: float = 0.05) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    context: list[dict[str, Any]] = []
    seen: set[str] = set()
    used_tokens, cycles, verifier_calls = 0, 0, 0
    status = {"answer": None, "abstained": True, "reason": "cycle_budget_exhausted", "citations": [], "needs_more": True}
    for cycle in range(max_cycles):
        if time.perf_counter() - started >= time_budget_seconds:
            status["reason"] = "time_budget_exhausted"
            break
        retrieved = memory.search(query["text"], method=method, k=1, exclude=seen)
        cycles += 1
        if not retrieved:
            verifier_calls += 1
            status = verify_context(query, context, final=True)
            break
        doc = retrieved[0]
        seen.add(doc["id"])
        size = token_count(doc["text"])
        if used_tokens + size > max_context_tokens:
            status["reason"] = "context_budget_exhausted"
            break
        context.append(doc)
        used_tokens += size
        final = cycle == max_cycles - 1
        verifier_calls += 1
        status = verify_context(query, context, final=final)
        if not status["needs_more"]:
            break
    output_tokens = token_count(str(status.get("answer") or "")) + (1 if status.get("abstained") else 0)
    if output_tokens > max_output_tokens:
        status = {"answer": None, "abstained": True, "reason": "output_budget_exhausted", "citations": [], "needs_more": False}
        output_tokens = 1
    metadata = {"cycles": cycles, "retrieval_calls": cycles, "verifier_calls": verifier_calls,
                "context_tokens": used_tokens,
                "output_tokens": output_tokens, "elapsed_seconds": time.perf_counter() - started,
                "max_cycles": max_cycles, "max_context_tokens": max_context_tokens,
                "max_output_tokens": max_output_tokens, "time_budget_seconds": time_budget_seconds}
    return status, context, metadata


def retrieval_metrics(queries: list[dict[str, Any]], index: LexicalIndex, method: str, k: int) -> dict[str, Any]:
    recalls, precisions, rows = [], [], []
    for query in queries:
        gold = set(query["required_evidence"])
        if not gold:
            continue
        hits = gold & {doc["id"] for doc in index.search(query["text"], method, k)}
        recalls.append(len(hits) / len(gold))
        results = index.search(query["text"], method, k)
        precisions.append(len(hits) / k)
        rows.append({"query_id": query["id"], "hits": sorted(hits), "retrieved_ids": [doc["id"] for doc in results]})
    return {"k": k, "queries_with_gold": len(recalls), "recall_at_k": statistics.mean(recalls) if recalls else None,
            "precision_at_k": statistics.mean(precisions) if precisions else None, "per_query": rows}


def evaluate_modes(docs: list[dict[str, Any]], queries: list[dict[str, Any]], repeats: int = 5) -> dict[str, Any]:
    memory = VersionedMemory(docs)
    index = memory._index
    modes = ("none", "full_context", "retrieved_overlap", "retrieved_bm25", "bounded_controller")
    output = {}
    for mode in modes:
        rows, latencies, contexts = [], [], []
        for query in queries:
            query_responses = []
            for _ in range(repeats):
                start = time.perf_counter()
                if mode == "bounded_controller":
                    response, context, control = bounded_controller(query, memory)
                else:
                    response, context, control = answer_mode(query, docs, mode, index=index)
                elapsed = time.perf_counter() - start
                query_responses.append((response, context, control))
                latencies.append(elapsed)
            response, context, control = query_responses[0]
            correct = response_correct(query, response, context)
            contexts.append(sum(token_count(doc["text"]) for doc in context))
            rows.append({"query_id": query["id"], "correct": correct, "answer": response["answer"],
                         "abstained": response["abstained"], "reason": response["reason"],
                         "citations": response["citations"], "context_doc_ids": [d["id"] for d in context],
                         "context_tokens": contexts[-1], **control})
        output[mode] = {"accuracy": sum(row["correct"] for row in rows) / len(rows),
                        "correct": sum(row["correct"] for row in rows), "n": len(rows),
                        "mean_context_tokens": statistics.mean(contexts),
                        "mean_latency_ms": 1000 * statistics.mean(latencies),
                        "p50_latency_ms": 1000 * statistics.median(latencies),
                        "p95_latency_ms": 1000 * sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1],
                        "evidence_coverage": (sum(len(set(q["required_evidence"]) & {d["id"] for d in (
                            docs if mode == "full_context" else (rows[i]["context_doc_ids"] and
                            [memory.documents[doc_id] for doc_id in rows[i]["context_doc_ids"]]) or [])})
                            for i, q in enumerate(queries) if q["required_evidence"]) /
                            sum(len(q["required_evidence"]) for q in queries if q["required_evidence"])),
                        "per_query": rows}
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=Path, default=DOCS)
    parser.add_argument("--queries", type=Path, default=QUERIES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if available_mib() < 2560:
        raise SystemExit("MemAvailable below 2560 MiB guard")
    if not 1 <= args.repeats <= 20:
        raise SystemExit("repeats must be in 1..20")
    docs, queries = json.loads(args.docs.read_text()), json.loads(args.queries.read_text())
    start = time.perf_counter()
    index = LexicalIndex(docs)
    retrieval = {method: retrieval_metrics(queries, index, method, 3) for method in ("overlap", "bm25")}
    modes = evaluate_modes(docs, queries, args.repeats)
    memory = VersionedMemory(docs)
    controller_checks = [bounded_controller(q, memory)[2] for q in queries]
    result = {"schema_version": 1, "experiment": "tiny_provenance_retrieval_and_bounded_controller",
              "fixture": {"document_count": len(docs), "query_count": len(queries),
                          "document_sha256": __import__("hashlib").sha256(args.docs.read_bytes()).hexdigest(),
                          "query_sha256": __import__("hashlib").sha256(args.queries.read_bytes()).hexdigest(),
                          "license": "project-authored synthetic records; CC0-1.0"},
              "retrieval_metrics": retrieval, "answer_modes": modes,
              "controller_budget_audit": {"max_cycles_observed": max(x["cycles"] for x in controller_checks),
                                            "max_verifier_calls_observed": max(x["verifier_calls"] for x in controller_checks),
                                            "max_context_tokens_observed": max(x["context_tokens"] for x in controller_checks),
                                            "max_elapsed_seconds_observed": max(x["elapsed_seconds"] for x in controller_checks),
                                            "calls_within_fixed_limits": all(x["cycles"] <= 2 and x["verifier_calls"] <= 2 and x["context_tokens"] <= 64 and
                                                x["output_tokens"] <= 16 and x["elapsed_seconds"] <= 0.05 for x in controller_checks)},
              "index_postings_shallow_bytes": sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in index.postings.items()) +
                  sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in index.doc_tokens.items()),
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "run_seconds": time.perf_counter() - start,
              "environment": {"python": sys.version.split()[0], "cpu_threads": 1,
                              "mem_available_mib_at_start": available_mib(),
                              "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None},
              "embedding_status": "not run; no local sentence embedding model discovered; no model downloaded",
              "limitations": ["synthetic facts and structured answer resolver, not a language model", "whitespace/regex token count is a proxy, not model tokenizer tokens",
                              "index postings memory is a shallow estimate; peak RSS includes interpreter and process overhead",
                              "tiny corpus is not a retrieval-scale benchmark", "recall does not establish answer support; exact verifier checks provenance and validity"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "retrieval": {m: {k:v for k,v in x.items() if k != "per_query"}
        for m,x in retrieval.items()}, "modes": {m: {k:v for k,v in x.items() if k != "per_query"}
        for m,x in modes.items()}, "run_seconds": result["run_seconds"], "peak_rss_bytes": result["peak_rss_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
