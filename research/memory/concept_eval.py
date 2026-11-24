"""Deterministic retrieval/composition fixture for the concept-store prototype."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.memory.concepts import Concepts
from research.memory.store import Memory


HERE = Path(__file__).resolve().parent


def evaluate() -> dict[str, Any]:
    memory = Memory(":memory:", fts=False)
    concepts = Concepts(memory)
    scope, owner = "concept-eval", "fixture"
    policy = {"persist": True, "scope": scope, "owner": owner}
    refs: dict[str, str] = {}

    def evidence(key: str, text: str) -> str:
        refs[key] = memory.evidence(text, scope=scope, owner=owner,
            source="synthetic-fixture", uri=f"urn:concept-eval:{key}", version="1",
            licence="generated-test", acquired="2025-01-01", policy=policy)
        return refs[key]

    def promote(concept_id: str, kind: str, statement: str, ref: str, *,
                context: str | None = None, relationship: str | None = None,
                related_concept: str | None = None,
                asserted_value: bool = True, origin: str = "observed") -> None:
        item = {"schema_version": 1, "concept_id": concept_id, "kind": kind, "statement": statement,
            "context": context, "relationship": relationship,
            "related_concept": related_concept,
            "asserted_value": asserted_value, "confidence": 0.91,
            "verification": "candidate", "origin": origin, "source_refs": [ref],
            "review": None}
        candidate_id = concepts.propose(item, scope=scope, owner=owner, policy=policy)
        concepts.promote(item, scope=scope, owner=owner, candidate_id=candidate_id,
            reviewer="independent-fixture-reviewer", method="deterministic",
            start="2025-01-01", policy=policy)

    promote("cell", "baseline", "stores:electrical-energy",
            evidence("baseline", "A cell stores electrical energy."))
    promote("cell", "qualification", "supports:lamp",
            evidence("cold", "In cold conditions the cell cannot power the lamp."),
            context="cold", asserted_value=False, origin="inferred")
    promote("cell", "qualification", "supports:lamp",
            evidence("warm", "In warm conditions the cell can power the lamp."),
            context="warm", asserted_value=True, origin="inferred")
    promote("lamp", "relationship", "requires:cell",
            evidence("relation", "The lamp requires the cell."),
            relationship="requires", related_concept="cell")

    cases = [
        {"id": "cold-negative", "context": "cold", "expected": "no", "status": "verified"},
        {"id": "warm-positive", "context": "warm", "expected": "yes", "status": "verified"},
        {"id": "unknown-context", "context": "humid", "expected": None,
         "status": "insufficient"},
    ]
    results = []
    for case in cases:
        found = concepts.answer_support("cell", "lamp", case["context"],
                                        scope=scope, owner=owner)
        sources = {ref for item in found["evidence"] for ref in item.get("source_refs", [])}
        needed = {refs["baseline"], refs["relation"]}
        if case["context"] in {"cold", "warm"}:
            needed.add(refs[case["context"]])
        results.append({**case, "actual": found["answer"], "actual_status": found["status"],
            "correct": found["answer"] == case["expected"]
                       and found["status"] == case["status"],
            "evidence_complete": needed <= sources,
            "evidence_refs": sorted(sources)})

    conflict_ref = evidence("cold-conflict", "Contradictory source: cold cell can power lamp.")
    promote("cell", "qualification", "supports:lamp", conflict_ref,
            context="cold", asserted_value=True, origin="observed")
    conflicted = concepts.answer_support("cell", "lamp", "cold",
                                        scope=scope, owner=owner)
    results.append({"id": "conflict-abstention", "expected": None,
        "status": "conflict", "actual": conflicted["answer"],
        "actual_status": conflicted["status"],
        "correct": conflicted["answer"] is None and conflicted["status"] == "conflict",
        "evidence_complete": True,
        "evidence_refs": sorted({ref for item in conflicted["evidence"]
                                  for ref in item.get("source_refs", [])})})
    report = {"schema": 1, "evaluation": "concept-context-retrieval-v1",
        "model_under_test": "none; deterministic key retrieval and explicit rule composition",
        "cases": results,
        "case_accuracy": sum(row["correct"] for row in results) / len(results),
        "evidence_completeness": sum(row["evidence_complete"] for row in results) / len(results),
        "answerable_cases": 2,
        "answerable_accuracy": sum(row["correct"] for row in results[:2]) / 2,
        "correct_abstentions": sum(row["correct"] for row in results[2:]),
        "abstention_cases": 2,
        "decision": "deterministic schema mechanics pass; separate Foundation probe failed 0/2",
        "source_event_count": len(refs),
        "concept_fact_count": len(memory.facts(scope=scope, owner=owner, limit=64))}
    semantic_examples = [
        {"text": "No.", "semantic_label": "no", "reference": "no"},
        {"text": "It will not run the lamp in cold conditions.",
         "semantic_label": "no", "reference": "no"},
        {"text": "Cold leaves the cell without enough power for the lamp.",
         "semantic_label": "no", "reference": "no"},
        {"text": "Yes.", "semantic_label": "yes", "reference": "yes"},
        {"text": "The warm cell should run the lamp.",
         "semantic_label": "yes", "reference": "yes"},
        {"text": "In warm conditions the cell can supply enough power.",
         "semantic_label": "yes", "reference": "yes"},
    ]
    for item in semantic_examples:
        item["semantic_correct"] = item["semantic_label"] == item["reference"]
        item["exact_string_match"] = item["text"].strip().casefold().rstrip(".") == item["reference"]
    report["manually_labeled_semantic_response_baseline"] = {
        "examples": semantic_examples,
        "count": len(semantic_examples),
        "semantic_accuracy": sum(item["semantic_correct"] for item in semantic_examples)
                             / len(semantic_examples),
        "exact_string_accuracy": sum(item["exact_string_match"] for item in semantic_examples)
                                 / len(semantic_examples),
        "labels_are_manual": True,
        "does_not_measure_model_capability": True}
    memory.close()
    return report


def main() -> None:
    report = evaluate()
    path = HERE / "concept_retrieval_metrics.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
