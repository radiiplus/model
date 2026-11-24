"""Versioned concept claims stored in the existing evidence/fact lifecycle."""

from __future__ import annotations

import json
from typing import Any

from research.memory.store import Memory, allowed, bounded, encode


KINDS = {"baseline", "qualification", "relationship"}
ORIGINS = {"observed", "inferred"}
STATUSES = {"candidate", "verified", "disputed"}


def concept_key(concept_id: str, kind: str, context: str | None = None,
                relation: str | None = None,
                related_concept: str | None = None) -> str:
    bounded(concept_id, "concept ID", 256)
    if kind == "baseline":
        return f"concept:{concept_id}:baseline"
    if kind == "qualification":
        bounded(context or "", "context", 256)
        return f"concept:{concept_id}:context:{context}"
    if kind == "relationship":
        bounded(relation or "", "relationship type", 256)
        bounded(related_concept or "", "related concept", 256)
        return f"concept:{concept_id}:relationship:{relation}:{related_concept}"
    raise ValueError("unknown concept claim kind")


def validate_claim(value: dict[str, Any]) -> dict[str, Any]:
    required = {"schema_version", "concept_id", "kind", "statement", "context", "relationship",
                "related_concept", "asserted_value", "confidence", "verification", "origin",
                "source_refs", "review"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("concept claim has an invalid schema")
    if value["schema_version"] != 1:
        raise ValueError("unsupported concept claim schema version")
    bounded(value["concept_id"], "concept ID", 256)
    bounded(value["statement"], "concept statement")
    if value["kind"] not in KINDS or value["origin"] not in ORIGINS:
        raise ValueError("concept kind or evidence origin is invalid")
    if value["context"] is not None:
        bounded(value["context"], "context", 256)
    if value["relationship"] is not None:
        bounded(value["relationship"], "relationship type", 256)
    if value["related_concept"] is not None:
        bounded(value["related_concept"], "related concept", 256)
    if value["kind"] == "qualification" and value["context"] is None:
        raise ValueError("contextual qualifications need a context key")
    if value["kind"] == "relationship" and value["relationship"] is None:
        raise ValueError("relationship claims need a relationship type")
    if value["kind"] == "relationship" and value["related_concept"] is None:
        raise ValueError("relationship claims need an endpoint concept")
    if not isinstance(value["asserted_value"], (str, int, float, bool)):
        raise ValueError("asserted value must be a non-null scalar")
    confidence = value["confidence"]
    if (not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
            or not 0 <= confidence <= 1):
        raise ValueError("confidence must be between zero and one")
    if value["verification"] not in STATUSES:
        raise ValueError("verification status is invalid")
    review = value["review"]
    if review is not None and (not isinstance(review, dict)
            or set(review) != {"reviewer", "method", "independent", "passed"}
            or not isinstance(review["reviewer"], str) or not review["reviewer"]
            or review["method"] not in {"human", "deterministic", "evidence"}
            or review["independent"] is not True or review["passed"] is not True):
        raise ValueError("verified concept claims need a passing independent review")
    if value["verification"] == "verified" and review is None:
        raise ValueError("verified concept claims need review provenance")
    refs = value["source_refs"]
    if (not isinstance(refs, list) or not refs
            or any(not isinstance(ref, str) or not ref for ref in refs)):
        raise ValueError("concept claims need source evidence references")
    return {**value, "source_refs": sorted(set(refs)),
            "confidence": float(confidence)}


class Concepts:
    """Concept view over Memory.candidates/facts; source events remain immutable."""

    def __init__(self, memory: Memory):
        self.memory = memory

    def propose(self, claim: dict[str, Any], *, scope: str, owner: str,
                policy: dict[str, Any], private: bool = False) -> str:
        allowed(policy, scope, owner, private)
        item = validate_claim(claim)
        item["verification"] = "candidate"
        return self.memory.candidate(encode(item), item["source_refs"],
                                     scope=scope, owner=owner,
                                     policy=policy, private=private)

    def promote(self, claim: dict[str, Any], *, scope: str, owner: str,
                candidate_id: str, reviewer: str, method: str, start: str,
                policy: dict[str, Any], independent: bool = True,
                replace: bool = False, rollback: str | None = None,
                private: bool = False) -> str:
        allowed(policy, scope, owner, private)
        item = validate_claim(claim)
        candidates = self.memory.candidates(scope=scope, owner=owner, limit=128)
        candidate = next((row for row in candidates if row["id"] == candidate_id), None)
        if candidate is None:
            raise ValueError("concept promotion must reference a stored candidate")
        proposed = json.loads(candidate["claim"])
        expected = {**item, "verification": "candidate", "review": None}
        if proposed != expected:
            raise ValueError("promoted claim differs from its stored candidate")
        if reviewer == owner or independent is not True:
            raise ValueError("concept promotion requires an independent reviewer")
        if method not in {"human", "deterministic", "evidence"}:
            raise ValueError("unsupported concept review method")
        item["verification"] = "verified"
        item["review"] = {"reviewer": reviewer, "method": method,
                          "independent": True, "passed": True}
        item = validate_claim(item)
        if rollback is not None:
            replace = True
        key = concept_key(item["concept_id"], item["kind"], item["context"],
                          item["relationship"], item["related_concept"])
        prior = self.memory.facts(scope=scope, owner=owner, key=key)
        if prior and not replace and rollback is None:
            same = any(json.loads(row["claim"]) == item for row in prior)
            if not same:
                # Preserve both claims so Memory reports their conflict explicitly.
                replace = False
        return self.memory.fact(key, encode(item), item["source_refs"],
            scope=scope, owner=owner, method=method, start=start,
            replace=replace, rollback=rollback, policy=policy, private=private)

    def retrieve(self, concept_id: str, *, scope: str, owner: str,
                 context: str | None = None, limit: int = 64) -> dict[str, Any]:
        keys = [concept_key(concept_id, "baseline")]
        if context is not None:
            keys.append(concept_key(concept_id, "qualification", context))
        rows = []
        for key in keys:
            rows.extend(self.memory.facts(scope=scope, owner=owner, key=key, limit=limit))
        claims = []
        for row in rows:
            try:
                item = json.loads(row["claim"])
            except (TypeError, json.JSONDecodeError):
                continue
            claims.append({**item, "fact_id": row["id"], "fact_version": row["version"],
                           "fact_status": row["status"], "stale": row["stale"],
                           "conflict": row["conflict"], "rollback": row["rollback"]})
        return {"concept_id": concept_id, "context": context, "claims": claims,
                "conflict": any(row["conflict"] for row in claims)}

    def answer_support(self, provider: str, consumer: str, context: str, *,
                       scope: str, owner: str) -> dict[str, Any]:
        """Apply the explicit requires + contextual capability rule; abstain on conflict."""
        provider_data = self.retrieve(provider, scope=scope, owner=owner, context=context)
        consumer_data = self.retrieve(consumer, scope=scope, owner=owner)
        relation_key = concept_key(consumer, "relationship", relation="requires",
                                   related_concept=provider)
        relations = self.memory.facts(scope=scope, owner=owner, key=relation_key)
        relation_claims = []
        for row in relations:
            try:
                relation_claims.append({**json.loads(row["claim"]),
                    "fact_id": row["id"], "fact_status": row["status"],
                    "conflict": row["conflict"]})
            except (TypeError, json.JSONDecodeError):
                pass
        all_claims = provider_data["claims"] + consumer_data["claims"] + relation_claims
        if any(item.get("conflict") for item in all_claims):
            return {"answer": None, "status": "conflict", "evidence": all_claims}
        required = any(item.get("kind") == "relationship"
                       and item.get("concept_id") == consumer
                       and item.get("related_concept") == provider
                       and item.get("relationship") == "requires"
                       and item.get("verification") == "verified"
                       for item in relation_claims)
        baseline = next((item for item in provider_data["claims"]
                         if item.get("kind") == "baseline"
                         and item.get("verification") == "verified"), None)
        capability = next((item for item in provider_data["claims"]
                           if item.get("kind") == "qualification"
                           and item.get("context") == context
                           and item.get("statement") == f"supports:{consumer}"
                           and item.get("verification") == "verified"), None)
        if not required or baseline is None or capability is None:
            return {"answer": None, "status": "insufficient", "evidence": all_claims}
        return {"answer": "yes" if baseline.get("asserted_value") is True
                              and capability.get("asserted_value") is True else "no",
                "status": "verified", "evidence": all_claims}


__all__ = ["Concepts", "concept_key", "validate_claim"]
