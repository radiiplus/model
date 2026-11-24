"""Source-linked user correction capture for the existing experience ledger."""

from __future__ import annotations

import hashlib
from typing import Any

from research.memory.store import Memory, allowed, encode
from research.interaction.compact import Learn


class Corrections:
    """Capture supervised correction episodes without changing model or memory facts."""

    def __init__(self, memory: Memory, learn: Learn):
        self.memory = memory
        self.learn = learn

    def capture(self, prompt: str, response: str, correction: str, *,
                kind: str, meaning_confirmed: bool, scope: str, owner: str,
                policy: dict[str, Any], model: str, version: str,
                dataset: str = "interaction-corrections-v1",
                adapter: str = "none", origin: str = "user") -> dict[str, str]:
        allowed(policy, scope, owner, private=True)
        for value, name, limit in ((prompt, "prompt", 8192),
                                   (response, "response", 4096),
                                   (correction, "correction", 4096),
                                   (model, "model", 256),
                                   (version, "version", 256),
                                   (dataset, "dataset", 256),
                                   (adapter, "adapter", 256)):
            if not isinstance(value, str) or not value or len(value.encode("utf-8")) > limit:
                raise ValueError(f"{name} must be bounded nonempty text")
        if kind not in {"meaning", "expression"}:
            raise ValueError("correction kind must be meaning or expression")
        if origin not in {"user", "synthetic_fixture"}:
            raise ValueError("correction origin must be user or synthetic_fixture")
        if not isinstance(meaning_confirmed, bool):
            raise ValueError("meaning confirmation must be boolean")
        if (kind == "expression" and not meaning_confirmed
                or kind == "meaning" and meaning_confirmed):
            raise ValueError("correction kind and semantic confirmation disagree")
        content = {"schema": 1, "prompt": prompt, "response": response,
                   "correction": correction, "kind": kind,
                   "meaning_confirmed": meaning_confirmed, "origin": origin}
        body = encode(content)
        instance = hashlib.sha256(body.encode("utf-8")).hexdigest()
        event = self.memory.evidence(body, scope=scope, owner=owner,
            source=("user-correction" if origin == "user" else "synthetic-correction-fixture"),
            uri=f"urn:interaction-correction:{instance}", version="1",
            licence=("user-provided" if origin == "user" else "generated-test-data"),
            policy=policy, private=True,
            meta={"kind": kind, "meaning_confirmed": meaning_confirmed,
                  "origin": origin})
        episode = {"family": f"interaction-correction:{kind}", "instance": instance,
            "input": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "actions": [{"type": "user-correction", "kind": kind}],
            "answer": correction,
            "result": {"status": "corrected", "kind": kind,
                       "meaning_confirmed": meaning_confirmed, "origin": origin},
            "cost": {"prompt_chars": len(prompt), "response_chars": len(response),
                     "correction_chars": len(correction)},
            "model": model, "adapter": adapter,
            "strategy": {"loop": "semantic-baseline-then-expression-correction"},
            "dataset": dataset, "version": version, "sources": [event]}
        ident = self.learn.episode(episode, scope=scope, owner=owner,
                                   policy=policy, private=True)
        return {"event": event, "episode": ident, "status": "candidate-experience-only"}


__all__ = ["Corrections"]
