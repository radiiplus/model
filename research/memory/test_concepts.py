from __future__ import annotations

import json
import unittest

from research.memory.concepts import Concepts
from research.memory.store import Memory


class ConceptStore(unittest.TestCase):
    def setUp(self):
        self.memory = Memory(":memory:", fts=False)
        self.store = Concepts(self.memory)
        self.scope, self.owner = "concept-test", "researcher"
        self.policy = {"persist": True, "scope": self.scope, "owner": self.owner}
        self.refs = []

    def tearDown(self):
        self.memory.close()

    def evidence(self, text: str) -> str:
        event = self.memory.evidence(text, scope=self.scope, owner=self.owner,
            source="fixture", uri="urn:fixture:" + str(len(self.refs)), version="1",
            licence="test", policy=self.policy)
        self.refs.append(event)
        return event

    def claim(self, concept, kind, statement, ref, *, context=None,
              relationship=None, related_concept=None, asserted_value=True, origin="observed",
              verification="candidate", confidence=0.9):
        return {"schema_version": 1, "concept_id": concept, "kind": kind, "statement": statement,
            "context": context, "relationship": relationship,
            "related_concept": related_concept,
            "asserted_value": asserted_value, "confidence": confidence,
            "verification": verification, "origin": origin, "source_refs": [ref],
            "review": None}

    def promote(self, claim, start="2025-01-01"):
        candidate = self.store.propose(claim, scope=self.scope, owner=self.owner,
                                       policy=self.policy)
        return self.store.promote(claim, scope=self.scope, owner=self.owner,
            candidate_id=candidate, reviewer="independent-reviewer", method="human",
            start=start, policy=self.policy)

    def test_baseline_and_context_are_separate_and_combine_with_relation(self):
        base_ref = self.evidence("A cell stores electrical energy.")
        cold_ref = self.evidence("In cold conditions, this cell cannot power a lamp.")
        relation_ref = self.evidence("The lamp requires the cell.")
        self.promote(self.claim("cell", "baseline", "stores:electrical-energy", base_ref))
        self.promote(self.claim("cell", "qualification", "supports:lamp", cold_ref,
            context="cold", asserted_value=False, origin="inferred"))
        self.promote(self.claim("lamp", "relationship", "requires:cell", relation_ref,
            relationship="requires", related_concept="cell"))
        retrieved = self.store.retrieve("cell", scope=self.scope, owner=self.owner,
                                        context="cold")
        self.assertEqual({row["kind"] for row in retrieved["claims"]},
                         {"baseline", "qualification"})
        self.assertEqual(len({row["fact_id"] for row in retrieved["claims"]}), 2)
        self.assertEqual({row["origin"] for row in retrieved["claims"]},
                         {"observed", "inferred"})
        answer = self.store.answer_support("cell", "lamp", "cold",
            scope=self.scope, owner=self.owner)
        self.assertEqual(answer["answer"], "no")
        self.assertEqual(answer["status"], "verified")

    def test_conflicting_context_claims_are_reported_and_force_abstention(self):
        cold_ref = self.evidence("The cell is unavailable in cold weather.")
        warm_ref = self.evidence("The cell remains available in cold weather.")
        relation_ref = self.evidence("The lamp requires the cell.")
        first = self.claim("cell", "qualification", "supports:lamp", cold_ref,
            context="cold", asserted_value=False)
        second = self.claim("cell", "qualification", "supports:lamp", warm_ref,
            context="cold", asserted_value=True)
        self.promote(first)
        self.promote(second)
        self.promote(self.claim("lamp", "relationship", "requires:cell", relation_ref,
            relationship="requires", related_concept="cell"))
        result = self.store.answer_support("cell", "lamp", "cold",
            scope=self.scope, owner=self.owner)
        self.assertEqual(result["status"], "conflict")
        self.assertIsNone(result["answer"])
        self.assertTrue(any(row["conflict"] for row in result["evidence"]))

    def test_unverified_candidate_does_not_enter_fact_retrieval(self):
        ref = self.evidence("A stable definition from a source.")
        candidate = self.claim("cell", "baseline", "stores:energy", ref,
            verification="candidate")
        ident = self.store.propose(candidate, scope=self.scope, owner=self.owner,
                                   policy=self.policy)
        self.assertTrue(ident)
        self.assertEqual(self.store.retrieve("cell", scope=self.scope,
                         owner=self.owner)["claims"], [])
        self.assertEqual(self.memory.candidates(scope=self.scope,
                         owner=self.owner)[0]["status"], "unverified")
        with self.assertRaisesRegex(ValueError, "independent reviewer"):
            self.store.promote(candidate, scope=self.scope, owner=self.owner,
                candidate_id=ident, reviewer=self.owner, method="human",
                start="2025-01-01", policy=self.policy)

    def test_versions_rollback_and_source_evidence_are_immutable(self):
        ref = self.evidence("A cell stores electrical energy.")
        claim = self.claim("cell", "baseline", "stores:electrical-energy", ref)
        first = self.promote(claim)
        same_key = self.claim("cell", "baseline", "stores:chemical-energy", ref)
        second = self.store.promote(same_key, scope=self.scope, owner=self.owner,
            candidate_id=self.store.propose(same_key, scope=self.scope, owner=self.owner,
                                            policy=self.policy),
            reviewer="independent-reviewer", method="human", start="2025-01-01",
            policy=self.policy, replace=True)
        rows = self.memory.facts(scope=self.scope, owner=self.owner,
                                 key="concept:cell:baseline")
        self.assertEqual([row["version"] for row in rows], [1, 2])
        self.assertTrue(rows[0]["superseded"])
        restored = self.store.promote(claim, scope=self.scope, owner=self.owner,
            candidate_id=self.store.propose(claim, scope=self.scope, owner=self.owner,
                                            policy=self.policy),
            reviewer="independent-reviewer", method="human", start="2025-01-01",
            policy=self.policy, rollback=first)
        latest = self.memory.facts(scope=self.scope, owner=self.owner,
                                   key="concept:cell:baseline")[-1]
        self.assertEqual(latest["id"], restored)
        self.assertEqual(latest["rollback"], first)
        self.assertEqual(json.loads(latest["claim"])["statement"], claim["statement"])
        self.assertFalse(self.store.retrieve("cell", scope=self.scope,
                                             owner=self.owner)["conflict"])
        with self.assertRaises(Exception):
            self.memory.sql.execute("DELETE FROM events WHERE id=?", (ref,))
        with self.assertRaises(Exception):
            self.memory.sql.execute("UPDATE events SET source='changed' WHERE id=?", (ref,))


if __name__ == "__main__":
    unittest.main()
