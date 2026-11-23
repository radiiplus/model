import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval_controller import (LexicalIndex, VersionedMemory, answer_mode,
                                  bounded_controller, canonical_id, retrieval_metrics,
                                  verify_context)


HERE = Path(__file__).resolve().parent
DOCS = json.loads((HERE / "fixtures/tiny_knowledge_base.json").read_text())
QUERIES = {q["id"]: q for q in json.loads((HERE / "fixtures/tiny_retrieval_queries.json").read_text())}


class RetrievalControllerTests(unittest.TestCase):
    def test_bm25_and_overlap_find_rare_fact(self):
        index = LexicalIndex(DOCS)
        query = QUERIES["q03"]["text"]
        self.assertEqual(index.search(query, "bm25", 1)[0]["id"], "d04")
        self.assertEqual(index.search(query, "overlap", 1)[0]["id"], "d04")

    def test_precision_at_k_uses_fixed_k_when_fewer_docs_return(self):
        index = LexicalIndex(DOCS)
        result = retrieval_metrics([QUERIES["q06"]], index, "bm25", 3)
        self.assertAlmostEqual(result["precision_at_k"], 2 / 3)

    def test_stale_fact_is_filtered_and_conflict_abstains(self):
        old = verify_context(QUERIES["q05"], DOCS)
        current = verify_context(QUERIES["q04"], DOCS)
        conflict = verify_context(QUERIES["q06"], DOCS)
        self.assertEqual(old["answer"], "08:00")
        self.assertEqual(current["answer"], "09:00")
        self.assertTrue(conflict["abstained"])
        self.assertEqual(conflict["reason"], "conflicting_current_sources")

    def test_retrieval_is_not_itself_support(self):
        query = QUERIES["q04"]
        unsupported = {**DOCS[5], "value": "08:00", "valid_to": None, "valid_from": "2025-01-01"}
        response = {"answer": "09:00", "abstained": False, "citations": ["d06"]}
        from retrieval_controller import response_correct
        self.assertFalse(response_correct(query, response, [unsupported]))

    def test_controller_obeys_cycle_and_context_budgets(self):
        for query in QUERIES.values():
            response, context, metadata = bounded_controller(query, VersionedMemory(DOCS))
            self.assertLessEqual(metadata["cycles"], 2)
            self.assertLessEqual(metadata["verifier_calls"], 2)
            self.assertLessEqual(metadata["context_tokens"], 64)
            self.assertLessEqual(metadata["output_tokens"], 16)
            self.assertLessEqual(metadata["elapsed_seconds"], 0.05)
            if query["id"] == "q06":
                self.assertTrue(response["abstained"])

    def test_full_context_and_no_context_differ(self):
        query = QUERIES["q03"]
        full, _, _ = answer_mode(query, DOCS, "full_context")
        none, _, _ = answer_mode(query, DOCS, "none")
        self.assertEqual(full["answer"], "NQ-17")
        self.assertTrue(none["abstained"])

    def test_memory_writes_require_provenance_and_invalidate_cache(self):
        memory = VersionedMemory(DOCS)
        before = memory.search("Veloria capital Qelth", k=1)
        self.assertEqual(before[0]["id"], "d01")
        bad = {"id": "new", "text": "Veloria population 100", "entity": "Veloria", "attribute": "population", "value": "100"}
        with self.assertRaises(ValueError):
            memory.add(bad)
        changed = dict(DOCS[0], text="Xylar has its capital in Qelth.", entity="Xylar")
        old_version = memory.version
        memory.update("d01", changed)
        self.assertGreater(memory.version, old_version)
        updated = {row["id"]: row for row in memory.search("Veloria capital Qelth", k=5)}
        self.assertEqual(updated["d01"]["entity"], "Xylar")
        memory.delete("d02")
        self.assertNotIn("d02", [row["id"] for row in memory.search("Veloria capital Qelth", k=5)])

    def test_duplicate_provenance_resolves_to_canonical_source(self):
        docs = {doc["id"]: doc for doc in DOCS}
        self.assertEqual(canonical_id("d02", docs), "d01")
        self.assertEqual(canonical_id("d02", {"d02": docs["d02"]}), "d02")


if __name__ == "__main__":
    unittest.main()
