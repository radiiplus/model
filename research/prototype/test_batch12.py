from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from research.experiments.retrieval_controller import DOCS, QUERIES, response_correct
from research.prototype.data import load_splits
from research.prototype.interfaces import ByteTokenizer
from research.prototype.model import ByteLanguageModel
from research.prototype.system import OfflineEvidenceSystem, load_fixture
from research.prototype.trainer import available, configure_torch, load_config, parse, snapshot, train_to


HERE = Path(__file__).resolve().parent
TEXTS = HERE.parent / "experiments/fixtures/tokenizer_corpus.jsonl"
SPLITS = HERE / "splits.json"
CONFIG = HERE / "byte_config.json"


class Batch12Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_torch()
        cls.tokenizer = ByteTokenizer()
        cls.encoded, cls.audit = load_splits(TEXTS, SPLITS, cls.tokenizer)
        cls.config = load_config(CONFIG)

    def test_byte_tokenizer_roundtrip_and_invalid_id(self):
        source = "東京 café 🧪 — bytes"
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(source)), source)
        with self.assertRaises(ValueError):
            self.tokenizer.decode([256])

    def test_splits_are_disjoint_and_partition_fixture(self):
        self.assertEqual(self.audit["document_counts"], {"train": 11, "validation": 3, "test": 4})
        self.assertEqual(self.audit["cross_split_exact_or_normalized_duplicate_count"], 0)
        ids = [x for group in self.audit["split_ids"].values() for x in group]
        self.assertEqual(len(ids), len(set(ids)))

    def test_cross_split_duplicate_text_is_rejected(self):
        rows = [json.loads(line) for line in TEXTS.read_text(encoding="utf-8").splitlines()]
        rows[-1]["text"] = rows[0]["text"]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "duplicate.jsonl"
            path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cross-split duplicate"):
                load_splits(path, SPLITS, self.tokenizer)

    def test_training_sampler_is_seeded(self):
        from research.prototype.data import ByteWindowLoader
        rows = [[0, 1, 2, 3, 4], [8, 9, 10], [20, 21], [30, 31, 32, 33]]
        loader = ByteWindowLoader(rows, 3)
        self.assertEqual(len(loader.windows), 3)
        self.assertEqual([loader.window(index).tolist() for index in loader.windows],
                         [[0, 1, 2, 3], [1, 2, 3, 4], [30, 31, 32, 33]])
        self.assertFalse(any(isinstance(item, torch.Tensor) for item in loader.windows))
        with self.assertRaises(IndexError):
            loader.window(3)
        with self.assertRaisesRegex(ValueError, "no complete context windows"):
            ByteWindowLoader([[1, 2], [3, 4, 5]], 3)
        loader = ByteWindowLoader(self.encoded["train"], self.config["context_length"])
        a = loader.sample(2, torch.Generator().manual_seed(8))
        b = loader.sample(2, torch.Generator().manual_seed(8))
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(parse("1000", "400"), (1000, 400, 600))
        self.assertEqual(parse("max", "400"), (None, 400, None))
        self.assertEqual(parse("9223372036854771712", "400"), (None, 400, None))
        self.assertEqual(parse("max", None), (None, None, None))
        self.assertEqual(available(900, [{"available": 250}]), 250)
        state = snapshot()
        self.assertEqual(state["device"], "cpu")
        self.assertGreater(state["available"], 0)
        self.assertTrue(all(row["kind"] in {"v1", "v2"} for row in state["groups"]))
        with self.assertRaises(ValueError):
            parse("1000", None)
        with self.assertRaises(RuntimeError):
            available(None, [])

    def test_interruption_resume_matches_uninterrupted_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            full, partial = root / "full.pt", root / "partial.pt"
            with patch("research.prototype.trainer.available_mib", return_value=1):
                with self.assertRaisesRegex(MemoryError, "effective available memory"):
                    train_to(self.config, self.encoded["train"], self.encoded["validation"],
                             seed=79, total_steps=1, checkpoint_path=root / "refused.pt")
            train_to({**self.config, "seed": 79}, self.encoded["train"], self.encoded["validation"],
                     seed=79, total_steps=8, checkpoint_path=full)
            stopped = train_to({**self.config, "seed": 79}, self.encoded["train"], self.encoded["validation"],
                               seed=79, total_steps=8, checkpoint_path=partial, stop_after_step=4)
            self.assertTrue(stopped["interrupted"])
            self.assertEqual(stopped["device"], "cpu")
            resumed = train_to({**self.config, "seed": 79}, self.encoded["train"], self.encoded["validation"],
                               seed=79, total_steps=8, checkpoint_path=partial, resume_from=partial)
            full_payload = torch.load(full, map_location="cpu", weights_only=True)
            resume_payload = torch.load(partial, map_location="cpu", weights_only=True)
            self.assertEqual(resumed["steps_completed"], 8)
            for name in full_payload["model"]:
                self.assertTrue(torch.equal(full_payload["model"][name], resume_payload["model"][name]), name)

    def test_deterministic_inference(self):
        model = ByteLanguageModel(self.config, seed=18)
        prefix = self.tokenizer.encode("A ")
        first = model.generate(prefix, 5)
        self.assertEqual(first, model.generate(prefix, 5))
        self.assertTrue(all(0 <= token < self.tokenizer.vocab_size for token in first))

    def test_system_ablation_supports_abstains_and_respects_budgets(self):
        docs, queries = load_fixture(DOCS, QUERIES)
        system = OfflineEvidenceSystem(docs)
        expected = {"none": 3, "full_context": 9, "bm25_unverified": None,
                    "bm25_top3": 9, "bounded_bm25": 9}
        for mode, correct in expected.items():
            rows = [system.answer(q, mode) for q in queries]
            if correct is not None:
                self.assertEqual(sum(response_correct(q, out, ctx)
                                     for q, (out, ctx, _) in zip(queries, rows)), correct)
            for out, _, meta in rows:
                self.assertIn("abstained", out)
                self.assertLessEqual(meta["retrieval_calls"], 2)
                self.assertLessEqual(meta["verifier_calls"], 2)
                self.assertLessEqual(meta["context_tokens"], 105 if mode == "full_context" else 64)
                if mode == "bounded_bm25":
                    self.assertLessEqual(meta["output_tokens"], 16)
                    self.assertLessEqual(meta["elapsed_seconds"], 0.05)

    def test_conflict_and_no_answer_abstain(self):
        docs, queries = load_fixture(DOCS, QUERIES)
        system = OfflineEvidenceSystem(docs)
        conflict = system.answer(queries[5], "bounded_bm25")[0]
        missing = system.answer(queries[6], "bounded_bm25")[0]
        self.assertTrue(conflict["abstained"])
        self.assertEqual(conflict["reason"], "conflicting_current_sources")
        self.assertTrue(missing["abstained"])

    def test_unverified_retrieval_is_not_treated_as_supported(self):
        docs, queries = load_fixture(DOCS, QUERIES)
        system = OfflineEvidenceSystem(docs)
        rows = [system.answer(query, "bm25_unverified") for query in queries]
        unsupported = sum(not response_correct(query, output, evidence)
                          for query, (output, evidence, _) in zip(queries, rows)
                          if not output["abstained"])
        self.assertGreater(unsupported, 0)
        self.assertTrue(all(meta["verifier_calls"] == 0 for _, _, meta in rows))


if __name__ == "__main__":
    unittest.main()
