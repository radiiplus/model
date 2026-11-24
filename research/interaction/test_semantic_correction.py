from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research.interaction.semantic_benchmark import (
    ANSWER_BANK, SPECS, generate, make_rows, semantic_label, write_dataset,
)


class SemanticBenchmark(unittest.TestCase):
    def test_world_families_and_surface_variants_stay_in_one_split(self):
        data = generate()
        assignments = {}
        for split, rows in data.items():
            for row in rows:
                prior = assignments.setdefault(row["world_id"], split)
                self.assertEqual(prior, split)
            self.assertEqual(len({row["surface"] for row in rows}), 3)
        self.assertEqual(len(assignments), sum(count for _, count in SPECS.values()))
        self.assertEqual({row["world"]["pattern"] for row in data["train"]},
                         {"direct_supply"})
        self.assertEqual({row["world"]["pattern"] for row in data["compositional_test"]},
                         {"two_hop_supply"})

    def test_counterfactual_action_never_reduces_available_supply(self):
        rows = make_rows("counterfactual_test")
        by_world_context = {}
        for row in rows:
            by_world_context.setdefault((row["world_id"], row["context"]), {})[row["action"]] = row
        for pair in by_world_context.values():
            self.assertGreaterEqual(pair["insulate"]["effective_capacity"],
                                    pair["none"]["effective_capacity"])
            if pair["none"]["answer_semantic"] == "yes":
                self.assertEqual(pair["insulate"]["answer_semantic"], "yes")

    def test_paraphrase_tolerant_semantic_scoring_is_distinct_from_exact_text(self):
        examples = [("No, the output falls short.", "no"),
                    ("It will not power the lamp in cold weather.", "no"),
                    ("It is not sufficient to run the lamp.", "no"),
                    ("The connection should work; supply meets demand.", "yes"),
                    ("Yes, it can operate when warm.", "yes"),
                    ("to to to", "unknown")]
        for text, expected in examples:
            self.assertEqual(semantic_label(text), expected)
        exact = "No. The output is below demand."
        paraphrase = "It will not power the lamp in cold weather."
        self.assertNotEqual(exact, paraphrase)
        self.assertEqual(semantic_label(exact), semantic_label(paraphrase))

    def test_answer_templates_are_split_specific(self):
        signatures = [text for split in ANSWER_BANK for text in ANSWER_BANK[split].values()]
        self.assertEqual(len(signatures), len(set(signatures)))

    def test_dataset_manifest_records_byte_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_dataset(Path(directory) / "bench")
            self.assertEqual(manifest["worlds"], 56)
            self.assertEqual(manifest["examples"], 384)
            self.assertEqual(len(manifest["manifest_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
