import json
import unittest

from research.structure.evaluate import (PARTITIONS, build_memory_extractor,
    choose_rows, constraint_violations, load_rows, normalized, score_structure,
    symbolic_answer)
from research.adaptive import loop as strategy_loop


class EvaluationTests(unittest.TestCase):
    def test_world_sample_keeps_surface_variants_together(self):
        for split in PARTITIONS:
            rows = choose_rows(load_rows(split), 1)
            worlds = {row["world_id"] for row in rows}
            self.assertLessEqual(len(worlds), 2)
            self.assertTrue(all(row["split"] == split for row in rows))
            for world in worlds:
                self.assertEqual({row["surface_id"] for row in rows if row["world_id"] == world},
                                 {"structured", "natural", "paraphrased", "reordered",
                                  "irrelevant", "counterfactual"})

    def test_structured_rule_solver_reconstructs_answers_without_hidden_labels(self):
        for split in PARTITIONS:
            rows = [row for row in load_rows(split) if row["surface_id"] == "structured"]
            for row in rows:
                actual = symbolic_answer(row["input"])
                self.assertEqual(normalized(actual), normalized(row["answer_value"]), row["example_id"])

    def test_existing_strategy_extractor_is_train_only_exact_memory(self):
        candidate, episodes = build_memory_extractor(load_rows("train"))
        self.assertEqual(candidate["procedure"], "memorize")
        self.assertTrue(all(item["split"] == "train" for item in episodes))
        test_row = next(row for row in load_rows("iid_test") if row["surface_id"] == "structured")
        answer, actions = strategy_loop.run(candidate,
            {"family": "structural-world", "prompt": test_row["input"]}, episodes)
        self.assertEqual(answer, "")
        self.assertEqual(actions, [])

    def test_exact_structure_scoring_uses_parsed_json(self):
        row = next(row for row in load_rows("iid_test") if row["surface_id"] == "structured")
        metrics = score_structure(row["structure_target"], row)
        self.assertTrue(metrics["json_valid"])
        self.assertTrue(metrics["exact"])
        self.assertTrue(metrics["transition_exact"])
        self.assertFalse(metrics["constraint_violations"])

    def test_constraint_checker_flags_self_edge(self):
        row = next(row for row in load_rows("iid_test") if row["surface_id"] == "structured")
        parsed = json.loads(row["structure_target"])
        parsed["r"][0][3] = parsed["r"][0][2]
        self.assertIn("self-edge", constraint_violations(parsed))

    def test_malformed_json_and_valid_nonprotocol_json_fail_closed(self):
        row = next(row for row in load_rows("iid_test") if row["surface_id"] == "structured")
        self.assertFalse(score_structure("{", row)["json_valid"])
        malformed = score_structure("[]", row)
        self.assertTrue(malformed["json_valid"])
        self.assertIn("malformed-or-missing-fields", malformed["constraint_violations"])


if __name__ == "__main__":
    unittest.main()
