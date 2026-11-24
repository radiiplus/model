from __future__ import annotations

import copy
import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from research.foundation import trainer as foundation_trainer
from research.foundation.model import Foundation
from research.structure import generator, schema, splits, train


def same(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            same(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(
            same(one, two) for one, two in zip(left, right))
    return left == right


class WorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, cls.audit = generator.generate_records()

    def test_deterministic_world_and_all_relation_types(self):
        first = generator.make_world("train", 4)
        second = generator.make_world("train", 4)
        self.assertEqual(first, second)
        self.assertEqual({row["type"] for row in first["relationships"]},
                         set(schema.RELATION_TYPES))
        self.assertEqual(len(first["entities"]), 6)

    def test_action_state_transitions_are_exact_and_nonmutating(self):
        world = generator.make_world("train", 3)
        before = copy.deepcopy(world["initial_state"])
        actions = generator.actions_for(world)
        self.assertEqual([name for name, _ in actions],
                         ["none", "remove", "add", "ownership", "property"])
        for name, action in actions:
            result = generator.apply_action(world, action)
            self.assertEqual(world["initial_state"], before)
            if name == "remove":
                self.assertLess(len(result["active_relationships"]), len(before["active_relationships"]))
            elif name == "add":
                self.assertGreater(len(result["active_relationships"]), len(before["active_relationships"]))
                self.assertTrue(result["added_relationships"])
            elif name == "ownership":
                self.assertNotIn("r0", result["active_relationships"])
                self.assertIn("x0", result["added_relationships"])
            elif name == "property":
                self.assertEqual(result["property_overrides"]["e0"]["status"], "restricted")

    def test_multistep_influence_and_held_out_patterns(self):
        comp = generator.make_world("compositional_test", 0)
        answer = generator.query_values(comp, None, generator.apply_action(comp, None))["influence"]
        names = {row["id"]: row["name"] for row in comp["entities"]}
        self.assertEqual(answer["target"], "e2")
        self.assertEqual(answer["entities"], [names["e0"], names["e1"]])
        self.assertNotIn("mixed_chain", splits.SPLIT_SPECS["train"]["patterns"])
        self.assertNotIn("diamond", splits.SPLIT_SPECS["train"]["patterns"])
        self.assertNotIn("gated_cycle", splits.SPLIT_SPECS["train"]["patterns"])

    def test_language_surfaces_expose_all_structural_ground_truth(self):
        world = generator.make_world("pattern_test", 0)
        description, _ = generator._describe(
            world, None, "natural", "pattern_test", random.Random(0))
        for entity in world["entities"]:
            values = entity["properties"]
            self.assertIn(entity["name"], description)
            for value in values.values():
                self.assertIn(str(value), description)
        for rule in world["rules"]:
            self.assertIn(rule, description)
        for constraint in world["constraints"]:
            self.assertIn(constraint, description)
        self.assertIn("Initial:", description)
        cf_world = generator.make_world("counterfactual_test", 0)
        cf_description, _ = generator._describe(
            cf_world, None, "natural", "counterfactual_test", random.Random(0))
        relation = cf_world["relationships"][0]
        names = {row["id"]: row["name"] for row in cf_world["entities"]}
        self.assertIn(f"owns {names[relation['target']]}", cf_description)
        self.assertNotIn(f"{names[relation['target']]} owns {names[relation['source']]}",
                         cf_description)

    def test_every_surface_and_question_is_kept_with_its_world(self):
        self.assertEqual(self.audit["worlds"], 108)
        self.assertEqual(self.audit["examples"], 12096)
        self.assertEqual(set(self.audit["worlds_by_split"]), set(splits.SPLIT_SPECS))
        for split, rows in self.records.items():
            first_world = rows[0]["world_id"]
            variants = [row for row in rows if row["world_id"] == first_world]
            self.assertEqual({row["surface_id"] for row in variants},
                             {"structured", "natural", "paraphrased", "reordered",
                              "irrelevant", "counterfactual"})
            self.assertTrue(all(row["split"] == split for row in variants))
        template_sets = [set(self.audit["templates_by_split"][split])
                         for split in self.audit["templates_by_split"]]
        for left in range(len(template_sets)):
            for right in range(left + 1, len(template_sets)):
                self.assertFalse(template_sets[left] & template_sets[right])

    def test_longest_examples_fit_foundation_context(self):
        rows = [row for values in self.records.values() for row in values]
        candidates = sorted(rows, key=lambda row: len(row["input"])
                            + len(row["structure_target"]), reverse=True)[:12]
        token = train.Tokenizer.load(train.TOKENIZER)
        limit = foundation_trainer.load()["model"]["context_length"]
        for row in candidates:
            answer, structure, *_ = train.encode_pair(row, token, limit)
            self.assertLessEqual(max(len(answer), len(structure)), limit, row["example_id"])

    def test_split_validator_rejects_world_and_template_leakage(self):
        rows = [
            {"example_id": "one", "world_id": "same", "generator_seed": 11000,
             "split": "train", "action_id": "none", "surface_id": "natural",
             "query_type": "influence", "template_id": "train:input",
             "answer_template_id": "train:answer"},
            {"example_id": "two", "world_id": "same", "generator_seed": 21000,
             "split": "validation", "action_id": "none", "surface_id": "natural",
             "query_type": "influence", "template_id": "validation:input",
             "answer_template_id": "validation:answer"},
        ]
        with self.assertRaisesRegex(ValueError, "cross splits"):
            splits.validate_records(rows)

    def test_wording_templates_are_distinct_across_splits(self):
        banks = generator.SURFACE_BANKS
        for field in ("header", "structured_header", "question_prefix", "rule", "answer",
                      "fact", "paraphrase"):
            values = [banks[split][field][0] if isinstance(banks[split][field], list)
                      else banks[split][field] for split in splits.SPLIT_SPECS]
            self.assertEqual(len(values), len(set(values)), field)
        for query in schema.QUERY_TYPES:
            values = [banks[split]["query"][query] for split in splits.SPLIT_SPECS]
            self.assertEqual(len(values), len(set(values)), query)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, _ = generator.generate_records()
        cls.row = next(row for row in cls.rows["train"]
                       if row["surface_id"] == "natural" and row["query_type"] == "influence")

    def _fixture(self, root: Path) -> tuple[Path, Path, dict]:
        data = root / "dataset"
        data.mkdir()
        files = []
        for split in ("train", "validation", "iid_test", "compositional_test",
                      "pattern_test", "counterfactual_test"):
            path = data / f"{split}.jsonl"
            row = dict(self.row, split=split, example_id=f"{split}-example",
                       world_id=f"{split}-world")
            path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
            files.append({"split": split, "path": path.name, "examples": 1,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        (data / "manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")
        config = copy.deepcopy(foundation_trainer.load())
        config["model"].update({"context_length": 1024, "layers": 1,
                                "d_model": 16, "n_heads": 4, "d_ff": 32})
        config["restore_optimizer"] = False
        config["condition"] = "language"
        config["train"].update({"rate": 0.0003, "penalty": 0.1, "clip": 1.0,
                                "warm": 0, "decay": 4, "floor": 0.1,
                                "batch": 1, "steps": 2, "checkpoint_interval": 1,
                                "eval_examples": 1, "seed": 81})
        model = Foundation(config["model"])
        initial = root / "initial.pt"
        foundation_trainer.save(initial, {"config": config, "model": model.state_dict()})
        return data, initial, config

    def test_three_objectives_and_exact_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, initial, config = self._fixture(root)
            whole = root / "whole.pt"
            partial = root / "partial.pt"
            with patch.object(train.guard, "available_mib", return_value=4096), \
                    patch.object(train.guard, "snapshot", return_value={"available": 4096}), \
                    patch.object(train.guard, "swap_counters", return_value=None):
                full = train.run(config, "a", initial, whole, data)
                train.run(config, "a", initial, partial, data, stop=1)
                resumed = train.run(config, "a", initial, partial, data, resume=partial)
                b = train.run(dict(config, condition="structure"), "b", initial,
                              root / "b.pt", data)
                c = train.run(dict(config, condition="combined"), "c", initial,
                              root / "c.pt", data)
            one = torch.load(whole, map_location="cpu", weights_only=True)
            two = torch.load(partial, map_location="cpu", weights_only=True)
            self.assertEqual(full["starting_weights_sha256"], b["starting_weights_sha256"])
            self.assertEqual(full["starting_weights_sha256"], c["starting_weights_sha256"])
            self.assertGreater(full["parameters"], 0)
            self.assertEqual(full["parameters"], b["parameters"])
            self.assertEqual(full["auxiliary_parameters"], 0)
            self.assertEqual(resumed["updates"], 2)
            self.assertEqual(one["sampler"], two["sampler"])
            deterministic_history = lambda history: [
                {key: value for key, value in item.items() if key != "selection_seconds"}
                for item in history]
            self.assertEqual(deterministic_history(one["history"]),
                             deterministic_history(two["history"]))
            self.assertEqual(one["history"][0]["example_ids"],
                             two["history"][0]["example_ids"])
            self.assertTrue(same(one["model"], two["model"]))
            self.assertTrue(same(one["optimizer"], two["optimizer"]))
            self.assertTrue(same(one["scheduler"], two["scheduler"]))
            self.assertEqual(full["validation_after"], resumed["validation_after"])
            self.assertEqual(set(full["validation_after"]),
                             {"validation", "iid_test", "compositional_test",
                              "pattern_test", "counterfactual_test"})
            self.assertEqual(b["objective"], "structural_recovery")
            self.assertEqual(c["objective"], "combined_answer_and_structure")
            self.assertGreater(full["language_target_tokens_seen"], 0)
            self.assertGreater(full["structure_target_tokens_seen"], 0)
            self.assertIn("answer_semantic_accuracy",
                          full["validation_after"]["validation"])
            self.assertIn("answer_generation_samples",
                          full["validation_after"]["validation"])

    def test_resource_guard_refuses_low_memory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, initial, config = self._fixture(root)
            with patch.object(train.guard, "available_mib", return_value=1):
                with self.assertRaisesRegex(MemoryError, "2560 MiB"):
                    train.run(config, "a", initial, root / "blocked.pt", data)

    def test_value_sampler_checkpoint_resume_is_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, initial, config = self._fixture(root)
            train_path = data / "train.jsonl"
            rows = [dict(self.row, split="train", example_id=f"train-{index}",
                         world_id=f"train-world-{index}") for index in range(4)]
            train_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                                  encoding="utf-8")
            manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
            item = next(value for value in manifest["files"] if value["split"] == "train")
            item["examples"] = len(rows)
            item["sha256"] = hashlib.sha256(train_path.read_bytes()).hexdigest()
            (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            token = train.Tokenizer.load(train.TOKENIZER)
            answer_ids, structure_ids, *_ = train.encode_pair(
                rows[0], token, config["model"]["context_length"])
            token_cost = len(answer_ids) + len(structure_ids) - 2
            plan = {"schema": 1, "seed": 31, "updates": 2, "pool_size": 2,
                    "pools": [{"token_cost": token_cost, "indices": [0, 1]},
                              {"token_cost": token_cost, "indices": [2, 3]}],
                    "budget_tokens_per_update": [token_cost, token_cost]}
            plan_path = root / "plan.json"
            initial_loss_path = root / "initial.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            initial_loss_path.write_text(json.dumps({row["example_id"]: 0.0 for row in rows}),
                                         encoding="utf-8")
            config["condition"] = "structure"
            config["train"]["steps"] = 2
            config["sampling"] = {"mode": "learning_progress", "plan_path": str(plan_path),
                "initial_losses_path": str(initial_loss_path), "reference_losses_path": None}
            whole, partial = root / "whole.pt", root / "partial.pt"
            with patch.object(train.guard, "available_mib", return_value=4096), \
                    patch.object(train.guard, "snapshot", return_value={"available": 4096}), \
                    patch.object(train.guard, "swap_counters", return_value=None):
                full = train.run(config, "b", initial, whole, data)
                train.run(config, "b", initial, partial, data, stop=1)
                resumed = train.run(config, "b", initial, partial, data, resume=partial)
            one = torch.load(whole, map_location="cpu", weights_only=True)
            two = torch.load(partial, map_location="cpu", weights_only=True)
            deterministic_history = lambda history: [
                {key: value for key, value in item.items() if key != "selection_seconds"}
                for item in history]
            self.assertEqual(deterministic_history(one["history"]),
                             deterministic_history(two["history"]))
            self.assertTrue(same(one["model"], two["model"]))
            self.assertTrue(same(one["optimizer"], two["optimizer"]))
            self.assertTrue(same(one["scheduler"], two["scheduler"]))
            self.assertEqual(one["sampler"], two["sampler"])
            self.assertEqual(full["input_tokens_seen"], resumed["input_tokens_seen"])


if __name__ == "__main__":
    unittest.main()
