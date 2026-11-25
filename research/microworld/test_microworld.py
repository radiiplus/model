from __future__ import annotations

import copy
import unittest
from collections import Counter

from research.microworld.baselines import evaluate
from research.microworld.generator import (
    build_world_examples, episodes_for, make_world,
)
from research.microworld.oracle import answer_query, execute, relationships, run_episode
from research.microworld.schema import constraint_violations, validate_world
from research.microworld.splits import (
    MAX_WORLDS, split_assignments, validate_world_and_example_splits,
)


class MicroworldTests(unittest.TestCase):
    def test_seeded_worlds_and_partition_sizes_are_deterministic(self):
        first = make_world(7, "train", seed=70000)
        self.assertEqual(first, make_world(7, "train", seed=70000))
        self.assertEqual(split_assignments(80), split_assignments(80))
        self.assertEqual(sum(Counter(split_assignments(80).values()).values()), 80)
        self.assertEqual(MAX_WORLDS, 200_000)
        self.assertEqual(len(split_assignments(MAX_WORLDS, seed=80)), MAX_WORLDS)
        validate_world(first)

    def test_actions_have_preconditions_effects_and_nonmutating_failures(self):
        world = make_world(1, "train", seed=71000)
        before = copy.deepcopy(world["initial_state"])
        result = run_episode(world, next(row["actions"] for row in episodes_for(world)
                                         if row["kind"] == "store"))
        self.assertTrue(all(row["valid"] for row in result["history"]))
        self.assertNotEqual(result["resulting_state"], before)
        self.assertEqual(world["initial_state"], before)
        invalid = next(row for row in episodes_for(world) if row["kind"] == "invalid")
        run = run_episode(world, invalid["actions"])
        self.assertTrue(run["history"][0]["valid"])
        self.assertFalse(run["history"][1]["valid"])
        self.assertEqual(run["history"][1]["reason"]["code"], "container_closed")
        self.assertEqual(run["history"][1]["state"], run["history"][0]["state"])
        self.assertEqual(constraint_violations(world, run["resulting_state"]), [])
        inaccessible = execute(world, before, {
            "type": "pick_up", "agent": "agent-0", "object": "object-1"})
        self.assertFalse(inaccessible["valid"])
        self.assertEqual(inaccessible["reason"]["code"], "object_inside_container")
        self.assertEqual(inaccessible["state"], before)

    def test_relationships_include_ownership_containment_and_location(self):
        world = make_world(3, "train", seed=72000)
        edges = relationships(world, world["initial_state"])
        kinds = {row["type"] for row in edges}
        self.assertTrue({"owns", "contains", "located_at", "connected_to"} <= kinds)
        flat_edges = {(row["type"], row["source"], row["target"])
                      for row in edges}
        self.assertIn(("contains", "container-0", "container-1"), flat_edges)
        self.assertNotIn(("contains", "container-1", "object-2"), flat_edges)
        nested = make_world(2, "compositional_test", seed=72000)
        edge_pairs = {(row["type"], row["source"], row["target"])
                      for row in relationships(nested, nested["initial_state"])}
        self.assertIn(("contains", "container-0", "container-1"), edge_pairs)
        self.assertIn(("contains", "container-1", "object-2"), edge_pairs)

    def test_counterfactual_changes_the_referenced_object_state(self):
        world = make_world(3, "counterfactual_test", seed=73000)
        episode = next(row for row in episodes_for(world) if row["kind"] == "counterfactual")
        primary = run_episode(world, episode["actions"])["resulting_state"]
        alternate = run_episode(world, episode["counterfactual_actions"])["resulting_state"]
        ident = episode["counterfactual_entity"]
        self.assertNotEqual(primary["positions"][ident], alternate["positions"][ident])
        rows = [row for row in build_world_examples(world)
                if row["query_type"] == "counterfactual"]
        self.assertTrue(rows)
        self.assertTrue(all(row["ground_truth"]["outcome_changed"] for row in rows))

    def test_oracle_labels_match_every_rendered_example(self):
        worlds = [make_world(4, "train", seed=74000),
                  make_world(5, "compositional_test", seed=74000),
                  make_world(6, "counterfactual_test", seed=74000),
                  make_world(7, "grammar_test", seed=74000)]
        rows = [row for world in worlds for row in build_world_examples(world)]
        world_index = {world["world_id"]: world for world in worlds}
        for row in rows:
            self.assertTrue(row["prompt"])
            self.assertTrue(row["answer_text"])
            self.assertFalse(row["answer_text"].startswith("{"))
            self.assertEqual(answer_query(world_index[row["world_id"]],
                                          row["oracle_state"], row["query"]),
                             row["ground_truth"])
        for world in worlds:
            world_rows = [row for row in rows if row["world_id"] == world["world_id"]]
            self.assertEqual(len(world_rows), len({row["prompt"] for row in world_rows}))
        train_constructions = {row["construction_id"] for row in rows
                               if row["split"] == "train"}
        grammar_constructions = {row["construction_id"] for row in rows
                                 if row["split"] == "grammar_test"}
        self.assertTrue(train_constructions)
        self.assertFalse(train_constructions & grammar_constructions)
        self.assertEqual({row["construction_id"] for row in rows
                          if row["split"] == "grammar_test"},
                         {"heldout:negated_question", "heldout:inverted_order"})

    def test_holdout_compositions_templates_and_worlds_do_not_leak(self):
        train_world = make_world(8, "train", seed=75000)
        comp_world = make_world(9, "compositional_test", seed=75000)
        grammar_world = make_world(10, "grammar_test", seed=75000)
        worlds = [train_world, comp_world, grammar_world]
        rows = [row for world in worlds for row in build_world_examples(world)]
        summary = validate_world_and_example_splits(worlds, rows)
        self.assertEqual(summary["worlds"], 3)
        self.assertEqual(summary["counterfactual_pairs_checked"], 0)
        self.assertEqual(train_world["composition"], "flat_placement")
        self.assertEqual(comp_world["composition"], "nested_container_chain")
        self.assertEqual(len(summary["heldout_grammar_templates"]), 14)
        self.assertEqual(len(summary["heldout_grammar_answer_templates"]), 7)
        contaminated = copy.deepcopy(comp_world)
        contaminated["world_id"] = "contaminated-train-world"
        contaminated["split"] = "train"
        with self.assertRaisesRegex(ValueError, "held-out composition"):
            validate_world_and_example_splits(
                [train_world, comp_world, contaminated, grammar_world], rows)
        leaked_answers = copy.deepcopy(rows)
        grammar_row = next(row for row in leaked_answers if row["split"] == "grammar_test")
        grammar_row["answer_template_id"] = "common_answer:entities"
        with self.assertRaisesRegex(ValueError, "answer template"):
            validate_world_and_example_splits(worlds, leaked_answers)
        conflicting = copy.deepcopy(rows)
        conflict = copy.deepcopy(conflicting[1])
        conflict["example_id"] += ":conflict"
        conflict["ground_truth"] = {"incorrect": True}
        conflicting.append(conflict)
        with self.assertRaisesRegex(ValueError, "conflicting ground-truth"):
            validate_world_and_example_splits(worlds, conflicting)

    def test_baselines_have_expected_roles_and_metrics(self):
        worlds = [make_world(11, "train", seed=76000),
                  make_world(12, "grammar_test", seed=76000)]
        rows = [row for world in worlds for row in build_world_examples(world)]
        report = evaluate(worlds, rows)
        heldout = report["partitions"]["grammar_test"]["methods"]
        self.assertEqual(heldout["direct_oracle"]["exact_structural_accuracy"], 1.0)
        self.assertEqual(heldout["template_system"]["exact_structural_accuracy"], 1.0)
        self.assertEqual(heldout["template_system"]["exact_answer_text_accuracy"], 1.0)
        self.assertEqual(heldout["exact_retrieval"]["covered"], 0)
        self.assertGreaterEqual(heldout["majority_answer"]["covered"], 0)


if __name__ == "__main__":
    unittest.main()
