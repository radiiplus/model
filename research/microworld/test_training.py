from __future__ import annotations

import unittest

import torch

from research.microworld.train import expected_signature, nested_equal, parse_answer


class SemanticGenerationParserTests(unittest.TestCase):
    def setUp(self):
        self.world = {"entities": [
            {"id": "agent-0", "name": "Mira", "type": "agent"},
            {"id": "object-0", "name": "key", "type": "object"},
            {"id": "location-0", "name": "hall", "type": "location"},
        ]}

    def parse(self, query_type: str, answer: str):
        row = {"query_type": query_type, "ground_truth": {}}
        return parse_answer(answer, row, self.world)

    def test_entities_and_relationship_bindings(self):
        entities = self.parse("entities", "Oracle result: Entities: e0:agent, e1:object.")
        self.assertEqual(entities["signature"], (("agent-0", "agent"),
                                                   ("object-0", "object")))
        edges = self.parse("relationships", "Relations: owns(e0,e1).")
        self.assertEqual(edges["signature"], (("owns", "agent-0", "object-0"),))

    def test_properties_location_action_and_constraints(self):
        props = self.parse("properties", 'key: color is "red"; portable is true.')
        self.assertEqual(props["signature"],
                         ("object-0", (("color", "red"), ("portable", True))))
        location = self.parse("location", "key is inside hall.")
        self.assertEqual(location["signature"], ("object-0", "inside", "location-0"))
        action = self.parse("action_validity", "The action is invalid. Code: target_closed.")
        self.assertEqual(action["signature"], (False, "target_closed"))
        constraints = self.parse("constraints", "Violated constraints are capacity, locked.")
        self.assertEqual(constraints["signature"], ("capacity", "locked"))

    def test_transition_and_counterfactual(self):
        transition_row = {"query_type": "transition", "ground_truth": {
            "before": {"positions": {"object-0": {"relation": "at", "target": "location-0"}},
                       "open_containers": set(), "owners": {}},
            "after": {"positions": {"object-0": {"relation": "inside", "target": "location-0"}},
                      "open_containers": set(), "owners": {}},
        }}
        transition = parse_answer(
            "key changes from at hall to inside hall.", transition_row, self.world)
        self.assertEqual(transition["signature"], ((("object-0", {
            "relation": "inside", "target": "location-0"}),), (), ()))
        counterfactual_row = {"query_type": "counterfactual", "ground_truth": {
            "changed_entities": ["object-0"],
            "primary_state": {"positions": {"object-0": {
                "relation": "inside", "target": "location-0"}}},
            "alternate_state": {"positions": {"object-0": {
                "relation": "at", "target": "location-0"}}},
        }}
        parsed = parse_answer(
            "key is inside hall after the listed action and at hall under the alternative.",
            counterfactual_row, self.world)
        self.assertEqual(parsed["signature"], expected_signature(counterfactual_row, self.world))

    def test_invalid_entity_reference_is_not_parseable(self):
        result = self.parse("relationships", "owns(e0,e9).")
        self.assertTrue(result["parseable"])
        self.assertEqual(result["invalid_entity_references"], 1)


class ExactStateComparisonTests(unittest.TestCase):
    def test_nested_optimizer_state_comparison_handles_tensors(self):
        left = {"state": {"step": torch.tensor(2), "momentum": [torch.tensor([1.0])]}}
        right = {"state": {"step": torch.tensor(2), "momentum": [torch.tensor([1.0])]}}
        changed = {"state": {"step": torch.tensor(2), "momentum": [torch.tensor([2.0])]}}
        self.assertTrue(nested_equal(left, right))
        self.assertFalse(nested_equal(left, changed))


if __name__ == "__main__":
    unittest.main()
