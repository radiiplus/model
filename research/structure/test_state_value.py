from __future__ import annotations

import unittest

from research.structure import generator
from research.structure.state_value import (
    ACTION_ORDER, candidate_actions, encode_pair, episodes, goal_reward,
    state_vector, verify_transition,
)


class StateValue(unittest.TestCase):
    def test_all_action_types_have_exact_supervised_targets(self):
        world = generator.make_world("train", 0)
        rows = episodes("train")
        self.assertEqual(ACTION_ORDER, ("none", "add_relationship"))
        self.assertEqual({row["action_id"] for row in rows},
                         {"none", "add-e1", "add-e2", "add-e3", "add-e4", "add-e5"})
        for action_id, action in candidate_actions(world):
            outcome = generator.apply_action(world, action)
            features, target = encode_pair(world, action, outcome)
            self.assertTrue(features)
            self.assertTrue(verify_transition(world, action, target))
            if target:
                corrupted = list(target)
                corrupted[0] = 1.0 - corrupted[0]
                self.assertFalse(verify_transition(world, action, corrupted))
            self.assertGreaterEqual(goal_reward(world, action, outcome), -0.05)

    def test_verifier_rejects_invalid_action_without_trusting_prediction(self):
        world = generator.make_world("train", 0)
        prediction = state_vector(world, world["initial_state"])
        self.assertFalse(verify_transition(
            world, {"type": "remove_relationship", "relationship": "missing"}, prediction))


if __name__ == "__main__":
    unittest.main()
