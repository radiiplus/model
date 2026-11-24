import unittest

from research.structure.sampler import CandidateSampler, MODES, build_plan


class CandidateSamplerTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"example_id": f"ex-{i}", "world_id": f"w-{i // 2}",
                      "query_type": ("influence", "invariant")[i % 2],
                      "surface_id": ("natural", "structured")[i % 2]}
                     for i in range(64)]
        self.costs = [1200] * len(self.rows)
        self.plan = build_plan(self.costs, self.rows, list(range(64)), 4, 8, 41)
        self.losses = {row["example_id"]: float(i) for i, row in enumerate(self.rows)}
        self.rho = {row["example_id"]: float(i % 3) for i, row in enumerate(self.rows)}

    def test_candidate_pools_are_disjoint_and_token_budget_is_fixed(self):
        all_indices = [index for pool in self.plan["pools"] for index in pool["indices"]]
        self.assertEqual(len(all_indices), len(set(all_indices)))
        self.assertEqual(self.plan["budget_tokens_per_update"], [1200] * 4)

    def test_all_modes_are_deterministic_and_share_candidate_pools(self):
        for mode in MODES:
            left = CandidateSampler(self.rows, self.plan, mode, 42, self.losses, self.rho)
            right = CandidateSampler(self.rows, self.plan, mode, 42, self.losses, self.rho)
            score_fn = lambda rows: [self.losses[row["example_id"]] for row in rows]
            self.assertEqual(left.take(1, score_fn if mode != "uniform" else None),
                             right.take(1, score_fn if mode != "uniform" else None))

    def test_interrupted_sampler_resume_replays_exact_next_choice(self):
        score_fn = lambda rows: [self.losses[row["example_id"]] for row in rows]
        full = CandidateSampler(self.rows, self.plan, "learning_progress", 42, self.losses)
        full.take(1, score_fn)
        expected = full.take(1, score_fn)
        interrupted = CandidateSampler(self.rows, self.plan, "learning_progress", 42, self.losses)
        interrupted.take(1, score_fn)
        resumed = CandidateSampler(self.rows, self.plan, "learning_progress", 42,
                                   self.losses, state=interrupted.state())
        self.assertEqual(expected, resumed.take(1, score_fn))

    def test_coverage_rule_prefers_underrepresented_query_surface_and_world(self):
        rows = [
            {"example_id": "covered", "world_id": "w0", "query_type": "influence",
             "surface_id": "natural"},
            {"example_id": "new-group", "world_id": "w1", "query_type": "invariant",
             "surface_id": "structured"},
            {"example_id": "same-group-new-world", "world_id": "w2",
             "query_type": "influence", "surface_id": "natural"},
        ]
        plan = {"schema": 1, "seed": 8, "updates": 1, "pool_size": 3,
                "pools": [{"token_cost": 10, "indices": [0, 1, 2]}],
                "budget_tokens_per_update": [10]}
        sampler = CandidateSampler(rows, plan, "uniform", 9)
        sampler.group_counts["influence|natural"] = 1
        selected, _ = sampler.take(1)
        self.assertEqual(selected[0], 1)
        world_plan = {"schema": 1, "seed": 9, "updates": 1, "pool_size": 2,
                      "pools": [{"token_cost": 10, "indices": [0, 2]}],
                      "budget_tokens_per_update": [10]}
        world_sampler = CandidateSampler(rows, world_plan, "uniform", 10)
        world_sampler.world_counts["w0"] = 1
        world_selected, _ = world_sampler.take(1)
        self.assertEqual(world_selected, [2])

    def test_difficulty_avoids_extreme_top_loss(self):
        rows = [{"example_id": f"d-{i}", "world_id": "one-world",
                 "query_type": "influence", "surface_id": "natural"} for i in range(8)]
        plan = {"schema": 1, "seed": 1, "updates": 1, "pool_size": 8,
                "pools": [{"token_cost": 10, "indices": list(range(8))}],
                "budget_tokens_per_update": [10]}
        sampler = CandidateSampler(rows, plan, "difficulty", 3)
        _, meta = sampler.take(1, lambda values: list(range(1, 9)))
        self.assertLess(meta["selected_score"], 8)
        self.assertGreaterEqual(meta["selected_score"], 5)

    def test_learning_progress_and_rho_use_reducible_scores(self):
        rows = [{"example_id": f"s-{i}", "world_id": f"w-{i}",
                 "query_type": "influence", "surface_id": "natural"} for i in range(4)]
        plan = {"schema": 1, "seed": 2, "updates": 1, "pool_size": 4,
                "pools": [{"token_cost": 10, "indices": list(range(4))}],
                "budget_tokens_per_update": [10]}
        current = [3.0, 2.0, 8.0, 1.0]
        initial = {f"s-{i}": value for i, value in enumerate([4.0, 8.0, 9.0, 1.5])}
        progress = CandidateSampler(rows, plan, "learning_progress", 9, initial)
        _, progress_meta = progress.take(1, lambda _: current)
        self.assertEqual(progress_meta["selected_id"], "s-1")
        reference = {f"s-{i}": value for i, value in enumerate([1.0, 1.0, 1.0, 1.0])}
        rho = CandidateSampler(rows, plan, "rho_loss", 9, initial, reference)
        _, rho_meta = rho.take(1, lambda _: current)
        self.assertEqual(rho_meta["selected_id"], "s-2")


if __name__ == "__main__":
    unittest.main()
