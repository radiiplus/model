"""Deterministic value-based candidate selection for structural training."""

from __future__ import annotations

import random
import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Callable


MODES = ("uniform", "difficulty", "learning_progress", "rho_loss")


def build_plan(costs: list[int], rows: list[dict[str, Any]], allowed: list[int],
               updates: int, pool_size: int, seed: int) -> dict[str, Any]:
    if updates < 1 or pool_size < 2 or not allowed:
        raise ValueError("plan requires updates, a candidate pool >=2, and allowed rows")
    if len(costs) != len(rows) or any(not 0 <= index < len(rows) for index in allowed):
        raise ValueError("candidate costs or training row indices are invalid")
    rng = random.Random(seed)
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in allowed:
        grouped[int(costs[index])].append(index)
    remaining = {}
    for cost, indices in grouped.items():
        rng.shuffle(indices)
        remaining[cost] = list(indices)
    pools = []
    for _ in range(updates):
        eligible = [cost for cost, indices in remaining.items() if len(indices) >= pool_size]
        if not eligible:
            raise ValueError("not enough unique, equal-token candidate rows for the declared budget")
        weights = [len(remaining[cost]) for cost in eligible]
        cost = rng.choices(eligible, weights=weights, k=1)[0]
        indices = [remaining[cost].pop() for _ in range(pool_size)]
        pools.append({"token_cost": cost, "indices": indices})
    return {"schema": 1, "seed": seed, "updates": updates, "pool_size": pool_size,
            "pools": pools, "budget_tokens_per_update": [item["token_cost"] for item in pools],
            "allowed_indices_sha256": hashlib.sha256(json.dumps(allowed).encode()).hexdigest()}


class CandidateSampler:
    def __init__(self, rows: list[dict[str, Any]], plan: dict[str, Any], mode: str,
                 seed: int, initial_losses: dict[str, float] | None = None,
                 reference_losses: dict[str, float] | None = None,
                 state: dict[str, Any] | None = None):
        if mode not in MODES:
            raise ValueError(f"unknown sampler mode: {mode}")
        self.rows = rows
        self.plan = plan
        self.mode = mode
        self.initial_losses = initial_losses or {}
        self.reference_losses = reference_losses or {}
        self.rng = random.Random(seed)
        self.cursor = 0
        self.group_counts: Counter = Counter()
        self.world_counts: Counter = Counter()
        if state is not None:
            if state.get("mode") != mode or state.get("plan_seed") != plan["seed"]:
                raise ValueError("sampler state does not match mode or candidate plan")
            self.cursor = int(state["cursor"])
            self.group_counts.update(state["group_counts"])
            self.world_counts.update(state["world_counts"])
            self.rng.setstate(state["rng"])
            if not 0 <= self.cursor <= len(plan["pools"]):
                raise ValueError("sampler cursor is outside the candidate plan")

    def take(self, count: int, score_fn: Callable[[list[dict[str, Any]]], list[float]] | None = None
             ) -> tuple[list[int], dict[str, Any]]:
        if count != 1:
            raise ValueError("value samplers require batch size one for matched candidate scoring")
        if self.cursor >= len(self.plan["pools"]):
            raise StopIteration("candidate plan is exhausted")
        batch = self.plan["pools"][self.cursor]
        indices = list(batch["indices"])
        candidates = [self.rows[index] for index in indices]
        if len({int(self.plan["pools"][self.cursor]["token_cost"])
                for _ in candidates}) != 1:
            raise AssertionError("candidate pool has inconsistent token costs")
        scores = None
        if self.mode != "uniform":
            if score_fn is None:
                raise ValueError("scored sampling requires an active-loss callback")
            scores = score_fn(candidates)
            if len(scores) != len(candidates):
                raise ValueError("score callback returned the wrong number of values")
        eligible = self._coverage_candidates(candidates)
        local = [candidates.index(row) for row in eligible]
        selection_scores = None
        if self.mode == "uniform":
            selected = self.rng.choice(local)
        elif self.mode == "difficulty":
            ordered = sorted(local, key=lambda i: (scores[i], i))
            lo = int(len(ordered) * 0.50)
            hi = max(lo + 1, int(len(ordered) * 0.90))
            band = ordered[lo:hi]
            selected = self.rng.choice(band)
            selection_scores = scores
        elif self.mode == "learning_progress":
            progress = [float(self.initial_losses.get(candidates[i]["example_id"], scores[i]) - scores[i])
                        for i in range(len(candidates))]
            best = max(local, key=lambda i: (progress[i], -i))
            selected = best if progress[best] > 0 else self.rng.choice(local)
            selection_scores = progress
        else:
            if not self.reference_losses:
                raise ValueError("RHO-style selection requires training-only reference losses")
            reducible = [float(scores[i] - self.reference_losses[candidates[i]["example_id"]])
                         for i in range(len(candidates))]
            selected = max(local, key=lambda i: (reducible[i], -i))
            selection_scores = reducible
        row = candidates[selected]
        group = f"{row['query_type']}|{row['surface_id']}"
        self.group_counts[group] += 1
        self.world_counts[row["world_id"]] += 1
        self.cursor += 1
        return [indices[selected]], {"mode": self.mode, "candidate_indices": indices,
            "candidate_ids": [item["example_id"] for item in candidates],
            "selected_id": row["example_id"], "token_cost": batch["token_cost"],
            "selected_score": None if selection_scores is None else selection_scores[selected],
            "coverage_group": group}

    def _coverage_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        group_counts = [self.group_counts[f"{row['query_type']}|{row['surface_id']}"]
                        for row in candidates]
        min_group = min(group_counts)
        group_limited = [row for row, amount in zip(candidates, group_counts) if amount == min_group]
        world_counts = [self.world_counts[row["world_id"]] for row in group_limited]
        min_world = min(world_counts)
        return [row for row, amount in zip(group_limited, world_counts) if amount == min_world]

    def state(self) -> dict[str, Any]:
        return {"mode": self.mode, "plan_seed": self.plan["seed"], "cursor": self.cursor,
                "group_counts": dict(self.group_counts), "world_counts": dict(self.world_counts),
                "rng": self.rng.getstate()}
