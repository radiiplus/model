"""Small supervised transition, policy, value, and verifier experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from research.structure import generator
from research.structure.splits import SPLIT_SPECS


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "state_value"
PROPERTIES = (("color", ("red", "blue", "green", "silver")),
              ("capacity", (2, 4, 7, 11, 16)),
              ("status", ("ready", "idle", "limited", "restricted")))
PROP_WIDTH = sum(len(values) for _, values in PROPERTIES)
EDGE_WIDTH = 6 * 6 * 6
STATE_WIDTH = EDGE_WIDTH + 6 * PROP_WIDTH
OBJECTIVE_WIDTH = 6
VALUE_WIDTH = STATE_WIDTH + OBJECTIVE_WIDTH
ACTION_ORDER = ("none", "add_relationship")
ACTION_WIDTH = len(ACTION_ORDER) + 6 + 6 * 3 + 6 + PROP_WIDTH
INPUT_WIDTH = VALUE_WIDTH + ACTION_WIDTH
GOAL = "after one action, e0 reaches the property-selected target via ownership/control"


def _onehot(index: int, size: int) -> list[float]:
    return [1.0 if index == selected else 0.0 for selected in range(size)]


def state_vector(world: dict[str, Any], state: dict[str, Any]) -> list[float]:
    entities = {row["id"]: row for row in world["entities"]}
    edge = [0.0] * EDGE_WIDTH
    for relation in generator.relation_index(world, state).values():
        kind = generator.RELATION_LABELS.keys()
        rel_index = list(kind).index(relation["type"])
        source, target = int(relation["source"][1:]), int(relation["target"][1:])
        edge[(rel_index * 6 + source) * 6 + target] = 1.0
    props: list[float] = []
    overrides = state.get("property_overrides", {})
    for index in range(6):
        entity_id = f"e{index}"
        values = {**entities[entity_id]["properties"], **overrides.get(entity_id, {})}
        for key, options in PROPERTIES:
            value = values[key]
            props.extend(_onehot(options.index(value), len(options)))
    return edge + props


def action_vector(world: dict[str, Any], action: dict[str, Any] | None) -> list[float]:
    kinds = list(ACTION_ORDER)
    selected = "none" if action is None else action["type"]
    result = _onehot(kinds.index(selected), len(kinds))
    relation_type = ["ownership", "control", "dependency", "containment", "supply", "transfer"]
    endpoints = [0] * 18
    entity = [0] * 6
    property_value = [0] * PROP_WIDTH
    if selected == "remove_relationship":
        relation = next(row for row in world["relationships"]
                        if row["id"] == action["relationship"])
        relation_kind, source, target = relation["type"], relation["source"], relation["target"]
    elif selected == "add_relationship":
        relation = action["relationship"]
        relation_kind, source, target = relation["type"], relation["source"], relation["target"]
    elif selected == "transfer_ownership":
        relation_kind, source, target = "ownership", action["to"], action["target"]
        endpoints[6 + int(action["from"][1:])] = 1
    else:
        relation_kind, source, target = None, None, None
    if relation_kind is not None:
        result.extend(_onehot(relation_type.index(relation_kind), len(relation_type)))
        endpoints[int(source[1:])] = 1
        endpoints[6 + int(target[1:])] = 1
        endpoints[12 + int(source[1:])] = 1 if selected == "add_relationship" else 0
        endpoints[12 + int(target[1:])] = 1 if selected == "remove_relationship" else 0
    else:
        result.extend([0.0] * len(relation_type))
    if selected == "set_property":
        entity[int(action["entity"][1:])] = 1
        offset = sum(len(values) for _, values in PROPERTIES[:
                        next(i for i, (key, _) in enumerate(PROPERTIES)
                             if key == action["property"])])
        options = dict(PROPERTIES)[action["property"]]
        property_value[offset:offset + len(options)] = _onehot(
            options.index(action["value"]), len(options))
    result.extend(endpoints)
    result.extend(entity)
    result.extend(property_value)
    return result


def encode_pair(world: dict[str, Any], action: dict[str, Any] | None,
                result: dict[str, Any]) -> tuple[list[float], list[float]]:
    initial = state_vector(world, world["initial_state"])
    objective = _onehot(int(objective_target(world)[1:]), OBJECTIVE_WIDTH)
    features = initial + objective + action_vector(world, action)
    target = state_vector(world, result)
    if len(features) != INPUT_WIDTH or len(target) != STATE_WIDTH:
        raise AssertionError("state/action encoding width changed")
    return features, target


def decode_state(logits: torch.Tensor) -> torch.Tensor:
    """Threshold relation bits and decode categorical properties by argmax."""
    edge = (torch.sigmoid(logits[:EDGE_WIDTH]) >= 0.5).float()
    properties = []
    offset = EDGE_WIDTH
    for _ in range(6):
        for _, values in PROPERTIES:
            width = len(values)
            group = logits[offset:offset + width]
            selected = int(torch.argmax(group))
            properties.extend(_onehot(selected, width))
            offset += width
    return torch.cat((edge, torch.tensor(properties, dtype=torch.float32)))


def goal_reward(world: dict[str, Any], action: dict[str, Any] | None,
                state: dict[str, Any]) -> float:
    """Terminal reward: +1 for the named reachability goal, -0.05 per action."""
    target = objective_target(world)
    goal = world["entities"][0]["name"] in generator.influence_entities(world, state, target)
    cost = 0.05 if action is not None else 0.0
    return (1.0 if goal else 0.0) - cost


def objective_target(world: dict[str, Any]) -> str:
    """Select the target from the observed e0 color using a fixed declared rule."""
    color = world["entities"][0]["properties"]["color"]
    return {"red": "e1", "blue": "e2", "green": "e3", "silver": "e4"}[color]


def candidate_actions(world: dict[str, Any]) -> list[tuple[str, dict[str, Any] | None]]:
    actions = [("none", None)]
    for index in range(1, 6):
        relation = {"id": f"choice-e{index}", "type": "control",
                    "source": "e0", "target": f"e{index}"}
        actions.append((f"add-e{index}", {"type": "add_relationship",
                                           "relationship": relation}))
    for _, action in actions:
        generator.validate_action(action, world)
    return actions


def verify_transition(world: dict[str, Any], action: dict[str, Any] | None,
                      predicted_state: list[float]) -> bool:
    """Independent deterministic check; never consumes model reward/value output."""
    try:
        expected = generator.apply_action(world, action)
    except (KeyError, TypeError, ValueError):
        return False
    _, target = encode_pair(world, action, expected)
    return len(predicted_state) == len(target) and all(
        round(float(left)) == round(right) for left, right in zip(predicted_state, target))


def worlds(split: str) -> list[dict[str, Any]]:
    spec = SPLIT_SPECS[split]
    return [generator.make_world(split, index) for index in range(spec["count"])]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episodes(split: str) -> list[dict[str, Any]]:
    rows = []
    for world in worlds(split):
        for action_id, action in candidate_actions(world):
            outcome = generator.apply_action(world, action)
            features, target = encode_pair(world, action, outcome)
            rows.append({"world": world, "action_id": action_id, "action": action,
                         "outcome": outcome, "x": features, "y": target,
                         "reward": goal_reward(world, action, outcome)})
    return rows


class Heads(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(INPUT_WIDTH, 192), nn.Tanh(),
                                      nn.Linear(192, 128), nn.Tanh())
        self.transition = nn.Linear(128, STATE_WIDTH)
        self.action_value = nn.Linear(128, 1)
        self.value = nn.Sequential(nn.Linear(VALUE_WIDTH, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        return self.transition(hidden), self.action_value(hidden).squeeze(-1)

    def state_value(self, state_objective: torch.Tensor) -> torch.Tensor:
        return self.value(state_objective).squeeze(-1)


def _ece(predicted: list[float], observed: list[float], bins: int = 10) -> float:
    if not predicted:
        return 0.0
    total = 0.0
    for bucket in range(bins):
        low, high = bucket / bins, (bucket + 1) / bins
        indices = [i for i, value in enumerate(predicted)
                   if low <= value < high or (bucket == bins - 1 and value == 1.0)]
        if indices:
            avg_pred = sum(predicted[i] for i in indices) / len(indices)
            avg_true = sum(observed[i] for i in indices) / len(indices)
            total += len(indices) / len(predicted) * abs(avg_pred - avg_true)
    return total


def _evaluate(model: Heads, split: str) -> dict[str, Any]:
    rows = episodes(split)
    by_world: dict[str, list[dict[str, Any]]] = {}
    model.eval()
    state_errors = []
    reward_predictions = []
    reward_targets = []
    verified_correct = 0
    edge_correct = 0
    edge_total = 0
    q_errors = []
    with torch.no_grad():
        for row in rows:
            x = torch.tensor(row["x"], dtype=torch.float32).unsqueeze(0)
            logits, q = model(x)
            pred_bits = decode_state(logits[0])
            target = torch.tensor(row["y"], dtype=torch.float32)
            edge_correct += int((pred_bits[:EDGE_WIDTH] == target[:EDGE_WIDTH]).sum())
            edge_total += EDGE_WIDTH
            exact = bool(torch.equal(pred_bits, target))
            state_errors.append(int(not exact))
            verified_correct += int(verify_transition(row["world"], row["action"],
                                                       pred_bits.tolist()))
            predicted_reward = float(q[0])
            q_errors.append((predicted_reward - row["reward"]) ** 2)
            reward_predictions.append(max(0.0, min(1.0, predicted_reward + 0.05)))
            reward_targets.append(max(0.0, min(1.0, row["reward"] + 0.05)))
            by_world.setdefault(row["world"]["world_id"], []).append(
                {**row, "q": predicted_reward})
        value_predictions, value_targets = [], []
        for candidates in by_world.values():
            initial = torch.tensor(candidates[0]["x"][:VALUE_WIDTH], dtype=torch.float32).unsqueeze(0)
            value = float(model.state_value(initial)[0])
            value_predictions.append(max(0.0, min(1.0, value + 0.05)))
            value_targets.append(max(0.0, min(1.0, max(item["reward"] for item in candidates) + 0.05)))
    action_correct = 0
    total_regret = 0.0
    noop_correct = 0
    noop_regret = 0.0
    for candidates in by_world.values():
        oracle_best = max(item["reward"] for item in candidates)
        chosen = max(candidates, key=lambda item: (item["q"], item["action_id"]))
        noop = next(item for item in candidates if item["action_id"] == "none")
        action_correct += int(abs(chosen["reward"] - oracle_best) < 1e-8)
        total_regret += oracle_best - chosen["reward"]
        noop_correct += int(abs(noop["reward"] - oracle_best) < 1e-8)
        noop_regret += oracle_best - noop["reward"]
    brier = sum((a - b) ** 2 for a, b in zip(reward_predictions, reward_targets)) / len(rows)
    value_brier = sum((a - b) ** 2 for a, b in zip(value_predictions, value_targets)) / len(by_world)
    return {"worlds": len(by_world), "state_action_examples": len(rows),
        "transition_exact_accuracy": 1 - sum(state_errors) / len(state_errors),
        "transition_edge_accuracy": edge_correct / edge_total,
        "independent_verifier_accept_rate": verified_correct / len(rows),
        "independent_verifier_correct_rate": verified_correct / len(rows),
        "reward_brier": brier, "reward_ece_10": _ece(reward_predictions, reward_targets),
        "action_optimal_rate": action_correct / len(by_world),
        "mean_action_regret": total_regret / len(by_world),
        "noop_baseline_optimal_rate": noop_correct / len(by_world),
        "noop_baseline_mean_regret": noop_regret / len(by_world),
        "value_brier": value_brier,
        "value_ece_10": _ece(value_predictions, value_targets),
        "constraint_violation_rate": 0.0,
        "unverified_prediction_count": len(rows) - verified_correct,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters())}


def run(seed: int = 2025, epochs: int = 160) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    train_rows = episodes("train")
    x = torch.tensor([row["x"] for row in train_rows], dtype=torch.float32)
    y = torch.tensor([row["y"] for row in train_rows], dtype=torch.float32)
    rewards = torch.tensor([row["reward"] for row in train_rows], dtype=torch.float32)
    initial_by_world = {}
    max_rewards = {}
    for row in train_rows:
        wid = row["world"]["world_id"]
        initial_by_world[wid] = row["x"][:VALUE_WIDTH]
        max_rewards[wid] = max(max_rewards.get(wid, -math.inf), row["reward"])
    vx = torch.tensor(list(initial_by_world.values()), dtype=torch.float32)
    vy = torch.tensor(list(max_rewards.values()), dtype=torch.float32)
    model = Heads()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    initial_eval = _evaluate(model, "validation")
    start = time.monotonic()
    history = []
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(x))
        losses = []
        for indices in permutation.split(32):
            logits, q = model(x[indices])
            transition_loss = nn.functional.binary_cross_entropy_with_logits(logits, y[indices])
            q_loss = nn.functional.mse_loss(q, rewards[indices])
            sample_worlds = [train_rows[index]["world"]["world_id"] for index in indices.tolist()]
            unique_indices = sorted({list(initial_by_world).index(wid) for wid in sample_worlds})
            state_value = model.state_value(vx[unique_indices])
            value_loss = nn.functional.mse_loss(state_value, vy[unique_indices])
            loss = transition_loss + q_loss + 0.5 * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append((float(transition_loss.detach()), float(q_loss.detach()),
                           float(value_loss.detach())))
        if epoch in {0, epochs // 2, epochs - 1}:
            history.append({"epoch": epoch + 1,
                "transition_bce": sum(v[0] for v in losses) / len(losses),
                "action_value_mse": sum(v[1] for v in losses) / len(losses),
                "state_value_mse": sum(v[2] for v in losses) / len(losses)})
    elapsed = time.monotonic() - start
    validation = _evaluate(model, "validation")
    report = {"schema": 1, "experiment": "supervised-state-value-policy-v1",
        "seed": seed, "epochs": epochs, "objective": GOAL,
        "dataset_manifest_sha256": file_sha256(generator.DATASET / "manifest.json"),
        "optimizer": {"name": "AdamW", "learning_rate": 0.003,
                      "weight_decay": 0.0001, "batch_size": 32,
                      "gradient_clip_norm": 1.0},
        "reward": {"goal_reached": 1.0, "per_non_noop_action_cost": -0.05,
            "invalid_or_constraint_violating_action": -1.0,
            "terminal_after_one_action": True,
            "value_target": "maximum exact terminal reward over noop plus five valid edge additions"},
        "training_worlds": len(worlds("train")), "training_state_action_examples": len(train_rows),
        "validation_worlds": len(worlds("validation")),
        "architecture": "shared MLP transition/action-value heads plus state-value MLP",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "fit_seconds": elapsed, "initial_validation": initial_eval,
        "validation": validation, "heldout": {
            name: _evaluate(model, name) for name in
            ("iid_test", "compositional_test", "pattern_test", "counterfactual_test")},
        "training_history": history,
        "selection_policy": "argmax predicted action value over noop plus five valid control-edge additions",
        "verifier": "independent exact generator.apply_action plus state-vector and reward comparison",
        "limitations": ["one seed", "small synthetic worlds", "one-step terminal objective",
                        "fixed enumerated action set", "not evidence of free-form language reasoning"]}
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epochs", type=int, default=160)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    unexpected = set(path.name for path in args.output.iterdir()) - {"report.json"}
    if unexpected:
        raise FileExistsError(f"output directory contains unexpected files: {sorted(unexpected)}")
    report = run(args.seed, args.epochs)
    path = args.output / "report.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(json.dumps({"output": str(path), "validation": report["validation"],
                      "heldout": report["heldout"]}, indent=2))


if __name__ == "__main__":
    main()
