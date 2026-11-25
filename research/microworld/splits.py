"""Deterministic world-level partitions and leakage checks."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from typing import Any


SPLIT_RATIOS = {
    "train": 0.70,
    "validation": 0.10,
    "iid_test": 0.10,
    "compositional_test": 0.04,
    "counterfactual_test": 0.04,
    "grammar_test": 0.02,
}
MAX_WORLDS = 200_000


def split_assignments(world_count: int, seed: int = 20261125) -> dict[int, str]:
    if not 50 <= world_count <= MAX_WORLDS:
        raise ValueError(f"world count must be between 50 and {MAX_WORLDS}")
    exact = {name: world_count * ratio for name, ratio in SPLIT_RATIOS.items()}
    counts = {name: int(value) for name, value in exact.items()}
    remainder = world_count - sum(counts.values())
    for name in sorted(SPLIT_RATIOS, key=lambda key: exact[key] - counts[key], reverse=True)[:remainder]:
        counts[name] += 1
    if any(counts[name] < 1 for name in SPLIT_RATIOS):
        raise ValueError("world count must provide at least one world per partition")
    order = list(range(world_count))
    random.Random(seed).shuffle(order)
    result = {}
    cursor = 0
    for name, count in counts.items():
        for ident in order[cursor:cursor + count]:
            result[ident] = name
        cursor += count
    return result


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_world_and_example_splits(worlds: list[dict[str, Any]],
                                      examples: list[dict[str, Any]]) -> dict[str, Any]:
    world_splits: dict[str, str] = {}
    composition_by_split: dict[str, set[str]] = defaultdict(set)
    for world in worlds:
        ident, split = world["world_id"], world["split"]
        prior = world_splits.setdefault(ident, split)
        if prior != split:
            raise ValueError("world variants cross split boundaries")
        composition_by_split[split].add(world["composition"])
    if not world_splits:
        raise ValueError("no worlds supplied")

    if ("nested_container_chain" in composition_by_split["compositional_test"]
            and "nested_container_chain" in composition_by_split["train"]):
        raise ValueError("held-out composition occurs in training worlds")
    if ("nested_container_chain" in composition_by_split["compositional_test"]
            and not composition_by_split["train"]):
        raise ValueError("training composition inventory is empty")

    prompts: dict[str, tuple[str, str]] = {}
    templates: dict[str, set[str]] = defaultdict(set)
    answer_templates: dict[str, set[str]] = defaultdict(set)
    transition_splits: dict[str, str] = {}
    examples_by_split = defaultdict(int)
    counterfactuals = set()
    for row in examples:
        split = row["split"]
        if world_splits.get(row["world_id"]) != split:
            raise ValueError("example does not belong to its world's partition")
        prompt_key = " ".join(row["prompt"].casefold().split())
        target_hash = stable_hash(row["ground_truth"])
        previous = prompts.setdefault(prompt_key, (split, target_hash))
        if previous[0] != split:
            raise ValueError("duplicated prompt text crosses partitions")
        if previous[1] != target_hash:
            raise ValueError("one prompt maps to conflicting ground-truth targets")
        templates[split].add(row["template_id"])
        answer_templates[split].add(row["answer_template_id"])
        transition_id = row["transition_id"]
        previous_split = transition_splits.setdefault(transition_id, split)
        if previous_split != split:
            raise ValueError("underlying state transition crosses partitions")
        examples_by_split[split] += 1
        alternate = row.get("counterfactual")
        if split == "counterfactual_test" and alternate:
            if alternate["primary_state"] == alternate["alternate_state"]:
                raise ValueError("counterfactual does not change the resulting state")
            if alternate["primary_state"]["positions"][alternate["entity"]] == \
                    alternate["alternate_state"]["positions"][alternate["entity"]]:
                raise ValueError("counterfactual does not change the queried entity outcome")
            counterfactuals.add(row["episode_id"])

    grammar_templates = templates["grammar_test"]
    other_templates = set().union(*(values for split, values in templates.items()
                                    if split != "grammar_test"))
    if grammar_templates & other_templates:
        raise ValueError("held-out grammar/template construction leaked")
    grammar_answers = answer_templates["grammar_test"]
    other_answers = set().union(*(values for split, values in answer_templates.items()
                                  if split != "grammar_test"))
    if grammar_answers & other_answers:
        raise ValueError("held-out answer template construction leaked")
    return {
        "worlds": len(world_splits),
        "examples": len(examples),
        "worlds_by_split": {split: sum(value == split for value in world_splits.values())
                            for split in SPLIT_RATIOS},
        "examples_by_split": {split: examples_by_split[split] for split in SPLIT_RATIOS},
        "unique_prompts": len(prompts),
        "unique_transitions": len(transition_splits),
        "heldout_grammar_templates": sorted(grammar_templates),
        "heldout_grammar_answer_templates": sorted(grammar_answers),
        "counterfactual_pairs_checked": len(counterfactuals),
        "world_partition_hash": stable_hash(sorted(world_splits)),
    }
