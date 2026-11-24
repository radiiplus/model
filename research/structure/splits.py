"""Seed and structural-pattern partitions for the synthetic world benchmark."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


SPLIT_SPECS = {
    "train": {"seed_start": 11000, "count": 48, "patterns": ["staggered"]},
    "validation": {"seed_start": 21000, "count": 12, "patterns": ["staggered"]},
    "iid_test": {"seed_start": 31000, "count": 12, "patterns": ["staggered"]},
    "compositional_test": {"seed_start": 41000, "count": 12, "patterns": ["mixed_chain"]},
    "pattern_test": {"seed_start": 51000, "count": 12,
                      "patterns": ["diamond", "gated_cycle"]},
    "counterfactual_test": {"seed_start": 61000, "count": 12,
                            "patterns": ["mixed_chain"]},
}


def seed_for(split: str, index: int) -> int:
    spec = SPLIT_SPECS[split]
    if not 0 <= index < spec["count"]:
        raise ValueError("structure index is outside the declared split")
    return spec["seed_start"] + index


def partition_for_seed(seed: int) -> str:
    for name, spec in SPLIT_SPECS.items():
        if spec["seed_start"] <= seed < spec["seed_start"] + spec["count"]:
            return name
    raise ValueError("generator seed is not assigned to a split")


def validate_records(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    worlds: dict[str, str] = {}
    signatures: dict[str, set[str]] = defaultdict(set)
    identifiers = set()
    grouped = defaultdict(set)
    counts = defaultdict(int)
    for row in rows:
        split = row.get("split")
        if split not in SPLIT_SPECS:
            raise ValueError("record has an unknown split")
        seed = row.get("generator_seed")
        if not isinstance(seed, int) or partition_for_seed(seed) != split:
            raise ValueError("record seed does not belong to its declared split")
        world_id = row.get("world_id")
        if not isinstance(world_id, str) or not world_id:
            raise ValueError("record is missing its structural family ID")
        previous = worlds.setdefault(world_id, split)
        if previous != split:
            raise ValueError("surface or action variants of one world cross splits")
        ident = row.get("example_id")
        if ident in identifiers:
            raise ValueError("example IDs must be unique")
        identifiers.add(ident)
        group = (world_id, row.get("action_id"), row.get("surface_id"),
                 row.get("query_type"))
        if group in grouped[split]:
            raise ValueError("duplicate world/action/surface/query variant")
        grouped[split].add(group)
        template = row.get("template_id")
        answer_template = row.get("answer_template_id")
        if not isinstance(template, str) or not template.startswith(split + ":"):
            raise ValueError("input template is not held out by split")
        if (not isinstance(answer_template, str)
                or not answer_template.startswith(split + ":")):
            raise ValueError("answer template is not held out by split")
        signatures[split].update((template, answer_template))
        counts[split] += 1
    if not worlds:
        raise ValueError("dataset must contain records")
    all_signatures = [item for values in signatures.values() for item in values]
    if len(all_signatures) != len(set(all_signatures)):
        raise ValueError("template IDs leak across splits")
    return {"examples": len(identifiers), "worlds": len(worlds),
            "examples_by_split": dict(sorted(counts.items())),
            "worlds_by_split": {name: sum(split == name for split in worlds.values())
                                for name in SPLIT_SPECS},
            "templates_by_split": {name: sorted(values)
                                   for name, values in sorted(signatures.items())}}
