"""Seeded world, episode, and multi-surface dataset generation."""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import io
import json
import random
import re
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from research.microworld import VERSION
from research.microworld.language import (
    COMMON_CONSTRUCTIONS, HELDOUT_CONSTRUCTIONS,
    answer_text, render,
)
from research.microworld.oracle import answer_query, run_episode
from research.microworld.schema import ACTION_SPECS, validate_world
from research.microworld.splits import (
    MAX_WORLDS, split_assignments, stable_hash, validate_world_and_example_splits,
)
from research.microworld.dense import align_annotations, audit_annotations
from research.interaction.tokenizer import Tokenizer


RULES = ["one_position_per_entity", "container_must_be_open_for_access",
         "agents_move_over_adjacent_locations", "carried_objects_follow_the_agent",
         "ownership_is_independent_of_position"]
CONSTRAINTS = ["containment_acyclic", "container_capacity", "agent_carry_capacity",
               "all_entities_resolve_to_one_location",
               "container_only_contains_objects_or_containers", "single_position"]
TOKENIZER_PATH = Path(__file__).resolve().parents[1] / "interaction" / "data" / "ready7" / "tokenizer.json"


def _base36(value: int) -> str:
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    digits = []
    while value:
        value, remainder = divmod(value, 36)
        digits.append(alphabet[remainder])
    return "".join(reversed(digits or ["0"]))
def make_world(index: int, split: str, seed: int = 920000) -> dict[str, Any]:
    world_seed = seed + index
    rng = random.Random(world_seed)
    world_id = f"world-{world_seed:08d}"
    suffix = _base36(world_seed)[-4:].rjust(4, "0")
    names = ("Mira", "Tovin", "Nell", "Oren", "Sela", "Pax", "Rumi", "Ivo")
    pronouns = ("she", "he", "they")
    agent_names = rng.sample(names, 2)
    entities = []

    def add(ident: str, kind: str, label: str, properties: dict[str, Any]) -> None:
        entities.append({"id": ident, "name": f"{label}-{suffix}", "type": kind,
                         "properties": properties})

    add("agent-0", "agent", agent_names[0], {
        "pronoun": rng.choice(pronouns), "carry_capacity": 2})
    add("agent-1", "agent", agent_names[1], {
        "pronoun": rng.choice(pronouns), "carry_capacity": 2})
    add("container-0", "container", "box", {"capacity": 4, "material": rng.choice(
        ("wood", "metal", "glass")), "latch_type": rng.choice(("hinged", "sliding"))})
    add("container-1", "container", "case", {"capacity": 2, "material": rng.choice(
        ("wood", "metal", "glass")), "latch_type": rng.choice(("hinged", "sliding"))})
    for number in range(3):
        add(f"object-{number}", "object", ("key", "book", "coin")[number], {
            "color": rng.choice(("red", "blue", "green", "silver")),
            "weight": rng.choice((1, 2)), "portable": True})
    for number in range(3):
        add(f"location-{number}", "location", ("hall", "study", "vault")[number], {
            "kind": "room", "level": number})

    if split == "compositional_test":
        composition = "nested_container_chain"
    elif split in {"train", "validation", "iid_test"} and index % 3 == 0:
        composition = "single_container_nesting"
    else:
        composition = "flat_placement"
    container_is_nested = composition in {"single_container_nesting", "nested_container_chain"}
    positions = {
        "agent-0": {"relation": "at", "target": "location-0"},
        "agent-1": {"relation": "at", "target": "location-2"},
        "container-0": {"relation": "at", "target": "location-0"},
        "container-1": ({"relation": "inside", "target": "container-0"}
                        if container_is_nested else
                        {"relation": "at", "target": "location-1"}),
        "object-0": {"relation": "at", "target": "location-0"},
        "object-1": {"relation": "inside", "target": "container-0"},
        "object-2": ({"relation": "inside", "target": "container-1"}
                     if composition == "nested_container_chain" else
                     {"relation": "at", "target": "location-1"}
                     if composition == "single_container_nesting" else
                     {"relation": "inside", "target": "container-1"}),
    }
    world = {
        "world_id": world_id, "seed": world_seed, "split": split,
        "composition": composition, "entities": entities,
        "adjacency": [["location-0", "location-1"], ["location-1", "location-2"]],
        "rules": list(RULES), "constraints": list(CONSTRAINTS),
        "action_specs": ACTION_SPECS,
        "initial_state": {"positions": positions,
                          "open_containers": ["container-0"],
                          "owners": {"object-0": "agent-0", "object-1": "agent-1"}},
    }
    validate_world(world)
    return world


def episodes_for(world: dict[str, Any]) -> list[dict[str, Any]]:
    actions = [
        {"type": "pick_up", "agent": "agent-0", "object": "object-0"},
        {"type": "put_in", "agent": "agent-0", "object": "object-0",
         "container": "container-0"},
        {"type": "move", "agent": "agent-0", "destination": "location-1"},
    ]
    give = [
        {"type": "pick_up", "agent": "agent-0", "object": "object-0"},
        {"type": "move", "agent": "agent-0", "destination": "location-1"},
        {"type": "move", "agent": "agent-0", "destination": "location-2"},
        {"type": "give", "agent": "agent-0", "recipient": "agent-1", "object": "object-0"},
    ]
    invalid = [
        {"type": "close", "agent": "agent-0", "container": "container-0"},
        {"type": "take_out", "agent": "agent-0", "object": "object-1",
         "container": "container-0"},
    ]
    result = [
        {"episode_id": f"{world['world_id']}:noop", "kind": "noop", "actions": []},
        {"episode_id": f"{world['world_id']}:pickup", "kind": "pickup", "actions": [actions[0]]},
        {"episode_id": f"{world['world_id']}:store", "kind": "store", "actions": actions[:2]},
        {"episode_id": f"{world['world_id']}:travel", "kind": "travel", "actions": [actions[2]]},
        {"episode_id": f"{world['world_id']}:give", "kind": "give", "actions": give},
        {"episode_id": f"{world['world_id']}:invalid", "kind": "invalid", "actions": invalid},
    ]
    if world["split"] == "counterfactual_test":
        result.append({
            "episode_id": f"{world['world_id']}:counterfactual", "kind": "counterfactual",
            "actions": [actions[0]],
            "counterfactual_actions": [],
            "counterfactual_entity": "object-0",
        })
    return result


def _last_action_state(world: dict[str, Any], episode: dict[str, Any]) -> dict[str, Any]:
    if len(episode["actions"]) < 2:
        return world["initial_state"]
    return run_episode(world, episode["actions"][:-1])["resulting_state"]


def _queries(world: dict[str, Any], episode: dict[str, Any],
             resulting_state: dict[str, Any]) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    queries = [
        {"type": "entities"},
        {"type": "properties", "entity": "object-0"},
        {"type": "relationships"},
        {"type": "location", "entity": "object-0"},
        {"type": "action_validity",
         "action": episode["actions"][-1] if episode["actions"] else None},
        {"type": "transition", "actions": episode["actions"]},
        {"type": "constraints"},
    ]
    for query in queries:
        state = (_last_action_state(world, episode) if query["type"] == "action_validity"
                 else resulting_state)
        if query["type"] == "transition":
            state = world["initial_state"]
        yield query, answer_query(world, state, query)
    if episode.get("counterfactual_actions") is not None:
        query = {"type": "counterfactual", "actions": episode["actions"],
                 "alternate_actions": episode["counterfactual_actions"]}
        yield query, answer_query(world, world["initial_state"], query)


def _counterfactual_record(world: dict[str, Any], episode: dict[str, Any],
                           answer: dict[str, Any]) -> dict[str, Any] | None:
    if not episode.get("counterfactual_actions") and episode.get("kind") != "counterfactual":
        return None
    if episode.get("kind") != "counterfactual":
        return None
    return {"primary_state": answer["primary_state"],
            "alternate_state": answer["alternate_state"],
            "entity": episode["counterfactual_entity"]}


def build_world_examples(world: dict[str, Any]) -> list[dict[str, Any]]:
    constructions = (HELDOUT_CONSTRUCTIONS if world["split"] == "grammar_test"
                    else COMMON_CONSTRUCTIONS)
    examples = []
    for episode in episodes_for(world):
        run = run_episode(world, episode["actions"])
        final_state = run["resulting_state"]
        transition_id = stable_hash({
            "entities": world["entities"], "initial": world["initial_state"],
            "actions": episode["actions"], "result": final_state,
            "history": [{key: item[key] for key in ("valid", "reason")}
                        for item in run["history"]],
        })
        for query, answer in _queries(world, episode, final_state):
            for construction in constructions:
                prompt, template_id = render(world, episode, query, construction)
                target = answer_text(answer, query, world, construction)
                cf = (_counterfactual_record(world, episode, answer)
                      if query["type"] == "counterfactual" else None)
                answer_template_id = (
                    ("heldout_answer:" if construction.startswith("heldout:")
                     else "common_answer:") + query["type"])
                examples.append({
                    "example_id": f"{episode['episode_id']}:{query['type']}:{construction}",
                    "world_id": world["world_id"], "episode_id": episode["episode_id"],
                    "split": world["split"], "composition": world["composition"],
                    "episode_kind": episode["kind"], "query_type": query["type"],
                    "construction_id": construction, "template_id": template_id,
                    "answer_template_id": answer_template_id,
                    "transition_id": transition_id, "prompt": prompt,
                    "actions": episode["actions"], "query": query,
                    "initial_state": world["initial_state"],
                    "resulting_state": final_state, "oracle_state": (
                        _last_action_state(world, episode)
                        if query["type"] == "action_validity" else
                        world["initial_state"] if query["type"] == "transition"
                        else final_state),
                    "ground_truth": answer, "answer_text": target,
                    "counterfactual": cf,
                })
    return examples


def iter_worlds(world_count: int, seed: int = 920000) -> Iterator[dict[str, Any]]:
    assignments = split_assignments(world_count, seed)
    for index in range(world_count):
        yield make_world(index, assignments[index], seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextlib.contextmanager
def _gzip_text_writer(path: Path) -> Iterator[Any]:
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename="", fileobj=raw, mode="wb",
                               compresslevel=1, mtime=0)
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
    try:
        yield text
    finally:
        text.close()
        raw.close()


def write_dataset(output: Path, world_count: int = 100, seed: int = 920000,
                  include_baselines: bool = False,
                  tokenizer_path: Path = TOKENIZER_PATH) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite dataset directory: {output}")
    output.mkdir(parents=True)
    assignments = split_assignments(world_count, seed)
    tokenizer = Tokenizer.load(tokenizer_path)
    tokenizer_sha256 = _sha256(tokenizer_path)
    split_names = list(dict.fromkeys(assignments.values()))
    file_counts: dict[str, int] = {}
    annotation_counts: Counter[str] = Counter()
    examples_by_split = {name: 0 for name in split_names}
    tokens_by_split = {name: 0 for name in split_names}
    model_tokens_by_split = {name: 0 for name in split_names}
    max_model_tokens_by_split = {name: 0 for name in split_names}
    templates = {name: set() for name in split_names}
    constructions = set()
    composition_by_split = {name: set() for name in split_names}
    composition_counts = {name: {} for name in split_names}
    episode_ids_by_split = {name: set() for name in split_names}
    heldout_templates: set[str] = set()
    other_templates: set[str] = set()
    heldout_answer_templates: set[str] = set()
    other_answer_templates: set[str] = set()
    counterfactual_episodes = set()
    world_ids = []
    world_buffer = [] if include_baselines else None
    example_buffer = [] if include_baselines else None
    if include_baselines and world_count > 1000:
        raise ValueError("baseline materialization is limited to 1,000 worlds; generate data without --baselines")
    with tempfile.TemporaryDirectory(prefix="microworld-audit-") as temp:
        database = sqlite3.connect(Path(temp) / "uniques.sqlite")
        database.execute("CREATE TABLE seen (kind TEXT, digest TEXT, split TEXT, PRIMARY KEY(kind, digest))")
        database.execute("CREATE TABLE uniques (kind TEXT, digest TEXT, PRIMARY KEY(kind, digest))")
        database.execute("CREATE TABLE prompt_targets (prompt TEXT PRIMARY KEY, target TEXT)")
        writers = {}
        try:
            with contextlib.ExitStack() as stack:
                for split in split_names:
                    for kind in ("worlds", "examples"):
                        path = output / f"{kind}_{split}.jsonl.gz"
                        writers[(kind, split)] = stack.enter_context(
                            _gzip_text_writer(path))
                        file_counts[f"{kind}:{split}"] = 0
                    annotation_path = output / f"annotations_{split}.jsonl.gz"
                    writers[("annotations", split)] = stack.enter_context(
                        _gzip_text_writer(annotation_path))
                    file_counts[f"annotations:{split}"] = 0

                for index in range(world_count):
                    split = assignments[index]
                    world = make_world(index, split, seed)
                    world_ids.append(world["world_id"])
                    composition_by_split[split].add(world["composition"])
                    composition_counts[split][world["composition"]] = (
                        composition_counts[split].get(world["composition"], 0) + 1)
                    world_line = json.dumps(world, sort_keys=True, separators=(",", ":"))
                    writers[("worlds", split)].write(world_line + "\n")
                    file_counts[f"worlds:{split}"] += 1
                    if world_buffer is not None:
                        world_buffer.append(world)
                    for row in build_world_examples(world):
                        prompt_hash = hashlib.sha256(row["prompt"].encode()).hexdigest()
                        transition_hash = row["transition_id"]
                        target_hash = stable_hash(row["ground_truth"])
                        prior_target = database.execute(
                            "SELECT target FROM prompt_targets WHERE prompt=?",
                            (prompt_hash,)).fetchone()
                        if prior_target and prior_target[0] != target_hash:
                            raise ValueError("one prompt maps to conflicting ground-truth targets")
                        database.execute("INSERT OR IGNORE INTO prompt_targets VALUES (?, ?)",
                                         (prompt_hash, target_hash))
                        for kind, digest in (("prompt", prompt_hash),
                                             ("transition", transition_hash)):
                            prior = database.execute(
                                "SELECT split FROM seen WHERE kind=? AND digest=?",
                                (kind, digest)).fetchone()
                            if prior and prior[0] != split:
                                raise ValueError(f"{kind} duplicates across partitions")
                            database.execute(
                                "INSERT OR IGNORE INTO seen VALUES (?, ?, ?)",
                                (kind, digest, split))
                            database.execute(
                                "INSERT OR IGNORE INTO uniques VALUES (?, ?)",
                                (kind, digest))
                        templates[split].add(row["template_id"])
                        episode_ids_by_split[split].add(row["episode_id"])
                        constructions.add(row["construction_id"])
                        for kind, value in (("template", row["template_id"]),
                                            ("answer_template", row["answer_template_id"]),
                                            ("construction", row["construction_id"])):
                            database.execute("INSERT OR IGNORE INTO uniques VALUES (?, ?)",
                                             (kind, hashlib.sha256(value.encode()).hexdigest()))
                        for sentence in re.split(
                                r"(?<=[.!?])\s+", row["prompt"] + " " + row["answer_text"]):
                            sentence = " ".join(sentence.casefold().split())
                            if sentence:
                                database.execute("INSERT OR IGNORE INTO uniques VALUES (?, ?)",
                                    ("sentence", hashlib.sha256(sentence.encode()).hexdigest()))
                        if split == "grammar_test":
                            heldout_templates.add(row["template_id"])
                            heldout_answer_templates.add(row["answer_template_id"])
                        else:
                            other_templates.add(row["template_id"])
                            other_answer_templates.add(row["answer_template_id"])
                        if row.get("counterfactual"):
                            pair = row["episode_id"]
                            if (row["counterfactual"]["primary_state"]
                                    == row["counterfactual"]["alternate_state"]):
                                raise ValueError("counterfactual has identical resulting states")
                            ident = row["counterfactual"]["entity"]
                            if (row["counterfactual"]["primary_state"]["positions"][ident]
                                    == row["counterfactual"]["alternate_state"]["positions"][ident]):
                                raise ValueError("counterfactual leaves queried outcome unchanged")
                            counterfactual_episodes.add(pair)
                        line = json.dumps(row, sort_keys=True, separators=(",", ":"))
                        writers[("examples", split)].write(line + "\n")
                        file_counts[f"examples:{split}"] += 1
                        aligned = align_annotations(row, world, tokenizer)
                        audit_annotations(row, world, aligned, tokenizer)
                        model_tokens = len(aligned["token_ids"])
                        if model_tokens > 1024:
                            raise ValueError(
                                f"{split} example {row['example_id']} exceeds 1024 tokenizer tokens")
                        annotation_counts.update(
                            item["type"] for item in aligned["annotations"])
                        writers[("annotations", split)].write(
                            json.dumps(aligned, sort_keys=True, separators=(",", ":")) + "\n")
                        file_counts[f"annotations:{split}"] += 1
                        examples_by_split[split] += 1
                        tokens_by_split[split] += (
                            len(row["prompt"].split()) + len(row["answer_text"].split()))
                        model_tokens_by_split[split] += model_tokens
                        max_model_tokens_by_split[split] = max(
                            max_model_tokens_by_split[split], model_tokens)
                        if example_buffer is not None:
                            example_buffer.append(row)
            unique = {kind: count for kind, count in database.execute(
                "SELECT kind, COUNT(*) FROM uniques GROUP BY kind")}
        finally:
            database.commit()
            database.close()

    if heldout_templates & other_templates:
        raise ValueError("held-out grammar/template construction leaked")
    if heldout_answer_templates & other_answer_templates:
        raise ValueError("held-out answer template construction leaked")
    if ("nested_container_chain" in composition_by_split["compositional_test"]
            and "nested_container_chain" in composition_by_split["train"]):
        raise ValueError("held-out composition appears in training")

    files = []
    for split in split_names:
        for kind in ("worlds", "examples", "annotations"):
            path = output / f"{kind}_{split}.jsonl.gz"
            files.append({"split": split, "kind": kind, "path": path.name,
                          "records": file_counts[f"{kind}:{split}"],
                          "bytes": path.stat().st_size, "sha256": _sha256(path)})
    baseline_report = None
    if world_buffer is not None and example_buffer is not None:
        from research.microworld.baselines import evaluate as evaluate_baselines
        baseline_report = evaluate_baselines(world_buffer, example_buffer)
        baseline_raw = (json.dumps(baseline_report, sort_keys=True, indent=2) + "\n").encode()
        (output / "baselines.json").write_bytes(baseline_raw)
        files.append({"split": "all", "kind": "baselines", "path": "baselines.json",
                      "records": len(baseline_report["partitions"]),
                      "bytes": len(baseline_raw),
                      "sha256": hashlib.sha256(baseline_raw).hexdigest()})
    audit = {
        "worlds": world_count, "examples": sum(examples_by_split.values()),
        "worlds_by_split": {split: file_counts[f"worlds:{split}"] for split in split_names},
        "examples_by_split": examples_by_split,
        "annotation_labels_by_type": dict(sorted(annotation_counts.items())),
        "annotation_labels": sum(annotation_counts.values()),
        "episodes_by_split": {split: len(values)
                              for split, values in episode_ids_by_split.items()},
        "compositions_by_split": composition_counts,
        "unique_prompts": unique.get("prompt", 0),
        "unique_transitions": unique.get("transition", 0),
        "heldout_grammar_templates": sorted(heldout_templates),
        "heldout_grammar_answer_templates": sorted(heldout_answer_templates),
        "counterfactual_pairs_checked": len(counterfactual_episodes),
        "world_partition_hash": stable_hash(sorted(world_ids)),
    }
    if include_baselines and world_buffer is not None and example_buffer is not None:
        # Also run the full list validator on the bounded pilot.
        audit = {**audit, **validate_world_and_example_splits(world_buffer, example_buffer)}
    manifest = {
        "schema": 1, "generator": VERSION, "seed": seed,
        "world_count_requested": world_count, "maximum_supported_worlds": MAX_WORLDS,
        "files": files, "statistics": {
            **audit, "unique_worlds": world_count,
            "unique_state_transitions": unique.get("transition", 0),
            "unique_grammar_constructions": unique.get("construction", 0),
            "unique_templates": unique.get("template", 0),
            "unique_answer_templates": unique.get("answer_template", 0),
            "unique_sentences": unique.get("sentence", 0),
            "total_tokens_whitespace": sum(tokens_by_split.values()),
            "total_training_presentations": examples_by_split["train"],
            "training_tokens_whitespace": tokens_by_split["train"],
            "tokens_by_split": tokens_by_split,
            "model_tokens_by_split": model_tokens_by_split,
            "maximum_model_tokens_by_split": max_model_tokens_by_split,
            "sequence_limit": 1024,
            "annotation_counts_by_type": dict(sorted(annotation_counts.items())),
            "annotation_labels": sum(annotation_counts.values()),
            "tokenizer_digest": tokenizer.digest,
            "tokenizer_file_sha256": tokenizer_sha256,
        },
        "baselines": baseline_report,
        "split_ratios": {"train": 0.70, "validation": 0.10, "iid_test": 0.10,
                         "compositional_test": 0.04, "counterfactual_test": 0.04,
                         "grammar_test": 0.02},
        "token_count_method": "whitespace word count over prompt plus natural-language target",
    }
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(
        manifest, sort_keys=True, indent=2).encode()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=920000)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH)
    parser.add_argument("--baselines", action="store_true",
                        help="run baseline comparisons (limited to 1,000 worlds)")
    args = parser.parse_args()
    if args.world_count > MAX_WORLDS:
        parser.error(f"--world-count cannot exceed {MAX_WORLDS}")
    print(json.dumps(write_dataset(args.output, args.world_count, args.seed, args.baselines,
                                  args.tokenizer),
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
