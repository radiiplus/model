"""Seeded, world-split semantic answer and correction benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DATASET = HERE / "semantic_correction_bench"
SPECS = {"train": (13000, 24), "validation": (23000, 8),
         "iid_test": (33000, 8), "compositional_test": (43000, 8),
         "counterfactual_test": (53000, 8)}
NAMES = ("Alder", "Beryl", "Cinder", "Dovetail", "Elm", "Fable",
         "Garnet", "Harbor", "Indigo", "Juniper", "Kestrel", "Lumen")
SURFACES = ("explicit", "narrative", "reordered")
CONTEXTS = {"cold": -2, "mild": 0, "warm": 1}

INPUT_BANK = {
    "train": {
        "explicit": "Supply ledger: {edges} Capacity at {context}: {effective}. {consumer} demand: {demand}. Can {source} power {consumer} in {context}?",
        "narrative": "{source} sends energy through {path} to {consumer}. In {context}, output is {effective} units and {consumer} needs {demand}. Would it run?",
        "reordered": "{consumer} needs {demand} units. After the {context} adjustment, {source} has {effective}. The route is {path}. Is supply sufficient?"},
    "validation": {
        "explicit": "Resource record: {edges} Under {context} conditions output={effective}; required={demand}. Does {source} support {consumer}?",
        "narrative": "For {consumer}, {source} provides {effective} units in {context} weather along {path}; the requirement is {demand}. Can it operate?",
        "reordered": "Required level: {demand}. Route: {path}. {context} output from {source}: {effective}. Is {consumer} supported?"},
    "iid_test": {
        "explicit": "Network facts: {edges} Context {context} leaves {source} with {effective}; {consumer} requires {demand}. Will the connection work?",
        "narrative": "The path from {source} to {consumer} goes by {path}. Available energy is {effective} in {context}, against a need of {demand}. Does it work?",
        "reordered": "{consumer} requires {demand}. In {context}, {source} provides {effective} using {path}. Is that enough?"},
    "compositional_test": {
        "explicit": "Topology: {edges} Effective {context} supply={effective}; endpoint demand={demand}. Can {source} serve {consumer}?",
        "narrative": "To reach {consumer}, energy must pass along {path}. {source} offers {effective} units in {context} while {consumer} needs {demand}. Is service possible?",
        "reordered": "Demand at {consumer} is {demand}; context is {context}; capacity at {source} is {effective}. Route: {path}. Can it serve?"},
    "counterfactual_test": {
        "explicit": "Alternative-state facts: {edges} If {action_text}, output becomes {effective} in {context}; demand={demand}. Would {source} power {consumer}?",
        "narrative": "Assume {action_text}. Along {path}, {source} would have {effective} units in {context}; {consumer} needs {demand}. Would it operate?",
        "reordered": "{consumer} needs {demand}. Under {context} and the alternative {action_text}, {source} has {effective}. Route: {path}. Does it work?"}}

ANSWER_BANK = {
    "train": {
        True: "Yes. {source} has enough available output to meet {consumer}'s demand in {context}.",
        False: "No. {source}'s available output is below {consumer}'s demand in {context}."},
    "validation": {
        True: "The connection should work: available supply meets {consumer}'s requirement in {context}.",
        False: "The connection will fail because {consumer} needs more than {source} can provide in {context}."},
    "iid_test": {
        True: "It can operate in {context}; {source} supplies at least the amount {consumer} needs.",
        False: "It cannot operate in {context}; the supply from {source} does not reach {consumer}'s required level."},
    "compositional_test": {
        True: "Yes, the route can serve {consumer} in {context} because the delivered amount covers its need.",
        False: "No, the route cannot serve {consumer} in {context}: the delivered amount is short of its need."},
    "counterfactual_test": {
        True: "With that alternative, {consumer} can run in {context} because supply is sufficient.",
        False: "Even with that alternative, {consumer} cannot run in {context}; supply remains insufficient."}}


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def world(split: str, index: int) -> dict[str, Any]:
    start, count = SPECS[split]
    if not 0 <= index < count:
        raise ValueError("world index is outside its split")
    seed = start + index
    rng = random.Random(seed)
    source, consumer, relay = rng.sample(NAMES, 3)
    base = rng.randint(2, 8)
    demand = rng.randint(2, 8)
    path_length = 2 if split == "compositional_test" else 1
    path = [source, consumer] if path_length == 1 else [source, relay, consumer]
    relations = [{"from": path[pos], "type": "supplies", "to": path[pos + 1]}
                 for pos in range(len(path) - 1)]
    return {"world_id": f"{split}-world-{index:03d}", "seed": seed,
            "source": source, "consumer": consumer, "relay": relay,
            "base_capacity": base, "demand": demand,
            "path": path, "relationships": relations,
            "pattern": "two_hop_supply" if path_length == 2 else "direct_supply"}


def make_rows(split: str) -> list[dict[str, Any]]:
    output = []
    start, count = SPECS[split]
    for index in range(count):
        graph = world(split, index)
        contexts = ("cold", "warm")
        actions = ("none", "insulate") if split == "counterfactual_test" else ("none",)
        for context in contexts:
            for action in actions:
                adjustment = CONTEXTS[context] + (2 if action == "insulate" else 0)
                effective = max(0, graph["base_capacity"] + adjustment)
                reachable = len(graph["path"]) >= 2 and graph["relationships"]
                answer = bool(reachable and effective >= graph["demand"])
                action_text = "add insulation" if action == "insulate" else "make no change"
                edges = "; ".join(f"{edge['from']} supplies {edge['to']}" for edge in graph["relationships"])
                for surface in SURFACES:
                    fields = {**graph, "context": context, "effective": effective,
                              "action": action, "action_text": action_text,
                              "edges": edges, "path_text": " -> ".join(graph["path"]),
                              "path": " -> ".join(graph["path"])}
                    prompt = INPUT_BANK[split][surface].format(**fields)
                    target = ANSWER_BANK[split][answer].format(**fields)
                    output.append({"example_id": f"{graph['world_id']}:{context}:{action}:{surface}",
                        "world_id": graph["world_id"], "generator_seed": graph["seed"],
                        "split": split, "pattern": graph["pattern"], "surface": surface,
                        "context": context, "action": action, "world": graph,
                        "effective_capacity": effective, "demand": graph["demand"],
                        "answer_semantic": "yes" if answer else "no",
                        "answer_target": target, "input": prompt})
    return output


def generate() -> dict[str, list[dict[str, Any]]]:
    data = {name: make_rows(name) for name in SPECS}
    seen: dict[str, str] = {}
    ids = set()
    for split, rows in data.items():
        for row in rows:
            if row["example_id"] in ids:
                raise ValueError("duplicate semantic example ID")
            ids.add(row["example_id"])
            prior = seen.setdefault(row["world_id"], split)
            if prior != split:
                raise ValueError("world or surface variants cross splits")
            if row["generator_seed"] != SPECS[split][0] + int(row["world_id"].rsplit("-", 1)[1]):
                raise ValueError("seed does not match the declared world split")
    return data


def write_dataset(output: Path = DATASET) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    rows = generate()
    entries = []
    for split, values in rows.items():
        raw = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                       for row in values).encode("utf-8")
        path = output / f"{split}.jsonl"
        path.write_bytes(raw)
        entries.append({"split": split, "path": path.name, "worlds": SPECS[split][1],
                        "examples": len(values), "sha256": sha_bytes(raw)})
    manifest = {"schema": 1, "generator": "semantic-correction-benchmark-v1",
        "seed_ranges": {key: {"start": value[0], "count": value[1]}
                        for key, value in SPECS.items()},
        "surfaces": list(SURFACES), "files": entries,
        "worlds": sum(count for _, count in SPECS.values()),
        "examples": sum(len(values) for values in rows.values()),
        "answer_labels": "deterministic capacity, context, relation-path oracle"}
    body = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")
    manifest["manifest_sha256"] = sha_bytes(body)
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n",
                                           encoding="utf-8")
    return manifest


def semantic_label(text: str) -> str:
    """Small reviewed yes/no parser. Unknown text is not silently counted wrong by exact match."""
    value = " ".join(text.casefold().replace("'", "").split())
    negatives = ("no", "not enough", "not sufficient", "not operate", "not work",
                 "cannot", "cant", "will not", "wont", "would not", "does not",
                 "doesnt", "below", "insufficient", "unable", "fail", "short of")
    positives = ("yes", "can", "will work", "will run", "should work", "enough",
                 "meets", "sufficient", "supports", "operates")
    if any(term in value for term in negatives):
        return "no"
    if any(term in value for term in positives):
        return "yes"
    return "unknown"


def selected_evaluation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if row["surface"] != "explicit":
            continue
        if row["split"] == "counterfactual_test" and row["action"] not in {"none", "insulate"}:
            continue
        selected.append(row)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DATASET)
    args = parser.parse_args()
    print(json.dumps(write_dataset(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
