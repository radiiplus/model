"""Seeded generator and deterministic oracle for structural-world examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

from research.structure.schema import RELATION_TYPES, canonical_structure, relation_index, validate_action, validate_world
from research.structure.splits import SPLIT_SPECS, seed_for, validate_records


HERE = Path(__file__).resolve().parent
GENERATOR_VERSION = "research.structure.generator-v5"
DATASET = HERE / "dataset_v5"
NAMES = ("Alder", "Beryl", "Cinder", "Dovetail", "Elm", "Fable", "Garnet",
         "Harbor", "Indigo", "Juniper", "Kestrel", "Lumen", "Morrow", "Nacre",
         "Oriel", "Pollen", "Quarry", "Runnel", "Sable", "Thistle", "Umber",
         "Vesper", "Willow", "Xylo", "Yarrow", "Zephyr", "Arden", "Bramble",
         "Cobalt", "Dahlia", "Esker", "Flint", "Grove", "Hollow", "Ilex",
         "Jasper", "Kelp", "Larch", "Mica", "North", "Opal", "Pine", "Quartz",
         "Reed", "Slate", "Tarn", "Upland", "Vale", "Wren", "Yonder")
RELATION_LABELS = {"ownership": "owns", "control": "controls",
                   "dependency": "depends on", "containment": "contains",
                   "supply": "supplies", "transfer": "transfers to"}

# Each split has a separately reviewed wording bank. Literal phrasing and
# template identifiers do not cross partition boundaries.
SURFACE_BANKS = {
    "train": {
        "header": "World ledger:", "verbs": ["holds title over", "directs", "relies upon",
            "houses", "provides for", "routes goods toward"],
        "structured_header": "Explicit world ledger record:",
        "question_prefix": "Question for the ledger:", "action_prefix": "Requested operation:",
        "selected_action_prefix": "Apply this operation:", "no_action": "Leave the state unchanged.",
        "counterfactual_prefix": "For a counterfactual ledger, consider:",
        "paraphrase": ["{a} --{rel}--> {b}."],
        "fact": ["{a} {verb} {b}."], "rule": "Influence follows ownership and control links.",
        "irrelevant": "The listed color is not needed for this question.",
        "query": {"influence": "Who can influence {target} through ownership or control, including indirect links?",
            "relation_after": "After the action, is {source} still {relation} {target}?",
            "invariant": "Name one listed relationship that stays unchanged after the action.",
            "action_effect": "Describe the exact change caused by the action."},
        "answer": "Oracle result: {value}."},
    "validation": {
        "header": "Registry of this setting:", "verbs": ["is the legal holder of", "has command over",
            "requires", "keeps inside", "delivers to", "forwards toward"],
        "structured_header": "Structured registry record:",
        "question_prefix": "Registry inquiry:", "action_prefix": "Chosen operation:",
        "selected_action_prefix": "Execute the operation:", "no_action": "No operation is applied.",
        "counterfactual_prefix": "For an alternate registry, suppose:",
        "paraphrase": ["{a} =>[{rel}] {b}."],
        "fact": ["{a} is linked to {b} by {verb}."],
        "rule": "Authority can pass along ownership and command links.",
        "irrelevant": "The color label is recorded separately from the requested relation.",
        "query": {"influence": "Which parties reach {target} by following ownership or command links?",
            "relation_after": "Once the operation is applied, does {source} retain {relation} to {target}?",
            "invariant": "Give a relationship that the operation leaves exactly as it was.",
            "action_effect": "State which relationship or property the operation changes."},
        "answer": "The computed result is {value}."},
    "iid_test": {
        "header": "Current network record:", "verbs": ["owns", "governs", "depends on",
            "stores", "supplies", "hands off to"],
        "structured_header": "Network state in structured form:",
        "question_prefix": "Network query:", "action_prefix": "State operation:",
        "selected_action_prefix": "Carry out this operation:", "no_action": "The network remains as recorded.",
        "counterfactual_prefix": "For the alternate network state, instead:",
        "paraphrase": ["{rel}: {a} -> {b}."],
        "fact": ["In this network, {a} {verb} {b}."],
        "rule": "Ownership and governance links compose into influence paths.",
        "irrelevant": "This color annotation does not affect the requested network result.",
        "query": {"influence": "Which entities have a path to {target} using ownership and governance?",
            "relation_after": "In the resulting state, does {source} still {relation} {target}?",
            "invariant": "Identify a relation preserved without change by the operation.",
            "action_effect": "What precise state update follows from the operation?"},
        "answer": "Result in this state: {value}."},
    "compositional_test": {
        "header": "Topology notes:", "verbs": ["has title to", "has authority over", "needs",
            "keeps within", "sends supplies to", "passes along"],
        "structured_header": "Topology encoded as a world record:",
        "question_prefix": "Topology problem:", "action_prefix": "Proposed state edit:",
        "selected_action_prefix": "Make the following edit:", "no_action": "Keep every state entry unchanged.",
        "counterfactual_prefix": "For a hypothetical topology, substitute:",
        "paraphrase": ["{a} -{rel}-> {b}."],
        "fact": ["The {rel} arrow points from {a} toward {b}."],
        "rule": "An ownership step followed by control steps carries influence transitively.",
        "irrelevant": "The color annotation is present but irrelevant to the requested relation.",
        "query": {"influence": "Starting from ownership and control edges, which entities reach {target}?",
            "relation_after": "Does {source} continue to {relation} {target} after the stated operation?",
            "invariant": "Report a relationship whose endpoints and status are unchanged.",
            "action_effect": "Summarize the resulting state difference exactly."},
        "answer": "Following the stated rules, {value}."},
    "pattern_test": {
        "header": "Graph facts for the case:", "verbs": ["owns", "has control over",
            "requires", "contains", "sends resources to", "transfers toward"],
        "structured_header": "Machine-readable graph case:",
        "question_prefix": "Graph task:", "action_prefix": "Graph operation:",
        "selected_action_prefix": "Execute this graph operation:", "no_action": "Do not alter the graph state.",
        "counterfactual_prefix": "In a hypothetical graph, use this instead:",
        "paraphrase": ["{a} >{rel}> {b}."],
        "fact": ["The graph's {rel} relation is {a} to {b}."],
        "rule": "Ownership and control arcs together define directed influence reachability.",
        "irrelevant": "The color value is unrelated to the graph question.",
        "query": {"influence": "Which nodes are upstream of {target} on ownership/control paths?",
            "relation_after": "After execution, does {source} remain related by {relation} to {target}?",
            "invariant": "Name a graph edge unchanged by execution.",
            "action_effect": "Describe the exact edge or property delta after execution."},
        "answer": "After applying the graph rules, {value}."},
    "counterfactual_test": {
        "header": "Alternative-state dossier:", "verbs": ["owns", "commands", "requires",
            "stores", "ships to", "redirects toward"],
        "structured_header": "Alternative world in explicit form:",
        "question_prefix": "Alternative-state question:", "action_prefix": "Hypothetical operation:",
        "selected_action_prefix": "Apply the hypothetical operation:", "no_action": "Assume no state update occurs.",
        "counterfactual_prefix": "Under the counterfactual assumption:",
        "paraphrase": ["{a} [{rel}]> {b}."],
        "fact": ["The alternative dossier places {a} as {verb} {b}."],
        "rule": "Influence propagates along ownership and command relations only.",
        "irrelevant": "This color note has no bearing on the alternative-state question.",
        "query": {"influence": "In this alternative, who reaches {target} over ownership/control links?",
            "relation_after": "Under this alternative action, does {source} still {relation} {target}?",
            "invariant": "Which relationship remains unchanged in the alternative?",
            "action_effect": "What exact change does this alternative action cause?"},
        "answer": "Under the selected alternative, {value}."},
}


def _relation(ident: str, kind: str, source: str, target: str) -> dict[str, str]:
    return {"id": ident, "type": kind, "source": source, "target": target}


def _edges(pattern: str) -> list[tuple[str, str, str]]:
    if pattern == "staggered":
        return [("ownership", "e0", "e1"), ("supply", "e1", "e2"),
                ("control", "e2", "e3"), ("containment", "e3", "e4"),
                ("dependency", "e4", "e5"), ("transfer", "e0", "e5")]
    if pattern == "mixed_chain":
        return [("ownership", "e0", "e1"), ("control", "e1", "e2"),
                ("dependency", "e2", "e3"), ("supply", "e3", "e4"),
                ("containment", "e4", "e5"), ("transfer", "e0", "e5")]
    if pattern == "diamond":
        return [("ownership", "e0", "e1"), ("control", "e1", "e3"),
                ("ownership", "e0", "e2"), ("control", "e2", "e3"),
                ("dependency", "e3", "e4"), ("supply", "e4", "e5"),
                ("containment", "e2", "e5"), ("transfer", "e0", "e5")]
    if pattern == "gated_cycle":
        return [("control", "e0", "e1"), ("control", "e1", "e2"),
                ("control", "e2", "e0"), ("ownership", "e3", "e4"),
                ("containment", "e4", "e5"), ("supply", "e3", "e5"),
                ("dependency", "e0", "e4"), ("transfer", "e2", "e5")]
    raise ValueError("unknown structural pattern")


def make_world(split: str, index: int) -> dict[str, Any]:
    seed = seed_for(split, index)
    rng = random.Random(seed)
    patterns = SPLIT_SPECS[split]["patterns"]
    pattern = patterns[index % len(patterns)]
    selected = rng.sample(NAMES, 6)
    entities = []
    for number, name in enumerate(selected):
        entities.append({"id": f"e{number}", "name": name,
                         "properties": {"color": rng.choice(("red", "blue", "green", "silver")),
                                        "capacity": rng.choice((2, 4, 7, 11, 16)),
                                        "status": rng.choice(("ready", "idle", "limited"))}})
    relations = [_relation(f"r{number}", kind, source, target)
                 for number, (kind, source, target) in enumerate(_edges(pattern))]
    world = {"world_id": f"{split}-world-{index:03d}", "generator_seed": seed,
             "pattern": pattern, "entities": entities, "relationships": relations,
             "rules": ["inf-owner-control", "same-type-path", "apply-action"],
             "constraints": ["no-self", "single-owner", "single-container", "dag-dependency"],
             "initial_state": {"active_relationships": [row["id"] for row in relations],
                               "property_overrides": {}}}
    validate_world(world)
    return world


def actions_for(world: dict[str, Any]) -> list[tuple[str, dict[str, Any] | None]]:
    relations = world["relationships"]
    owner = next(row for row in relations if row["type"] == "ownership")
    source = owner["source"]
    target = owner["target"]
    new_owner = next(row["id"] for row in world["entities"]
                     if row["id"] not in {source, target})
    remove = next(row for row in relations if row["type"] in {"control", "supply"})
    candidates = [
        ("none", None),
        ("remove", {"type": "remove_relationship", "relationship": remove["id"]}),
        ("add", {"type": "add_relationship", "relationship":
                  _relation("added", "control", "e0", "e5")}),
        ("ownership", {"type": "transfer_ownership", "from": source,
                        "to": new_owner, "target": target}),
        ("property", {"type": "set_property", "entity": "e0", "property": "status",
                       "value": "restricted"}),
    ]
    for _, action in candidates:
        validate_action(action, world)
    return candidates


def apply_action(world: dict[str, Any], action: dict[str, Any] | None) -> dict[str, Any]:
    validate_action(action, world)
    initial = world["initial_state"]
    active = list(initial["active_relationships"])
    overrides = json.loads(json.dumps(initial["property_overrides"]))
    added: dict[str, dict[str, str]] = {}
    if action is not None:
        kind = action["type"]
        if kind == "remove_relationship":
            if action["relationship"] not in active:
                raise ValueError("remove action cannot remove an inactive relationship")
            active.remove(action["relationship"])
        elif kind == "add_relationship":
            relation = action["relationship"]
            if relation["id"] in active:
                raise ValueError("add action cannot duplicate an active relationship")
            active.append(relation["id"])
            added[relation["id"]] = relation
        elif kind == "transfer_ownership":
            old = next((row for row in world["relationships"]
                        if row["type"] == "ownership" and row["source"] == action["from"]
                        and row["target"] == action["target"] and row["id"] in active), None)
            if old is None:
                raise ValueError("ownership transfer source does not own the target")
            active.remove(old["id"])
            ident = "x0"
            relation = _relation(ident, "ownership", action["to"], action["target"])
            added[ident] = relation
            active.append(ident)
        elif kind == "set_property":
            overrides.setdefault(action["entity"], {})[action["property"]] = action["value"]
    result = {"active_relationships": sorted(active), "property_overrides": overrides,
              "added_relationships": added}
    relation_index(world, result)
    return result


def influence_entities(world: dict[str, Any], state: dict[str, Any], target: str) -> list[str]:
    edges = relation_index(world, state).values()
    reverse: dict[str, list[str]] = {}
    for row in edges:
        if row["type"] in {"ownership", "control"}:
            reverse.setdefault(row["target"], []).append(row["source"])
    found = set()
    frontier = [target]
    while frontier:
        current = frontier.pop()
        for parent in reverse.get(current, []):
            if parent not in found and parent != target:
                found.add(parent)
                frontier.append(parent)
    order = {row["id"]: index for index, row in enumerate(world["entities"])}
    return [row["name"] for row in sorted(
        (item for item in world["entities"] if item["id"] in found),
        key=lambda item: order[item["id"]])]


def query_values(world: dict[str, Any], action: dict[str, Any] | None,
                 state: dict[str, Any]) -> dict[str, Any]:
    entities = world["entities"]
    relations = relation_index(world, state)
    initial_active = set(world["initial_state"]["active_relationships"])
    final_active = set(state["active_relationships"])
    ordered = {row["id"]: row for row in entities}
    target = "e2" if world["pattern"] in {"mixed_chain", "gated_cycle"} else "e3"
    answer_influence = influence_entities(world, state, target)
    rel = world["relationships"][0]
    active_relations = [row for row in world["relationships"]
                        if row["id"] in initial_active & final_active]
    invariant = next((row for row in active_relations if action is None
                      or row["id"] != action.get("relationship")), active_relations[0])
    added = final_active - initial_active
    removed = initial_active - final_active
    changed = []
    for ident in sorted(removed):
        row = next(item for item in world["relationships"] if item["id"] == ident)
        changed.append(f"removed {row['type']}({ordered[row['source']]['name']},{ordered[row['target']]['name']})")
    for ident in sorted(added):
        row = relations[ident]
        changed.append(f"added {row['type']}({ordered[row['source']]['name']},{ordered[row['target']]['name']})")
    if state["property_overrides"]:
        for entity, values in sorted(state["property_overrides"].items()):
            for prop, value in sorted(values.items()):
                before = ordered[entity]["properties"].get(prop)
                changed.append(f"{entity}.{prop}: {before} -> {value}")
    effect = "; ".join(changed) if changed else "no state changes"
    return {"influence": {"target": target, "entities": answer_influence,
                           "value": ", ".join(answer_influence) if answer_influence else "none"},
            "relation_after": {"relationship": rel["id"],
                               "source": rel["source"], "target": rel["target"],
                               "type": rel["type"], "holds": rel["id"] in final_active,
                               "value": "yes" if rel["id"] in final_active else "no"},
            "invariant": {"relationship": invariant,
                          "value": f"{invariant['type']}({ordered[invariant['source']]['name']},"
                                   f"{ordered[invariant['target']]['name']})"},
            "action_effect": {"value": effect}}


def _describe(world: dict[str, Any], action: dict[str, Any] | None,
              surface: str, split: str, rng: random.Random) -> tuple[str, str]:
    bank = SURFACE_BANKS[split]
    names = {row["id"]: row["name"] for row in world["entities"]}
    if surface == "structured":
        value = canonical_structure(world, action, None)
        return bank["structured_header"] + " " + json.dumps(value, sort_keys=True, separators=(",", ":")), \
            f"{split}:structured-v1"
    relations = list(world["relationships"])
    if surface == "reordered":
        relations.reverse()
    verbs = bank["verbs"]
    fact_lines = []
    for row in relations:
        index = RELATION_TYPES.index(row["type"])
        if surface == "paraphrased":
            template = bank["paraphrase"][0]
            text = template.format(a=names[row["source"]], b=names[row["target"]],
                                   rel=row["type"])
        else:
            template = bank["fact"][0]
            text = template.format(a=names[row["source"]], b=names[row["target"]],
                                   verb=verbs[index], rel=row["type"])
        fact_lines.append(text)
    property_rows = []
    for entity in world["entities"]:
        values = entity["properties"]
        property_rows.append(
            f"{entity['id']} {entity['name']}=[{values['color']},{values['capacity']},{values['status']}]")
    lines = [bank["header"],
             "E[id name color capacity status]: " + ";".join(property_rows),
             *fact_lines,
             "Rules: ownership/control determine influence; actions update initial state.",
             "R: " + ",".join(world["rules"]) + ".",
             "C: " + ",".join(world["constraints"]) + ".",
             "Initial: all listed relationships active; overrides none."]
    if surface == "irrelevant":
        distractor = world["entities"][2]
        lines.append(f"{bank['irrelevant']} {distractor['name']} has color {distractor['properties']['color']}.")
    if action is None:
        lines.append(bank["no_action"])
    else:
        action_prefix = (bank["counterfactual_prefix"] if surface == "counterfactual"
                         else bank["selected_action_prefix"])
        lines.append(action_prefix + " " + _action_clause(action, world) + ".")
    template_id = f"{split}:{surface}-v1"
    return " ".join(lines), template_id


def _action_clause(action: dict[str, Any] | None, world: dict[str, Any]) -> str:
    if action is None:
        return "no action"
    kind = action["type"]
    names = {row["id"]: row["name"] for row in world["entities"]}
    if kind == "remove_relationship":
        relation = next(row for row in world["relationships"]
                        if row["id"] == action["relationship"])
        return f"remove the {relation['type']} link from {names[relation['source']]} to {names[relation['target']]}"
    if kind == "add_relationship":
        relation = action["relationship"]
        return f"add a {relation['type']} link from {names[relation['source']]} to {names[relation['target']]}"
    if kind == "transfer_ownership":
        return f"move ownership of {names[action['target']]} from {names[action['from']]} to {names[action['to']]}"
    return f"set {names[action['entity']]}'s {action['property']} to {action['value']}"


def generate_records() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_SPECS}
    query_types = ("influence", "relation_after", "invariant", "action_effect")
    surfaces = ("structured", "natural", "paraphrased", "reordered", "irrelevant", "counterfactual")
    for split, spec in SPLIT_SPECS.items():
        for index in range(spec["count"]):
            world = make_world(split, index)
            actions = actions_for(world)
            if split == "counterfactual_test":
                actions = [actions[1 + index % 4], actions[1 + (index + 1) % 4]]
            for action_id, action in actions:
                resulting_state = apply_action(world, action)
                answers = query_values(world, action, resulting_state)
                structure = canonical_structure(world, action, resulting_state)
                structure_target = json.dumps(structure, sort_keys=True, separators=(",", ":"))
                for surface_number, surface in enumerate(surfaces):
                    desc, template_id = _describe(
                        world, action, surface, split, random.Random(world["generator_seed"] + surface_number))
                    for query_type in query_types:
                        value = answers[query_type]["value"]
                        bank = SURFACE_BANKS[split]
                        detail = answers[query_type]
                        query = bank["query"][query_type].format(
                            target=next(row["name"] for row in world["entities"]
                                        if row["id"] == detail.get("target", "e3")),
                            source=world["entities"][0]["name"],
                            relation=RELATION_LABELS[world["relationships"][0]["type"]])
                        input_text = f"{desc} {bank['question_prefix']} {query}"
                        answer = bank["answer"].format(value=value)
                        example_id = f"{world['world_id']}:{action_id}:{surface}:{query_type}"
                        records[split].append({
                            "example_id": example_id, "world_id": world["world_id"],
                            "generator_seed": world["generator_seed"], "split": split,
                            "pattern": world["pattern"], "action_id": action_id,
                            "action": action, "surface_id": surface, "query_type": query_type,
                            "template_id": template_id,
                            "answer_template_id": f"{split}:answer-{query_type}-v1",
                            "input": input_text, "answer_target": answer,
                            "answer_value": detail["value"], "structure_target": structure_target,
                            "oracle": detail, "world": world,
                            "resulting_state": resulting_state})
    all_rows = [row for rows in records.values() for row in rows]
    audit = validate_records(all_rows)
    return records, audit


def _sha(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_dataset(output: Path = DATASET) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    records, audit = generate_records()
    files = []
    for split, rows in records.items():
        path = output / f"{split}.jsonl"
        temp = path.with_suffix(".jsonl.tmp")
        with temp.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                        separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        files.append({"split": split, "path": path.name, "examples": len(rows),
                      "sha256": _sha(path), "bytes": path.stat().st_size})
    manifest = {"schema": 1, "generator": GENERATOR_VERSION,
                "seed_policy": SPLIT_SPECS, "relationship_types": list(RELATION_TYPES),
                "surface_ids": ["structured", "natural", "paraphrased", "reordered",
                                "irrelevant", "counterfactual"],
                "query_types": list(query_types := ("influence", "relation_after",
                                                       "invariant", "action_effect")),
                "records": audit, "files": files,
                "template_policy": "Manually authored split-specific input, query, and answer templates; no template IDs cross splits.",
                "oracle": "Deterministic graph/state rules in generator.py; no neural model generates targets."}
    manifest_path = output / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DATASET)
    args = parser.parse_args()
    print(json.dumps(write_dataset(args.output), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
