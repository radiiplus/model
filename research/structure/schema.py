"""Typed schema and validation for deterministic structural worlds."""

from __future__ import annotations

from typing import Any


RELATION_TYPES = (
    "ownership", "control", "dependency", "containment", "supply", "transfer",
)
ACTION_TYPES = ("remove_relationship", "add_relationship", "transfer_ownership",
                "set_property")
QUERY_TYPES = ("influence", "relation_after", "invariant", "action_effect")
RELATION_CODES = {name: code for name, code in zip(RELATION_TYPES, "ocdnst")}
ACTION_CODES = {"remove_relationship": "rm", "add_relationship": "add",
                "transfer_ownership": "own", "set_property": "prop"}


def validate_world(world: dict[str, Any]) -> None:
    required = {"world_id", "pattern", "entities", "relationships", "rules",
                "constraints", "initial_state"}
    if not isinstance(world, dict) or not required <= world.keys():
        raise ValueError("world is missing required structural fields")
    entities = world["entities"]
    if not isinstance(entities, list) or not entities:
        raise ValueError("world must contain entities")
    entity_ids = [row.get("id") for row in entities if isinstance(row, dict)]
    if len(entity_ids) != len(entities) or len(set(entity_ids)) != len(entity_ids):
        raise ValueError("entity IDs must be present and unique")
    if any(not isinstance(row.get("name"), str) or not row["name"]
           or not isinstance(row.get("properties"), dict) for row in entities):
        raise ValueError("entities need names and property mappings")
    relations = world["relationships"]
    if not isinstance(relations, list):
        raise ValueError("relationships must be a list")
    relation_ids = []
    for relation in relations:
        if not isinstance(relation, dict):
            raise ValueError("relationship entries must be mappings")
        if (not isinstance(relation.get("id"), str)
                or relation.get("type") not in RELATION_TYPES
                or relation.get("source") not in entity_ids
                or relation.get("target") not in entity_ids
                or relation["source"] == relation["target"]):
            raise ValueError("relationship type or endpoint is invalid")
        relation_ids.append(relation["id"])
    if len(set(relation_ids)) != len(relation_ids):
        raise ValueError("relationship IDs must be unique")
    state = world["initial_state"]
    if not isinstance(state, dict) or set(state) != {"active_relationships", "property_overrides"}:
        raise ValueError("initial state fields are invalid")
    active = state["active_relationships"]
    if (not isinstance(active, list) or len(set(active)) != len(active)
            or not set(active) <= set(relation_ids)):
        raise ValueError("initial relationship state is invalid")
    if not isinstance(state["property_overrides"], dict):
        raise ValueError("property overrides must be a mapping")
    if not isinstance(world["rules"], list) or not world["rules"]:
        raise ValueError("world must define rules")
    if not isinstance(world["constraints"], list) or not world["constraints"]:
        raise ValueError("world must define constraints")


def validate_action(action: dict[str, Any] | None, world: dict[str, Any]) -> None:
    if action is None:
        return
    if not isinstance(action, dict) or action.get("type") not in ACTION_TYPES:
        raise ValueError("action type is invalid")
    entities = {row["id"] for row in world["entities"]}
    relations = {row["id"] for row in world["relationships"]}
    kind = action["type"]
    if kind == "remove_relationship" and action.get("relationship") not in relations:
        raise ValueError("remove action references an unknown relationship")
    if kind == "add_relationship":
        relation = action.get("relationship")
        if (not isinstance(relation, dict) or relation.get("type") not in RELATION_TYPES
                or relation.get("source") not in entities or relation.get("target") not in entities
                or relation.get("source") == relation.get("target")
                or relation.get("id") in relations):
            raise ValueError("add action relationship is invalid")
    if kind == "transfer_ownership":
        if (action.get("from") not in entities or action.get("to") not in entities
                or action.get("target") not in entities
                or action["from"] == action["to"] or action["to"] == action["target"]):
            raise ValueError("ownership transfer endpoints are invalid")
    if kind == "set_property":
        if action.get("entity") not in entities or not isinstance(action.get("property"), str):
            raise ValueError("property action target is invalid")
        if not isinstance(action.get("value"), (str, int, float, bool)):
            raise ValueError("property action value must be scalar")


def relation_index(world: dict[str, Any], state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return active relationship records, including dynamically added edges."""
    base = {row["id"]: row for row in world["relationships"]}
    active = state["active_relationships"]
    dynamic = state.get("added_relationships", {})
    result = {}
    for ident in active:
        relation = base.get(ident, dynamic.get(ident))
        if relation is None:
            raise ValueError("state activates an unknown relationship")
        result[ident] = relation
    return result


def canonical_structure(world: dict[str, Any], action: dict[str, Any] | None,
                        resulting_state: dict[str, Any] | None) -> dict[str, Any]:
    """Stable compact JSON protocol for structural supervision.

    Entity property values follow the root ``p`` key order. Relationship rows
    are ``[id, type, source, target]``. Action arrays follow ACTION_TYPES field
    order, with null representing no action.
    """
    prop_names = ["color", "capacity", "status"]
    added = (resulting_state or {}).get("added_relationships", {})
    relationships = {row["id"]: row for row in world["relationships"]}
    relationships.update(added)
    relation_rows = [relationships[key] for key in sorted(relationships)]
    relation_order = [row["id"] for row in relation_rows]

    def compact_state(state: dict[str, Any]) -> dict[str, Any]:
        active = set(state["active_relationships"])
        return {"r": "".join("1" if ident in active else "0" for ident in relation_order),
                "p": state.get("property_overrides", {})}

    compact_action = None
    if action is not None:
        kind = action["type"]
        if kind == "remove_relationship":
            compact_action = [ACTION_CODES[kind], action["relationship"]]
        elif kind == "add_relationship":
            row = action["relationship"]
            compact_action = [ACTION_CODES[kind], row["id"], RELATION_CODES[row["type"]],
                              row["source"], row["target"]]
        elif kind == "transfer_ownership":
            compact_action = [ACTION_CODES[kind], action["from"], action["to"], action["target"]]
        else:
            compact_action = [ACTION_CODES[kind], action["entity"], action["property"], action["value"]]
    value = {"v": 1, "p": prop_names,
             "e": [[row["id"], row["name"],
                    [row["properties"].get(name) for name in prop_names]]
                   for row in world["entities"]],
             "t": [[code, name] for name, code in RELATION_CODES.items()],
             "r": [[row["id"], RELATION_CODES[row["type"]], row["source"], row["target"]]
                   for row in relation_rows],
             "u": list(world["rules"]), "c": list(world["constraints"]),
             "i": compact_state(world["initial_state"]), "a": compact_action}
    if resulting_state is not None:
        value["s"] = compact_state(resulting_state)
    return value
