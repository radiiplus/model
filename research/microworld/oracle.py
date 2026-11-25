"""Deterministic action engine and query oracle."""

from __future__ import annotations

import copy
from typing import Any

from research.microworld.schema import (
    ACTION_TYPES, constraint_violations, effective_location, entity_index,
    validate_state, validate_world,
)


def relationships(world: dict[str, Any], state: dict[str, Any]) -> list[dict[str, str]]:
    edges = []
    for ident, position in sorted(state["positions"].items()):
        relation, target = position["relation"], position["target"]
        if relation == "inside":
            edges.append({"type": "contains", "source": target, "target": ident})
        elif relation == "held_by":
            edges.append({"type": "holds", "source": target, "target": ident})
        else:
            edges.append({"type": "located_at", "source": ident, "target": target})
    for object_id, agent_id in sorted(state["owners"].items()):
        edges.append({"type": "owns", "source": agent_id, "target": object_id})
    for left, right in world["adjacency"]:
        edges.append({"type": "connected_to", "source": left, "target": right})
    return sorted(edges, key=lambda edge: (edge["type"], edge["source"], edge["target"]))


def _action_error(world: dict[str, Any], state: dict[str, Any],
                  action: Any) -> dict[str, str] | None:
    if not isinstance(action, dict) or action.get("type") not in ACTION_TYPES:
        return {"code": "unknown_action", "message": "The action type is not supported."}
    by_id = entity_index(world)
    kind = action["type"]

    def is_type(ident: Any, expected: str) -> bool:
        return ident in by_id and by_id[ident]["type"] == expected

    if kind == "move":
        agent, destination = action.get("agent"), action.get("destination")
        if not is_type(agent, "agent") or not is_type(destination, "location"):
            return {"code": "invalid_move_endpoint", "message": "The mover or destination is invalid."}
        current = effective_location(world, state, agent)
        if destination == current:
            return {"code": "already_there", "message": "The agent is already there."}
        if not any({current, destination} == set(edge) for edge in world["adjacency"]):
            return {"code": "not_adjacent", "message": "The destination is not directly connected."}
    elif kind == "pick_up":
        agent, obj = action.get("agent"), action.get("object")
        if not is_type(agent, "agent") or not is_type(obj, "object"):
            return {"code": "invalid_pickup_target", "message": "An agent and object are required."}
        if not by_id[obj]["properties"].get("portable", False):
            return {"code": "not_portable", "message": "The object is not portable."}
        if state["positions"][obj]["relation"] == "held_by":
            return {"code": "already_held", "message": "The object is already held."}
        if state["positions"][obj]["relation"] == "inside":
            return {"code": "object_inside_container",
                    "message": "The object must be taken out of its container first."}
        if effective_location(world, state, agent) != effective_location(world, state, obj):
            return {"code": "not_colocated", "message": "The agent and object are not in the same place."}
        carried = sum(position["relation"] == "held_by" and position["target"] == agent
                      for position in state["positions"].values())
        if carried >= int(by_id[agent]["properties"]["carry_capacity"]):
            return {"code": "carry_capacity", "message": "The agent cannot carry another object."}
    elif kind in {"put_in", "take_out"}:
        agent, obj, container = (action.get("agent"), action.get("object"),
                                 action.get("container"))
        if (not is_type(agent, "agent") or not is_type(obj, "object")
                or not is_type(container, "container")):
            return {"code": "invalid_container_action", "message": "The action endpoints are invalid."}
        if effective_location(world, state, agent) != effective_location(world, state, container):
            return {"code": "container_not_reachable", "message": "The agent is not beside the container."}
        if kind == "put_in":
            if state["positions"][obj] != {"relation": "held_by", "target": agent}:
                return {"code": "object_not_held", "message": "The agent is not holding that object."}
            if container not in state["open_containers"]:
                return {"code": "container_closed", "message": "The container is closed."}
            contents = sum(position == {"relation": "inside", "target": container}
                           for position in state["positions"].values())
            if contents >= int(by_id[container]["properties"]["capacity"]):
                return {"code": "container_capacity", "message": "The container has no free space."}
        else:
            if state["positions"][obj] != {"relation": "inside", "target": container}:
                return {"code": "object_not_inside", "message": "The object is not inside that container."}
            if container not in state["open_containers"]:
                return {"code": "container_closed", "message": "The container is closed."}
            carried = sum(position["relation"] == "held_by" and position["target"] == agent
                          for position in state["positions"].values())
            if carried >= int(by_id[agent]["properties"]["carry_capacity"]):
                return {"code": "carry_capacity", "message": "The agent cannot carry another object."}
    elif kind in {"open", "close"}:
        agent, container = action.get("agent"), action.get("container")
        if not is_type(agent, "agent") or not is_type(container, "container"):
            return {"code": "invalid_latch_target", "message": "An agent and container are required."}
        if effective_location(world, state, agent) != effective_location(world, state, container):
            return {"code": "container_not_reachable", "message": "The agent is not beside the container."}
        is_open = container in state["open_containers"]
        if kind == "open" and is_open:
            return {"code": "already_open", "message": "The container is already open."}
        if kind == "close" and not is_open:
            return {"code": "already_closed", "message": "The container is already closed."}
    elif kind == "give":
        agent, recipient, obj = (action.get("agent"), action.get("recipient"),
                                action.get("object"))
        if (not is_type(agent, "agent") or not is_type(recipient, "agent")
                or not is_type(obj, "object") or agent == recipient):
            return {"code": "invalid_give_target", "message": "The giver, recipient, or object is invalid."}
        if state["positions"][obj] != {"relation": "held_by", "target": agent}:
            return {"code": "object_not_held", "message": "The giver is not holding that object."}
        if effective_location(world, state, agent) != effective_location(world, state, recipient):
            return {"code": "recipient_not_colocated", "message": "The recipient is not nearby."}
    return None


def execute(world: dict[str, Any], state: dict[str, Any],
            action: dict[str, Any] | None) -> dict[str, Any]:
    """Apply one action. Invalid actions preserve state and return a reason code."""
    validate_world(world)
    validate_state(world, state)
    before = copy.deepcopy(state)
    if action is None:
        return {"valid": True, "state": before, "reason": None,
                "violated_constraints": [], "changed": False}
    error = _action_error(world, state, action)
    if error:
        violated = [error["code"]] if error["code"] in {
            "container_capacity", "carry_capacity", "not_adjacent", "container_closed"} else []
        return {"valid": False, "state": before, "reason": error,
                "violated_constraints": violated, "changed": False}
    after = copy.deepcopy(state)
    kind = action["type"]
    if kind == "move":
        after["positions"][action["agent"]] = {
            "relation": "at", "target": action["destination"]}
    elif kind == "pick_up":
        after["positions"][action["object"]] = {
            "relation": "held_by", "target": action["agent"]}
    elif kind == "put_in":
        after["positions"][action["object"]] = {
            "relation": "inside", "target": action["container"]}
    elif kind == "take_out":
        after["positions"][action["object"]] = {
            "relation": "held_by", "target": action["agent"]}
    elif kind == "open":
        after["open_containers"] = sorted(set(after["open_containers"]) | {action["container"]})
    elif kind == "close":
        after["open_containers"] = [item for item in after["open_containers"]
                                    if item != action["container"]]
    elif kind == "give":
        after["positions"][action["object"]] = {
            "relation": "held_by", "target": action["recipient"]}
        after["owners"][action["object"]] = action["recipient"]
    violations = constraint_violations(world, after)
    if violations:
        return {"valid": False, "state": before,
                "reason": {"code": "constraint_violation", "message": "; ".join(violations)},
                "violated_constraints": violations, "changed": False}
    validate_state(world, after)
    return {"valid": True, "state": after, "reason": None,
            "violated_constraints": [], "changed": after != before}


def run_episode(world: dict[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
    state = copy.deepcopy(world["initial_state"])
    history = []
    for action in actions:
        result = execute(world, state, action)
        history.append({"action": copy.deepcopy(action), **result})
        state = result["state"]
    return {"initial_state": copy.deepcopy(world["initial_state"]), "history": history,
            "resulting_state": state}


def answer_query(world: dict[str, Any], state: dict[str, Any],
                 query: dict[str, Any]) -> dict[str, Any]:
    by_id = entity_index(world)
    kind = query["type"]
    if kind == "entities":
        return {"entities": [{"id": row["id"], "type": row["type"]}
                             for row in world["entities"]]}
    if kind == "properties":
        entity = query["entity"]
        return {"entity": entity, "properties": copy.deepcopy(by_id[entity]["properties"])}
    if kind == "relationships":
        return {"relationships": relationships(world, state)}
    if kind == "location":
        entity = query["entity"]
        return {"entity": entity, "position": copy.deepcopy(state["positions"][entity]),
                "effective_location": effective_location(world, state, entity)}
    if kind == "action_validity":
        result = execute(world, state, query["action"])
        return {key: result[key] for key in (
            "valid", "reason", "violated_constraints", "changed")}
    if kind == "transition":
        episode = run_episode(world, query["actions"])
        invalid = next((row for row in episode["history"] if not row["valid"]), None)
        return {"valid": invalid is None, "before": copy.deepcopy(state),
                "after": episode["resulting_state"],
                "reason": invalid["reason"] if invalid else None,
                "violated_constraints": invalid["violated_constraints"] if invalid else []}
    if kind == "counterfactual":
        primary = run_episode(world, query["actions"])["resulting_state"]
        alternate = run_episode(world, query["alternate_actions"])["resulting_state"]
        changed = [ident for ident in sorted(primary["positions"])
                   if primary["positions"][ident] != alternate["positions"][ident]]
        return {"primary_state": primary, "alternate_state": alternate,
                "changed_entities": changed, "outcome_changed": bool(changed)}
    if kind == "constraints":
        return {"violations": constraint_violations(world, state)}
    raise ValueError(f"unsupported query type: {kind}")
