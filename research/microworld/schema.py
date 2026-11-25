"""World and state validation for the deterministic room microworld."""

from __future__ import annotations

from typing import Any


ENTITY_TYPES = ("agent", "object", "container", "location")
POSITION_RELATIONS = ("at", "inside", "held_by")
ACTION_TYPES = ("move", "pick_up", "put_in", "take_out", "open", "close", "give")
ACTION_SPECS = {
    "move": {"preconditions": ["agent exists", "destination is adjacent", "destination differs from current room"],
             "effects": ["agent position changes", "held objects follow agent"]},
    "pick_up": {"preconditions": ["object is portable", "object is not held or contained", "agent and object share a room", "carry capacity remains"],
                "effects": ["object position becomes held_by(agent)"]},
    "put_in": {"preconditions": ["agent holds object", "container is open", "agent is beside container", "container capacity remains"],
               "effects": ["object position becomes inside(container)"]},
    "take_out": {"preconditions": ["object is inside container", "container is open", "agent is beside container", "carry capacity remains"],
                 "effects": ["object position becomes held_by(agent)"]},
    "open": {"preconditions": ["agent is beside a closed container"],
             "effects": ["container enters open_containers"]},
    "close": {"preconditions": ["agent is beside an open container"],
              "effects": ["container leaves open_containers"]},
    "give": {"preconditions": ["agent holds object", "recipient is a different agent", "agents share a room"],
             "effects": ["recipient holds object", "ownership transfers to recipient"]},
}


def entity_index(world: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entity["id"]: entity for entity in world["entities"]}


def validate_world(world: dict[str, Any]) -> None:
    required = {"world_id", "seed", "split", "composition", "entities", "adjacency",
                "rules", "constraints", "action_specs", "initial_state"}
    if not isinstance(world, dict) or not required <= world.keys():
        raise ValueError("world is missing required fields")
    entities = world["entities"]
    if not isinstance(entities, list) or not entities:
        raise ValueError("world needs entities")
    ids = [row.get("id") for row in entities if isinstance(row, dict)]
    if len(ids) != len(entities) or len(ids) != len(set(ids)):
        raise ValueError("entity IDs must be present and unique")
    by_id = entity_index(world)
    if any(row.get("type") not in ENTITY_TYPES or not row.get("name")
           or not isinstance(row.get("properties"), dict) for row in entities):
        raise ValueError("entity type, name, or properties are invalid")
    locations = {ident for ident, row in by_id.items() if row["type"] == "location"}
    if not locations:
        raise ValueError("world needs at least one location")
    for edge in world["adjacency"]:
        if (len(edge) != 2 or edge[0] not in locations or edge[1] not in locations
                or edge[0] == edge[1]):
            raise ValueError("location adjacency is invalid")
    if not isinstance(world["rules"], list) or not world["rules"]:
        raise ValueError("world must define rules")
    if not isinstance(world["constraints"], list) or not world["constraints"]:
        raise ValueError("world must define constraints")
    if world["action_specs"] != ACTION_SPECS:
        raise ValueError("action specifications do not match the oracle rules")
    validate_state(world, world["initial_state"])


def validate_state(world: dict[str, Any], state: dict[str, Any]) -> list[str]:
    by_id = entity_index(world)
    locations = {ident for ident, row in by_id.items() if row["type"] == "location"}
    positions = state.get("positions")
    if not isinstance(positions, dict) or set(positions) != set(by_id) - locations:
        raise ValueError("every non-location entity must have exactly one position")
    owners = state.get("owners")
    if not isinstance(owners, dict):
        raise ValueError("ownership state must be a mapping")
    for ident, position in positions.items():
        if not isinstance(position, dict) or set(position) != {"relation", "target"}:
            raise ValueError("position entries must contain a relation and target")
        relation, target = position["relation"], position["target"]
        if relation not in POSITION_RELATIONS or target not in by_id:
            raise ValueError("position relation or endpoint is invalid")
        kind, target_kind = by_id[ident]["type"], by_id[target]["type"]
        if ((relation == "at" and target_kind != "location")
                or (relation == "inside" and target_kind != "container")
                or (relation == "inside" and kind not in {"object", "container"})
                or (relation == "held_by" and (kind != "object" or target_kind != "agent"))):
            raise ValueError("position relation has incompatible entity types")
        if ident == target:
            raise ValueError("an entity cannot be positioned inside itself")
    for object_id, owner_id in owners.items():
        if (object_id not in by_id or by_id[object_id]["type"] != "object"
                or owner_id not in by_id or by_id[owner_id]["type"] != "agent"):
            raise ValueError("ownership endpoints are invalid")
    opened = state.get("open_containers")
    if (not isinstance(opened, list) or len(opened) != len(set(opened))
            or any(item not in by_id or by_id[item]["type"] != "container" for item in opened)):
        raise ValueError("open-container state is invalid")
    violations = constraint_violations(world, state)
    if violations:
        raise ValueError("state violates constraints: " + ", ".join(violations))
    return []


def effective_location(world: dict[str, Any], state: dict[str, Any], ident: str) -> str:
    by_id = entity_index(world)
    seen = set()
    current = ident
    while current in state["positions"]:
        if current in seen:
            raise ValueError("position cycle detected")
        seen.add(current)
        position = state["positions"][current]
        relation, target = position["relation"], position["target"]
        if relation == "at":
            return target
        current = target
        if relation == "held_by" and by_id[current]["type"] != "agent":
            raise ValueError("held object target must be an agent")
    if by_id.get(current, {}).get("type") == "location":
        return current
    raise ValueError("entity does not resolve to a location")


def constraint_violations(world: dict[str, Any], state: dict[str, Any]) -> list[str]:
    by_id = entity_index(world)
    positions = state.get("positions", {})
    violations = []
    for ident, position in positions.items():
        if position.get("relation") == "inside":
            parent = position.get("target")
            if parent not in by_id or by_id[parent]["type"] != "container":
                violations.append("container_only_contains_objects_or_containers")
                continue
            seen = {ident}
            current = parent
            while current in positions and positions[current].get("relation") == "inside":
                if current in seen:
                    violations.append("containment_acyclic")
                    break
                seen.add(current)
                current = positions[current]["target"]
        try:
            effective_location(world, state, ident)
        except (KeyError, ValueError):
            violations.append("all_entities_resolve_to_location")
    for container_id, container in by_id.items():
        if container["type"] != "container":
            continue
        contents = sum(position.get("relation") == "inside"
                       and position.get("target") == container_id
                       for position in positions.values())
        if contents > int(container["properties"].get("capacity", 0)):
            violations.append("container_capacity")
    held_count: dict[str, int] = {}
    for position in positions.values():
        if position.get("relation") == "held_by":
            held_count[position["target"]] = held_count.get(position["target"], 0) + 1
    for agent, count in held_count.items():
        if count > int(by_id[agent]["properties"].get("carry_capacity", 0)):
            violations.append("agent_carry_capacity")
    return sorted(set(violations))
