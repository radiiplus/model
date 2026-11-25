"""Manually authored surface forms grounded in the world and oracle state."""

from __future__ import annotations

import json
from typing import Any

from research.microworld.oracle import relationships
from research.microworld.schema import entity_index


COMMON_CONSTRUCTIONS = (
    "common:declarative_svo",
    "common:inverse_relation",
    "common:compact_ledger",
    "common:pronoun_reference",
    "common:temporal_sequence",
    "common:compound_clause",
)
HELDOUT_CONSTRUCTIONS = (
    "heldout:negated_question",
    "heldout:inverted_order",
)
ALL_CONSTRUCTIONS = COMMON_CONSTRUCTIONS + HELDOUT_CONSTRUCTIONS


def _edge_sentence(edge: dict[str, str], by_id: dict[str, dict[str, Any]],
                   construction: str) -> str:
    source, target, kind = by_id[edge["source"]]["name"], by_id[edge["target"]]["name"], edge["type"]
    if construction == "common:compact_ledger":
        return f"{kind}({source}, {target})."
    if construction == "common:inverse_relation":
        if kind == "contains":
            return f"{target} is inside {source}."
        if kind == "owns":
            return f"{target} belongs to {source}."
        if kind == "located_at":
            return f"{source} is at {target}."
        if kind == "holds":
            return f"{target} is held by {source}."
        return f"{source} connects to {target}."
    if construction == "heldout:inverted_order":
        if kind == "contains":
            return f"Inside {source} is {target}."
        if kind == "owns":
            return f"To {source} belongs {target}."
        if kind == "located_at":
            return f"At {target} is {source}."
        if kind == "holds":
            return f"In the hands of {source} is {target}."
        return f"Connected to {target} is {source}."
    if kind == "contains":
        return f"{source} contains {target}."
    if kind == "owns":
        return f"{source} owns {target}."
    if kind == "located_at":
        return f"{source} is located at {target}."
    if kind == "holds":
        return f"{source} is holding {target}."
    return f"{source} is connected to {target}."


def _pronoun(entity: dict[str, Any]) -> str:
    return entity["properties"].get("pronoun", "it")


def describe(world: dict[str, Any], state: dict[str, Any], construction: str,
             episode: dict[str, Any]) -> str:
    by_id = entity_index(world)
    entities = []
    for row in world["entities"]:
        props = "; ".join(f"{key}={json.dumps(value)}"
                          for key, value in sorted(row["properties"].items()))
        slot = world["entities"].index(row)
        entities.append(f"e{slot}={row['name']} ({row['type']}; {props}).")
    facts = relationships(world, state)
    if construction == "common:pronoun_reference":
        first = next((row for row in world["entities"] if row["type"] == "agent"), None)
        pronoun_lines = []
        if first:
            pronoun_lines.append(f"{first['name']} is an agent.")
            pronoun_lines.append(f"{_pronoun(first).capitalize()} is at "
                                 f"{by_id[state['positions'][first['id']]['target']]['name']}.")
        entities.extend(pronoun_lines)
    relation_lines = [_edge_sentence(edge, by_id, construction) for edge in facts]
    if construction == "common:compound_clause":
        relation_lines = [
            " ".join(relation_lines[index:index + 2])
            for index in range(0, len(relation_lines), 2)
        ]
    state_facts = []
    for ident in sorted(state["open_containers"]):
        state_facts.append(f"{by_id[ident]['name']} is open.")
    # Properties are already stated with each entity binding above.
    for ident, values in sorted(state["owners"].items()):
        state_facts.append(f"{by_id[values]['name']} owns {by_id[ident]['name']}.")
    rules = [
        "Rules: move=linked rooms; pickup=portable, unheld, nearby, spare capacity; put/take=open nearby container; open=closed nearby container; close=open nearby container; give=nearby agents together. Held objects follow agents; gifts transfer ownership; containment is acyclic and capacity limited."
    ]
    return " ".join(entities + relation_lines + state_facts + rules)


def action_text(action: dict[str, Any], world: dict[str, Any]) -> str:
    by_id = entity_index(world)
    kind = action["type"]
    if kind == "move":
        return f"move {by_id[action['agent']]['name']} to {by_id[action['destination']]['name']}"
    if kind == "pick_up":
        return f"have {by_id[action['agent']]['name']} pick up {by_id[action['object']]['name']}"
    if kind == "put_in":
        return f"put {by_id[action['object']]['name']} into {by_id[action['container']]['name']}"
    if kind == "take_out":
        return f"take {by_id[action['object']]['name']} out of {by_id[action['container']]['name']}"
    if kind in {"open", "close"}:
        return f"{kind} {by_id[action['container']]['name']}"
    return (f"have {by_id[action['agent']]['name']} give {by_id[action['object']]['name']} "
            f"to {by_id[action['recipient']]['name']}")


def question_text(query: dict[str, Any], world: dict[str, Any], construction: str,
                  episode: dict[str, Any]) -> str:
    by_id = entity_index(world)
    kind = query["type"]
    if kind == "entities":
        question = "Which entities exist in this world?"
    elif kind == "properties":
        question = f"What properties does {by_id[query['entity']]['name']} have?"
    elif kind == "relationships":
        question = "Which typed relationships hold in the current state?"
    elif kind == "location":
        question = f"Where is {by_id[query['entity']]['name']}?"
    elif kind == "action_validity":
        action = episode["actions"][-1] if episode["actions"] else None
        question = ("Is the listed action valid, and what precondition explains the result?"
                    if action else "Is doing nothing valid?")
    elif kind == "transition":
        question = "What exact state follows after applying the listed action sequence?"
    elif kind == "constraints":
        question = "Which world constraints are violated after the listed actions?"
    else:
        question = "Would the alternative action change the result?"

    if construction == "heldout:negated_question":
        if kind == "action_validity":
            question = "Is it false that the final action is invalid? Explain the precondition."
        else:
            question = "Which listed fact is not absent from the world? " + question
    elif construction == "common:pronoun_reference" and kind == "location":
        question = f"Where is it, referring to {by_id[query['entity']]['name']}?"
    elif construction == "common:temporal_sequence":
        question = "At the relevant time, " + question[0].lower() + question[1:]
    return question


def render(world: dict[str, Any], episode: dict[str, Any], query: dict[str, Any],
           construction: str) -> tuple[str, str]:
    state = world["initial_state"]
    description = describe(world, state, construction, episode)
    actions = [action_text(action, world) for action in episode["actions"]]
    counterfactual_line = ""
    if episode.get("kind") == "counterfactual":
        alternative = [action_text(action, world)
                       for action in episode["counterfactual_actions"]]
        counterfactual_line = (" Counterfactually, instead " + "; then ".join(alternative) + "."
                               if alternative else " Counterfactually, make no change instead.")
    action_line = ("Before acting, " + "; then ".join(actions) + "."
                   if actions else "No action is applied.")
    if construction == "common:temporal_sequence":
        action_line = ("Initially the ledger is as stated; next, " + "; then ".join(actions) + "."
                       if actions else "Initially the ledger is as stated; no action follows.")
    action_line += counterfactual_line
    question = question_text(query, world, construction, episode)
    prompt = f"{description} {action_line} Question: {question}"
    if construction == "common:compound_clause":
        prompt = prompt.replace(" Question:", " As one compound account, answer:")
    return prompt, f"{construction}|{query['type']}"


def answer_text(answer: dict[str, Any], query: dict[str, Any], world: dict[str, Any],
                construction: str) -> str:
    by_id = entity_index(world)
    prefix = ("Ground truth under the held-out wording: "
              if construction.startswith("heldout:") else "Oracle result: ")
    if query["type"] == "action_validity":
        if answer["valid"]:
            body = "the action is valid and its preconditions are satisfied."
            return prefix + body
        reason = answer["reason"]
        return prefix + f"the action is invalid because {reason['message']} Code: {reason['code']}."
    if query["type"] == "constraints":
        body = ("no constraints are violated." if not answer["violations"]
                else "violated constraints are " + ", ".join(answer["violations"]) + ".")
        return prefix + body
    if query["type"] == "location":
        pos = answer["position"]
        if pos["relation"] == "held_by":
            body = f"{by_id[answer['entity']]['name']} is held by {by_id[pos['target']]['name']}."
        elif pos["relation"] == "inside":
            body = f"{by_id[answer['entity']]['name']} is inside {by_id[pos['target']]['name']}."
        else:
            body = f"{by_id[answer['entity']]['name']} is at {by_id[answer['effective_location']]['name']}."
        return prefix + body
    if query["type"] == "entities":
        slots = {entity["id"]: f"e{index}" for index, entity in enumerate(world["entities"])}
        entities = [f"{slots[row['id']]}:{row['type']}"
                    for row in answer["entities"]]
        return prefix + "Entities: " + ", ".join(entities) + "."
    if query["type"] == "properties":
        properties = answer["properties"]
        values = [f"{key} is {json.dumps(value)}" for key, value in sorted(properties.items())]
        return prefix + f"{by_id[answer['entity']]['name']}: " + "; ".join(values) + "."
    if query["type"] == "relationships":
        slots = {row["id"]: f"e{index}" for index, row in enumerate(world["entities"])}
        descriptions = [f"{edge['type']}({slots[edge['source']]},{slots[edge['target']]})"
                        for edge in answer["relationships"]]
        return prefix + ("Relations: " + "; ".join(descriptions) + "."
                         if descriptions else "no relationships hold.")
    if query["type"] == "transition":
        before, after = answer["before"], answer["after"]
        changes = []
        for ident in sorted(before["positions"]):
            old, new = before["positions"][ident], after["positions"][ident]
            if old != new:
                changes.append(f"{by_id[ident]['name']} changes from {_position_text(old, by_id)} "
                               f"to {_position_text(new, by_id)}")
        for ident in sorted(set(before["open_containers"]) ^ set(after["open_containers"])):
            changes.append(f"{by_id[ident]['name']} becomes "
                           f"{'open' if ident in after['open_containers'] else 'closed'}")
        for ident in sorted(set(before["owners"]) | set(after["owners"])):
            if before["owners"].get(ident) != after["owners"].get(ident):
                changes.append(f"ownership of {by_id[ident]['name']} goes to "
                               f"{by_id[after['owners'][ident]]['name']}")
        body = "the state is unchanged." if not changes else "; ".join(changes) + "."
        if not answer["valid"]:
            body += " The action sequence stops because " + answer["reason"]["message"]
        return prefix + body
    if query["type"] == "counterfactual":
        changed = answer["changed_entities"]
        if not changed:
            return prefix + "the alternative actions produce the same state."
        details = []
        for ident in changed:
            details.append(f"{by_id[ident]['name']} is {_position_text(answer['primary_state']['positions'][ident], by_id)} "
                           f"after the listed action and {_position_text(answer['alternate_state']['positions'][ident], by_id)} "
                           "under the alternative")
        return prefix + "the outcomes differ: " + "; ".join(details) + "."
    return prefix + json.dumps(answer, sort_keys=True)


def _position_text(position: dict[str, str], by_id: dict[str, dict[str, Any]]) -> str:
    relation, target = position["relation"], position["target"]
    if relation == "inside":
        return f"inside {by_id[target]['name']}"
    if relation == "held_by":
        return f"held by {by_id[target]['name']}"
    return f"at {by_id[target]['name']}"
