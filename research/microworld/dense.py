"""Oracle-derived span annotations aligned to the frozen byte-level BPE tokenizer."""

from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_left, bisect_right
from typing import Any

from research.interaction.tokenizer import Tokenizer
from research.microworld.language import _edge_sentence, action_text
from research.microworld.oracle import relationships
from research.microworld.schema import ACTION_SPECS, entity_index
from research.microworld.splits import stable_hash


ANNOTATION_TYPES = (
    "entity_mention", "entity_identity", "entity_type",
    "property_mention", "property_key", "property_value",
    "relationship_phrase", "relationship_type", "relationship_source",
    "relationship_target", "negation", "temporal_reference",
    "action_mention", "action_type", "action_validity", "precondition",
    "state_change", "state_change_entity", "invalid_action",
    "violated_constraint",
)

NEGATION_RE = re.compile(r"\b(?:no|not|never|cannot|can't|invalid|violated|without)\b", re.I)
TEMPORAL_RE = re.compile(r"\b(?:initially|before|after|next|then|current|resulting|following)\b", re.I)
PRECONDITION_CUES = {
    "move": ("linked rooms",),
    "pick_up": ("portable, unheld, nearby", "spare capacity"),
    "put_in": ("open nearby container",),
    "take_out": ("open nearby container",),
    "open": ("closed nearby container",),
    "close": ("open nearby container",),
    "give": ("nearby agents together",),
}


def _all_spans(text: str, phrase: str) -> list[tuple[int, int]]:
    if not phrase:
        return []
    result = []
    cursor = 0
    while True:
        start = text.find(phrase, cursor)
        if start < 0:
            return result
        result.append((start, start + len(phrase)))
        cursor = start + 1


def _add(annotations: list[dict[str, Any]], text: str, kind: str, phrase: str,
         label: str, **attributes: Any) -> None:
    for start, end in _all_spans(text, phrase):
        annotations.append({"type": kind, "label": str(label),
                            "char_start": start, "char_end": end,
                            "text": text[start:end], **attributes})


def _changed_entities(before: dict[str, Any], after: dict[str, Any]) -> set[str]:
    changed = {ident for ident in set(before["positions"]) | set(after["positions"])
               if before["positions"].get(ident) != after["positions"].get(ident)}
    if before.get("open_containers") != after.get("open_containers"):
        changed |= set(before.get("open_containers", ())) ^ set(after.get("open_containers", ()))
    if before.get("owners") != after.get("owners"):
        changed |= {ident for ident in set(before.get("owners", {})) | set(after.get("owners", {}))
                    if before.get("owners", {}).get(ident) != after.get("owners", {}).get(ident)}
    return changed


def _oracle_annotations(row: dict[str, Any], world: dict[str, Any]) -> list[dict[str, Any]]:
    text = row["prompt"] + "\nAnswer: " + row["answer_text"]
    by_id = entity_index(world)
    annotations: list[dict[str, Any]] = []

    for ident, entity in by_id.items():
        name = entity["name"]
        _add(annotations, text, "entity_mention", name, ident, entity_id=ident)
        _add(annotations, text, "entity_identity", name, ident, entity_id=ident)
        _add(annotations, text, "entity_type", name, entity["type"], entity_id=ident)
        type_phrase = f"{name} ({entity['type']};"
        for start, end in _all_spans(text, type_phrase):
            type_start = start + len(name) + 2
            annotations.append({"type": "entity_type", "label": entity["type"],
                "char_start": type_start, "char_end": type_start + len(entity["type"]),
                "text": entity["type"], "entity_id": ident, "mention": "copular"})

    for ident, entity in by_id.items():
        for key, value in entity["properties"].items():
            encoded = json.dumps(value, ensure_ascii=True)
            property_phrase = f"{key}={encoded}"
            descriptor = f"{entity['name']} ({entity['type']};"
            for descriptor_start, _ in _all_spans(text, descriptor):
                descriptor_end = text.find(").", descriptor_start)
                if descriptor_end < 0:
                    continue
                local_text = text[descriptor_start:descriptor_end]
                for local_start, local_end in _all_spans(local_text, property_phrase):
                    start, end = descriptor_start + local_start, descriptor_start + local_end
                    key_start = start
                    value_start = key_start + len(key) + 1
                    annotations.extend([
                        {"type": "property_mention", "label": key,
                         "char_start": key_start, "char_end": end,
                         "text": text[key_start:end], "entity_id": ident},
                        {"type": "property_key", "label": key,
                         "char_start": key_start, "char_end": key_start + len(key),
                         "text": key, "entity_id": ident},
                        {"type": "property_value", "label": str(value),
                         "char_start": value_start, "char_end": value_start + len(encoded),
                         "text": encoded, "entity_id": ident, "value": value},
                    ])

    state = row["initial_state"]
    for edge in relationships(world, state):
        phrase = _edge_sentence(edge, by_id, row["construction_id"].split("|", 1)[0])
        _add(annotations, text, "relationship_phrase", phrase, edge["type"],
             source_id=edge["source"], target_id=edge["target"])
        _add(annotations, text, "relationship_type", phrase, edge["type"],
             source_id=edge["source"], target_id=edge["target"])
        # Label relation endpoints only inside the phrase, preserving repeated names elsewhere.
        for phrase_start, phrase_end in _all_spans(text, phrase):
            source = by_id[edge["source"]]["name"]
            target = by_id[edge["target"]]["name"]
            for start, end in _all_spans(text[phrase_start:phrase_end], source):
                annotations.append({"type": "relationship_source", "label": edge["source"],
                    "char_start": phrase_start + start, "char_end": phrase_start + end,
                    "text": source, "entity_id": edge["source"],
                    "source_id": edge["source"], "target_id": edge["target"]})
            for start, end in _all_spans(text[phrase_start:phrase_end], target):
                annotations.append({"type": "relationship_target", "label": edge["target"],
                    "char_start": phrase_start + start, "char_end": phrase_start + end,
                    "text": target, "entity_id": edge["target"],
                    "source_id": edge["source"], "target_id": edge["target"]})

    if row["query_type"] == "relationships":
        slots = {entity["id"]: f"e{index}" for index, entity in enumerate(world["entities"])}
        for edge in row["ground_truth"]["relationships"]:
            phrase = f"{edge['type']}({slots[edge['source']]},{slots[edge['target']]})"
            _add(annotations, text, "relationship_phrase", phrase, edge["type"],
                 source_id=edge["source"], target_id=edge["target"], scope="answer")
            _add(annotations, text, "relationship_type", phrase, edge["type"],
                 source_id=edge["source"], target_id=edge["target"], scope="answer")
            for phrase_start, _ in _all_spans(text, phrase):
                for kind, ident, alias in (
                        ("relationship_source", edge["source"], slots[edge["source"]]),
                        ("relationship_target", edge["target"], slots[edge["target"]])):
                    for start, end in _all_spans(text[phrase_start:phrase_start + len(phrase)], alias):
                        annotations.append({"type": kind, "label": ident,
                            "char_start": phrase_start + start,
                            "char_end": phrase_start + end, "text": alias,
                            "entity_id": ident, "source_id": edge["source"],
                            "target_id": edge["target"], "scope": "answer"})

    for match in NEGATION_RE.finditer(text):
        annotations.append({"type": "negation", "label": match.group().casefold(),
                            "char_start": match.start(), "char_end": match.end(),
                            "text": match.group()})
    for match in TEMPORAL_RE.finditer(text):
        annotations.append({"type": "temporal_reference", "label": match.group().casefold(),
                            "char_start": match.start(), "char_end": match.end(),
                            "text": match.group()})

    actions = row.get("actions", [])
    if row["query_type"] == "action_validity" and actions:
        action = actions[-1]
        phrase = action_text(action, world)
        validity = "valid" if row["ground_truth"]["valid"] else "invalid"
        _add(annotations, text, "action_validity", phrase, validity,
             action_type=action["type"], reason_code=(
                 row["ground_truth"]["reason"]["code"]
                 if not row["ground_truth"]["valid"] else None))
    for action in actions:
        action_type = action["type"]
        phrase = action_text(action, world)
        _add(annotations, text, "action_mention", phrase, action_type,
             action_type=action_type, action=action)
        _add(annotations, text, "action_type", phrase, action_type, action_type=action_type)
        for cue in PRECONDITION_CUES.get(action_type, ()):
            _add(annotations, text, "precondition", cue, action_type,
                 action_type=action_type, precondition=cue)

    before, after = row["initial_state"], row["resulting_state"]
    changed = _changed_entities(before, after)
    for ident in changed:
        entity_name = by_id[ident]["name"]
        _add(annotations, text, "state_change_entity", entity_name, ident, entity_id=ident)
        for start, end in _all_spans(text, entity_name):
            sentence_start = text.rfind(".", 0, start) + 1
            sentence_end = text.find(".", end)
            sentence_end = len(text) if sentence_end < 0 else sentence_end + 1
            segment = text[sentence_start:sentence_end]
            if "changes from" in segment or "becomes " in segment or "goes to" in segment:
                annotations.append({"type": "state_change", "label": "transition",
                    "char_start": sentence_start, "char_end": sentence_end,
                    "text": text[sentence_start:sentence_end], "entity_id": ident})

    gt = row["ground_truth"]
    if row["query_type"] == "action_validity" and not gt.get("valid", True):
        _add(annotations, text, "invalid_action", "invalid", gt["reason"]["code"],
             action_type=(actions[-1]["type"] if actions else None),
             reason_code=gt["reason"]["code"])
        reason_code = gt["reason"]["code"]
        _add(annotations, text, "violated_constraint", reason_code, reason_code,
             constraint=reason_code)
        message = gt["reason"]["message"]
        _add(annotations, text, "violated_constraint", message, reason_code,
             constraint=reason_code, explanation=True)
    for constraint in gt.get("violated_constraints", []):
        _add(annotations, text, "violated_constraint", constraint, constraint,
             constraint=constraint)

    # Stable ordering preserves nested and overlapping labels without collapsing them.
    annotations.sort(key=lambda item: (item["char_start"], item["char_end"], item["type"],
                                       item["label"]))
    return annotations


def align_annotations(row: dict[str, Any], world: dict[str, Any],
                      tokenizer: Tokenizer) -> dict[str, Any]:
    source = row["prompt"] + "\nAnswer: " + row["answer_text"]
    encoding = tokenizer.encode_with_offsets(source)
    char_to_byte = [0]
    for character in source:
        char_to_byte.append(char_to_byte[-1] + len(character.encode("utf-8")))
    byte_ends = [right for _, right in encoding["byte_offsets"]]
    byte_starts = [left for left, _ in encoding["byte_offsets"]]
    aligned = []
    for annotation in _oracle_annotations(row, world):
        start, end = annotation["char_start"], annotation["char_end"]
        if not 0 <= start < end <= len(source):
            raise ValueError("annotation character span is outside source text")
        byte_start, byte_end = char_to_byte[start], char_to_byte[end]
        first = bisect_right(byte_ends, byte_start)
        last = bisect_left(byte_starts, byte_end)
        if first >= last:
            raise ValueError("annotation did not align to a tokenizer token")
        aligned.append({**annotation, "byte_start": byte_start, "byte_end": byte_end,
                        "token_start": first, "token_end": last,
                        "token_indices": list(range(first, last))})
    return {
        "example_id": row["example_id"], "world_id": row["world_id"],
        "episode_id": row["episode_id"], "split": row["split"],
        "source": source, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "tokenizer_digest": tokenizer.digest, "token_ids": encoding["ids"],
        "token_byte_offsets": encoding["byte_offsets"],
        "token_char_offsets": encoding["char_offsets"],
        "annotations": aligned,
        "annotation_digest": stable_hash(aligned),
    }


def audit_annotations(row: dict[str, Any], world: dict[str, Any],
                      aligned: dict[str, Any], tokenizer: Tokenizer) -> dict[str, Any]:
    if aligned["source"] != row["prompt"] + "\nAnswer: " + row["answer_text"]:
        raise ValueError("annotation source differs from example text")
    if aligned["tokenizer_digest"] != tokenizer.digest:
        raise ValueError("annotation tokenizer digest differs from active tokenizer")
    if tokenizer.decode(aligned["token_ids"]) != aligned["source"]:
        raise ValueError("token-aligned source does not round-trip")
    token_offsets = tokenizer.encode_with_offsets(aligned["source"])
    if (token_offsets["ids"] != aligned["token_ids"]
            or token_offsets["byte_offsets"] != [tuple(span)
                                                  for span in aligned["token_byte_offsets"]]
            or token_offsets["char_offsets"] != [tuple(span)
                                                  for span in aligned["token_char_offsets"]]):
        raise ValueError("stored tokenizer offsets differ from a fresh encoding")
    byte_starts = [left for left, _ in aligned["token_byte_offsets"]]
    byte_ends = [right for _, right in aligned["token_byte_offsets"]]
    by_id = entity_index(world)
    changed_entities = _changed_entities(row["initial_state"], row["resulting_state"])
    for item in aligned["annotations"]:
        start, end = item["char_start"], item["char_end"]
        if aligned["source"][start:end] != item["text"]:
            raise ValueError("annotation text does not match character offsets")
        if not 0 <= item["token_start"] < item["token_end"] <= len(aligned["token_ids"]):
            raise ValueError("annotation token span is outside encoded input")
        first = bisect_right(byte_ends, item["byte_start"])
        last = bisect_left(byte_starts, item["byte_end"])
        covered = list(range(first, last))
        if covered != item["token_indices"] or not covered:
            raise ValueError("annotation token indices do not cover its UTF-8 byte span")
        if (covered[0] != item["token_start"]
                or covered[-1] + 1 != item["token_end"]):
            raise ValueError("annotation token bounds disagree with token indices")
        if item["type"] in {"entity_mention", "entity_identity", "entity_type",
                            "relationship_source", "relationship_target",
                            "state_change_entity"} and item.get("entity_id") not in by_id:
            raise ValueError("entity annotation references an unknown oracle entity")
        if item["type"] == "entity_type":
            if item["label"] != by_id[item["entity_id"]]["type"]:
                raise ValueError("entity type annotation differs from world schema")
        if item["type"] in {"property_mention", "property_key"}:
            entity = by_id.get(item.get("entity_id"))
            if entity is None or item["label"] not in entity["properties"]:
                raise ValueError("property key annotation differs from world schema")
        if item["type"] == "property_value":
            entity = by_id.get(item.get("entity_id"))
            if (entity is None or item.get("value") not in entity["properties"].values()
                    or item["label"] != str(item["value"])):
                raise ValueError("property value annotation differs from world schema")
        if item["type"] == "relationship_phrase":
            state = (row["initial_state"] if item.get("scope") != "answer" else
                     row["oracle_state"])
            exists = any(edge["type"] == item["label"]
                         and edge["source"] == item["source_id"]
                         and edge["target"] == item["target_id"]
                         for edge in relationships(world, state))
            if not exists:
                raise ValueError("relationship label is not present in the oracle state")
        if item["type"] == "state_change":
            if item["entity_id"] not in changed_entities:
                raise ValueError("state-change annotation lacks an oracle state difference")
        if item["type"] == "state_change_entity" and item["entity_id"] not in changed_entities:
            raise ValueError("state-change entity is invariant in the oracle states")
        if item["type"] in {"action_mention", "action_type"}:
            if (item.get("action_type") not in ACTION_SPECS
                    or item["action_type"] not in {action["type"] for action in row.get("actions", [])}):
                raise ValueError("action annotation is not present in the oracle episode")
        if item["type"] == "action_validity":
            expected = "valid" if row["ground_truth"].get("valid", True) else "invalid"
            if row["query_type"] != "action_validity" or item["label"] != expected:
                raise ValueError("action-validity label differs from oracle result")
        if item["type"] == "precondition":
            if item.get("action_type") not in ACTION_SPECS:
                raise ValueError("precondition label references an unknown action specification")
        if item["type"] == "invalid_action":
            if row["ground_truth"].get("valid", True):
                raise ValueError("valid action received an invalid-action annotation")
            if item["reason_code"] != row["ground_truth"]["reason"]["code"]:
                raise ValueError("invalid-action reason differs from oracle")
    return {"annotations": len(aligned["annotations"]),
            "token_count": len(aligned["token_ids"]),
            "annotation_types": sorted({item["type"] for item in aligned["annotations"]})}
