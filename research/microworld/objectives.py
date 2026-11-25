"""Configurable dense token-level structural heads and normalized loss groups.

This module defines training objectives but intentionally does not launch training.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from research.foundation.model import Foundation


TASKS = (
    "entity_span", "entity_identity", "entity_type",
    "property_span", "property_key", "property_value",
    "relationship_span", "relationship_type", "relationship_source",
    "relationship_target", "negation", "temporal_reference",
    "action_span", "action_type", "action_validity", "precondition",
    "state_change_span", "state_change_entity", "invalid_action",
    "violated_constraint",
)
LOSS_GROUPS = {
    "entity": ("entity_span", "entity_identity", "entity_type"),
    "property": ("property_span", "property_key", "property_value"),
    "relationship": ("relationship_span", "relationship_type",
                     "relationship_source", "relationship_target"),
    "negation": ("negation",),
    "temporal": ("temporal_reference",),
    "action": ("action_span", "action_type", "action_validity"),
    "precondition": ("precondition",),
    "state_transition": ("state_change_span", "state_change_entity"),
    "invalid_action": ("invalid_action", "violated_constraint"),
}
PRECONDITION_CLASSES = {
    "linked rooms": "room_adjacency",
    "portable, unheld, nearby": "pickup_eligibility",
    "spare capacity": "available_capacity",
    "open nearby container": "open_nearby_container",
    "closed nearby container": "closed_nearby_container",
    "nearby agents together": "agents_together",
}


def _objective_label(annotation: dict[str, Any]) -> str:
    if annotation["type"] == "precondition":
        phrase = annotation.get("precondition", annotation["text"])
        return PRECONDITION_CLASSES.get(phrase, phrase)
    if annotation["type"] == "negation":
        return "negation_marker"
    if annotation["type"] == "temporal_reference":
        return "temporal_marker"
    return str(annotation["label"])


def label_vocabulary(annotations: list[dict[str, Any]]) -> dict[str, list[str]]:
    labels = {task: {"O"} for task in TASKS}
    span_mapping = {
        "entity_mention": "entity_span", "property_mention": "property_span",
        "property_value": "property_value", "relationship_phrase": "relationship_span",
        "negation": "negation", "temporal_reference": "temporal_reference",
        "action_mention": "action_span", "precondition": "precondition",
        "state_change": "state_change_span", "invalid_action": "invalid_action",
        "violated_constraint": "violated_constraint",
    }
    direct_mapping = {
        "entity_identity": "entity_identity", "entity_type": "entity_type",
        "property_key": "property_key", "relationship_type": "relationship_type",
        "relationship_source": "relationship_source",
        "relationship_target": "relationship_target",
        "action_type": "action_type", "action_validity": "action_validity",
        "state_change_entity": "state_change_entity",
    }
    for annotation in annotations:
        kind, label = annotation["type"], _objective_label(annotation)
        if kind in span_mapping:
            task = span_mapping[kind]
            labels[task].add(f"B:{label}")
            labels[task].add(f"I:{label}")
        if kind in direct_mapping:
            labels[direct_mapping[kind]].add(label)
    return {task: ["O"] + sorted(values - {"O"}) for task, values in labels.items()}


def token_labels(aligned: dict[str, Any],
                 vocabularies: dict[str, list[str]]) -> dict[str, list[int]]:
    token_count = len(aligned["token_ids"])
    labels = {task: [-100] * token_count for task in TASKS}
    span_mapping = {
        "entity_mention": "entity_span", "property_mention": "property_span",
        "property_value": "property_value", "relationship_phrase": "relationship_span",
        "negation": "negation", "temporal_reference": "temporal_reference",
        "action_mention": "action_span", "precondition": "precondition",
        "state_change": "state_change_span", "invalid_action": "invalid_action",
        "violated_constraint": "violated_constraint",
    }
    direct_mapping = {
        "entity_identity": "entity_identity", "entity_type": "entity_type",
        "property_key": "property_key", "relationship_type": "relationship_type",
        "relationship_source": "relationship_source",
        "relationship_target": "relationship_target",
        "action_type": "action_type", "action_validity": "action_validity",
        "state_change_entity": "state_change_entity",
    }
    indices = {task: {label: index for index, label in enumerate(values)}
               for task, values in vocabularies.items()}
    for annotation in aligned["annotations"]:
        start, end = annotation["token_start"], annotation["token_end"]
        label = _objective_label(annotation)
        if annotation["type"] in span_mapping:
            task = span_mapping[annotation["type"]]
            begin_label = f"B:{label}"
            inside_label = f"I:{label}"
            for position in range(start, end):
                value = indices[task][begin_label if position == start else inside_label]
                prior = labels[task][position]
                if prior not in {-100, value}:
                    raise ValueError(f"overlapping spans conflict for task {task}")
                labels[task][position] = value
        if annotation["type"] in direct_mapping:
            task = direct_mapping[annotation["type"]]
            value = indices[task][label]
            for position in range(start, end):
                # Nested same-channel labels are invalid in this schema; preserve explicit checks.
                prior = labels[task][position]
                if prior not in {-100, value}:
                    raise ValueError(f"conflicting token labels for task {task}")
                labels[task][position] = value

    for task in TASKS:
        if task in {"entity_span", "property_span", "property_value", "relationship_span",
                    "negation", "temporal_reference", "action_span", "precondition",
                    "state_change_span", "invalid_action", "violated_constraint"}:
            outside = indices[task]["O"]
            labels[task] = [outside if value == -100 else value for value in labels[task]]
    return labels


class DenseStructuralHeads(nn.Module):
    def __init__(self, width: int, vocabularies: dict[str, list[str]]):
        super().__init__()
        if set(vocabularies) != set(TASKS):
            raise ValueError("a class vocabulary is required for each structural task")
        self.vocabularies = {key: list(value) for key, value in vocabularies.items()}
        self.projection = nn.Sequential(nn.Linear(width, width), nn.GELU())
        self.heads = nn.ModuleDict({
            task: nn.Linear(width, len(vocabularies[task])) for task in TASKS
        })

    @property
    def params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        value = self.projection(hidden)
        return {task: head(value) for task, head in self.heads.items()}


class FoundationWithDenseHeads(nn.Module):
    def __init__(self, config: dict[str, Any], vocabularies: dict[str, list[str]]):
        super().__init__()
        self.backbone = Foundation(config)
        self.structural = DenseStructuralHeads(int(config["d_model"]), vocabularies)
        self.context = self.backbone.context
        self.vocab = self.backbone.vocab

    @property
    def backbone_params(self) -> int:
        return self.backbone.params

    @property
    def auxiliary_params(self) -> int:
        return self.structural.params

    @property
    def params(self) -> int:
        return self.backbone_params + self.auxiliary_params

    def forward(self, tokens: torch.Tensor,
                include_language: bool = True) -> dict[str, Any]:
        hidden = self.backbone.hidden(tokens)
        return {"language_logits": (F.linear(hidden, self.backbone.head)
                                    if include_language else None),
                "structural_logits": self.structural(hidden)}


def language_only_loss(backbone: Foundation, tokens: torch.Tensor) -> torch.Tensor:
    """Ordinary next-token objective without allocating auxiliary parameters."""
    if tokens.ndim != 2 or tokens.shape[1] < 2:
        raise ValueError("language input must contain at least two tokens")
    logits = backbone(tokens)
    return F.cross_entropy(logits[:, :-1].reshape(-1, backbone.vocab),
                           tokens[:, 1:].reshape(-1), reduction="mean")


def _normalized_task_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape[:-1] != targets.shape:
        raise ValueError("structural logits and target shape differ")
    flat_target = targets.reshape(-1)
    active = flat_target != -100
    if not bool(torch.any(active)):
        return logits.sum() * 0.0
    values = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), flat_target,
                             ignore_index=-100, reduction="none")
    active_values = values[active]
    active_targets = flat_target[active]
    class_means = []
    for class_id in torch.unique(active_targets):
        selected = active_targets == class_id
        class_means.append(active_values[selected].mean())
    return torch.stack(class_means).mean()


def objective_losses(model: FoundationWithDenseHeads, tokens: torch.Tensor,
                     labels: dict[str, torch.Tensor],
                     weights: dict[str, float] | None = None,
                     task_weights: dict[str, float] | None = None,
                     include_language: bool = True) -> dict[str, Any]:
    """Return separately auditable language, task, and normalized group losses."""
    if tokens.ndim != 2 or tokens.shape[1] < 2:
        raise ValueError("objective input must contain at least two tokens")
    prediction = model(tokens, include_language=include_language)
    task_losses = {}
    for task in TASKS:
        if task not in labels:
            raise ValueError(f"missing dense targets for {task}")
        task_losses[task] = _normalized_task_loss(
            prediction["structural_logits"][task], labels[task])
    task_weight_values = dict(task_weights or {task: 1.0 for task in TASKS})
    unknown_tasks = set(task_weight_values) - set(TASKS)
    if unknown_tasks:
        raise ValueError(f"unknown structural tasks: {sorted(unknown_tasks)}")
    if any(float(weight) < 0 for weight in task_weight_values.values()):
        raise ValueError("task loss weights must be nonnegative")
    group_losses = {}
    for group, members in LOSS_GROUPS.items():
        active = [(task_losses[task] * float(task_weight_values.get(task, 1.0)))
                  for task in members
                  if float(task_weight_values.get(task, 1.0)) > 0
                  and bool(torch.any(labels[task] != -100))]
        group_losses[group] = (torch.stack(active).mean() if active
                               else task_losses[members[0]] * 0.0)

    language_values = (F.cross_entropy(
        prediction["language_logits"][:, :-1].reshape(-1, model.vocab),
        tokens[:, 1:].reshape(-1), reduction="mean") if include_language
        else task_losses[TASKS[0]] * 0.0)
    weight_values = dict(weights or {group: 1.0 for group in group_losses})
    unknown = set(weight_values) - set(group_losses)
    if unknown:
        raise ValueError(f"unknown structural loss groups: {sorted(unknown)}")
    if any(float(weight) < 0 for weight in weight_values.values()):
        raise ValueError("group loss weights must be nonnegative")
    weighted_groups = [(group_losses[group] * float(weight))
                       for group, weight in weight_values.items() if float(weight) > 0]
    structural_total = (torch.stack(weighted_groups).mean() if weighted_groups
                        else task_losses[TASKS[0]] * 0.0)
    total = structural_total + language_values if include_language else structural_total
    return {"total": total, "language": language_values,
            "task_losses": task_losses, "group_losses": group_losses,
            "structural": structural_total,
            "backbone_parameters": model.backbone_params,
            "auxiliary_parameters": model.auxiliary_params,
            "total_parameters": model.params}


def arm_objectives() -> dict[str, dict[str, Any]]:
    return {
        "A": {"language": True, "structure": False,
              "description": "ordinary next-token loss on shared prompt+answer presentations"},
        "B": {"language": False, "structure": True,
              "description": "dense token-aligned structural losses only"},
        "C": {"language": True, "structure": True,
              "description": "same language targets plus dense structural losses"},
    }
