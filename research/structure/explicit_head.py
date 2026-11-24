"""Fixed-slot structural prediction heads over the Foundation backbone."""

from __future__ import annotations

import json
import argparse
import random
import resource
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from research.foundation import trainer as foundation_trainer
from research.foundation.model import Foundation
from research.interaction.tokenizer import SPECIAL, Tokenizer
from research.prototype import trainer as guard
from research.structure import train as structural_train
from research.structure.generator import NAMES
from research.structure.schema import RELATION_TYPES


ENTITY_SLOTS = 6
RELATION_SLOTS = 9
NULL = ENTITY_SLOTS
PROPERTIES = {
    "color": ("red", "blue", "green", "silver"),
    "capacity": (2, 4, 7, 11, 16),
    "status": ("ready", "idle", "limited", "restricted"),
}
ACTION_NAMES = ("none", "remove_relationship", "add_relationship",
                "transfer_ownership", "set_property")


class StructuralHead(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(width, width), nn.GELU())
        self.entity_name = nn.Linear(width, ENTITY_SLOTS * len(NAMES))
        self.entity_property = nn.ModuleDict({
            key: nn.Linear(width, ENTITY_SLOTS * len(values))
            for key, values in PROPERTIES.items()})
        self.entity_property_after = nn.ModuleDict({
            key: nn.Linear(width, ENTITY_SLOTS * len(values))
            for key, values in PROPERTIES.items()})
        self.relation_type = nn.Linear(width, RELATION_SLOTS * (len(RELATION_TYPES) + 1))
        self.relation_source = nn.Linear(width, RELATION_SLOTS * (ENTITY_SLOTS + 1))
        self.relation_target = nn.Linear(width, RELATION_SLOTS * (ENTITY_SLOTS + 1))
        self.initial_active = nn.Linear(width, RELATION_SLOTS)
        self.result_active = nn.Linear(width, RELATION_SLOTS)
        self.action_type = nn.Linear(width, len(ACTION_NAMES))

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        value = self.projection(hidden)
        batch = hidden.shape[0]
        result = {
            "entity_name": self.entity_name(value).view(batch, ENTITY_SLOTS, len(NAMES)),
            "relation_type": self.relation_type(value).view(
                batch, RELATION_SLOTS, len(RELATION_TYPES) + 1),
            "relation_source": self.relation_source(value).view(
                batch, RELATION_SLOTS, ENTITY_SLOTS + 1),
            "relation_target": self.relation_target(value).view(
                batch, RELATION_SLOTS, ENTITY_SLOTS + 1),
            "initial_active": self.initial_active(value),
            "result_active": self.result_active(value),
            "action_type": self.action_type(value),
        }
        for name, values in PROPERTIES.items():
            result[f"property_before_{name}"] = self.entity_property[name](value).view(
                batch, ENTITY_SLOTS, len(values))
            result[f"property_after_{name}"] = self.entity_property_after[name](value).view(
                batch, ENTITY_SLOTS, len(values))
        return result


class FoundationWithStructuralHead(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.backbone = Foundation(config)
        self.structural = StructuralHead(int(config["d_model"]))
        self.context = self.backbone.context
        self.vocab = self.backbone.vocab

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.structural(self.backbone.hidden(tokens)[:, -1])


def targets(row: dict[str, Any]) -> dict[str, Any]:
    world = row["world"]
    entities = world["entities"]
    if len(entities) != ENTITY_SLOTS:
        raise ValueError("structural head expects six entity slots")
    entity_index = {item["id"]: index for index, item in enumerate(entities)}
    name_index = {name: index for index, name in enumerate(NAMES)}
    entity_names = [name_index[item["name"]] for item in entities]
    before = {key: [] for key in PROPERTIES}
    after = {key: [] for key in PROPERTIES}
    overrides = row["resulting_state"].get("property_overrides", {})
    for item in entities:
        for key, values in PROPERTIES.items():
            before[key].append(values.index(item["properties"][key]))
            value = overrides.get(item["id"], {}).get(key, item["properties"][key])
            after[key].append(values.index(value))

    relations = list(world["relationships"])
    dynamic = row["resulting_state"].get("added_relationships", {})
    relations.extend(dynamic[key] for key in sorted(dynamic))
    if len(relations) > RELATION_SLOTS:
        raise ValueError("world exceeds the declared relation-slot capacity")
    active_before = set(world["initial_state"]["active_relationships"])
    active_after = set(row["resulting_state"]["active_relationships"])
    rel_types = [RELATION_TYPES.index(item["type"]) for item in relations]
    rel_source = [entity_index[item["source"]] for item in relations]
    rel_target = [entity_index[item["target"]] for item in relations]
    initial_active = [int(item["id"] in active_before) for item in relations]
    result_active = [int(item["id"] in active_after) for item in relations]
    valid = len(relations)
    pad = RELATION_SLOTS - valid
    rel_types += [len(RELATION_TYPES)] * pad
    rel_source += [NULL] * pad
    rel_target += [NULL] * pad
    initial_active += [0] * pad
    result_active += [0] * pad
    action = row.get("action")
    action_type = ACTION_NAMES.index(action["type"] if action else "none")
    return {
        "entity_name": entity_names,
        "property_before": before,
        "property_after": after,
        "relation_type": rel_types,
        "relation_source": rel_source,
        "relation_target": rel_target,
        "initial_active": initial_active,
        "result_active": result_active,
        "relation_valid": valid,
        "action_type": action_type,
    }


def _categorical(logits: torch.Tensor, values: list[int], count: int) -> torch.Tensor:
    return F.cross_entropy(logits[:, :count].reshape(-1, logits.shape[-1]),
                           torch.tensor(values[:count], dtype=torch.long))


def structural_loss(prediction: dict[str, torch.Tensor], row: dict[str, Any]
                    ) -> tuple[torch.Tensor, dict[str, float], int]:
    label = targets(row)
    losses = {}
    losses["entity_names"] = _categorical(prediction["entity_name"], label["entity_name"],
                                           ENTITY_SLOTS)
    property_losses = []
    for key, values in PROPERTIES.items():
        property_losses.append(_categorical(prediction[f"property_before_{key}"],
            label["property_before"][key], ENTITY_SLOTS))
    losses["properties_before"] = torch.stack(property_losses).mean()
    property_after_losses = []
    for key, values in PROPERTIES.items():
        property_after_losses.append(_categorical(prediction[f"property_after_{key}"],
            label["property_after"][key], ENTITY_SLOTS))
    losses["properties_after"] = torch.stack(property_after_losses).mean()
    rel_count = label["relation_valid"]
    losses["relation_type"] = _categorical(prediction["relation_type"],
        label["relation_type"], RELATION_SLOTS)
    losses["relation_source"] = _categorical(prediction["relation_source"],
        label["relation_source"], RELATION_SLOTS)
    losses["relation_target"] = _categorical(prediction["relation_target"],
        label["relation_target"], RELATION_SLOTS)
    losses["initial_state"] = F.binary_cross_entropy_with_logits(
        prediction["initial_active"][:, :rel_count],
        torch.tensor([label["initial_active"][:rel_count]], dtype=torch.float32))
    losses["result_state"] = F.binary_cross_entropy_with_logits(
        prediction["result_active"][:, :rel_count],
        torch.tensor([label["result_active"][:rel_count]], dtype=torch.float32))
    losses["action_type"] = F.cross_entropy(prediction["action_type"],
        torch.tensor([label["action_type"]], dtype=torch.long))
    group_values = [losses[name] for name in
                    ("entity_names", "properties_before", "properties_after", "relation_type",
                     "relation_source", "relation_target", "initial_state",
                     "result_state", "action_type")]
    total = torch.stack(group_values).mean()
    counts = {"entity_names": ENTITY_SLOTS,
              "property_fields": 2 * ENTITY_SLOTS * len(PROPERTIES),
              "relation_fields": rel_count * 3, "state_bits": rel_count * 2,
              "action_fields": 1}
    return total, {key: float(value.detach()) for key, value in losses.items()}, sum(counts.values())


def _decode(prediction: dict[str, torch.Tensor]) -> dict[str, Any]:
    result = {key: value.argmax(dim=-1)[0].tolist() for key, value in prediction.items()
              if key not in {"initial_active", "result_active"}}
    result["action_type"] = int(prediction["action_type"].argmax(dim=-1)[0])
    result["initial_active"] = (prediction["initial_active"][0] >= 0).to(torch.int64).tolist()
    result["result_active"] = (prediction["result_active"][0] >= 0).to(torch.int64).tolist()
    return result


def constraint_violations(decoded: dict[str, Any], valid: int) -> int:
    violations = 0
    owners: dict[int, int] = {}
    containers: dict[int, int] = {}
    dependency: dict[int, list[int]] = {}
    for index in range(valid):
        if not decoded["result_active"][index]:
            continue
        rel_type = decoded["relation_type"][index]
        source, target = decoded["relation_source"][index], decoded["relation_target"][index]
        if source >= ENTITY_SLOTS or target >= ENTITY_SLOTS or source == target:
            violations += 1
            continue
        if rel_type >= len(RELATION_TYPES):
            violations += 1
            continue
        if rel_type == RELATION_TYPES.index("ownership"):
            owners[target] = owners.get(target, 0) + 1
        if rel_type == RELATION_TYPES.index("containment"):
            containers[target] = containers.get(target, 0) + 1
        if rel_type == RELATION_TYPES.index("dependency"):
            dependency.setdefault(source, []).append(target)
    violations += sum(max(0, count - 1) for count in owners.values())
    violations += sum(max(0, count - 1) for count in containers.values())
    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(node: int) -> bool:
        if node in visiting:
            return False
        if node in visited:
            return True
        visiting.add(node)
        if not all(visit(child) for child in dependency.get(node, [])):
            return False
        visiting.remove(node)
        visited.add(node)
        return True

    violations += int(any(not visit(node) for node in list(dependency)))
    return violations


def structural_metrics(model: FoundationWithStructuralHead, rows: list[dict[str, Any]],
                       token: Tokenizer, limit: int, seed: int) -> dict[str, Any]:
    selected = structural_train.select_evaluation_rows(rows, limit, seed)
    totals = {key: 0 for key in ("entity_name_correct", "property_correct",
        "relation_type_correct", "relation_source_correct", "relation_target_correct",
        "initial_state_correct", "result_state_correct", "action_correct",
        "exact_structure", "exact_transition", "constraint_violations", "valid_relations")}
    model.eval()
    with torch.no_grad():
        for row in selected:
            prefix = token.chat([{"role": "user", "text": row["input"]}],
                               bos=True, eos=False) + [SPECIAL["ASSISTANT"]]
            values = torch.tensor([prefix], dtype=torch.long)
            prediction = model(values)
            decoded = _decode(prediction)
            label = targets(row)
            valid = label["relation_valid"]
            totals["entity_name_correct"] += sum(a == b for a, b in
                zip(decoded["entity_name"], label["entity_name"]))
            for key in PROPERTIES:
                totals["property_correct"] += sum(a == b for a, b in zip(
                    decoded[f"property_before_{key}"], label["property_before"][key]))
                totals["property_correct"] += sum(a == b for a, b in zip(
                    decoded[f"property_after_{key}"], label["property_after"][key]))
            relation_correct = True
            for name in ("relation_type", "relation_source", "relation_target"):
                matches = sum(a == b for a, b in zip(decoded[name][:valid], label[name][:valid]))
                totals[f"{name}_correct"] += matches
                relation_correct &= matches == valid
            before_correct = all(a == b for a, b in zip(
                decoded["initial_active"][:valid], label["initial_active"][:valid]))
            result_correct = all(a == b for a, b in zip(
                decoded["result_active"][:valid], label["result_active"][:valid]))
            property_after_exact = all(decoded[f"property_after_{key}"] ==
                label["property_after"][key] for key in PROPERTIES)
            totals["initial_state_correct"] += int(before_correct)
            totals["result_state_correct"] += int(result_correct)
            action_correct = decoded["action_type"] == label["action_type"]
            totals["action_correct"] += int(action_correct)
            property_exact = all(
                decoded[f"property_before_{key}"] == label["property_before"][key]
                and decoded[f"property_after_{key}"] == label["property_after"][key]
                for key in PROPERTIES)
            totals["exact_structure"] += int(
                decoded["entity_name"] == label["entity_name"] and property_exact and relation_correct)
            totals["exact_transition"] += int(before_correct and result_correct
                                                and property_after_exact and action_correct)
            totals["constraint_violations"] += constraint_violations(decoded, valid)
            totals["valid_relations"] += valid
    count = max(1, len(selected))
    return {
        "examples": len(selected),
        "entity_identification_accuracy": totals["entity_name_correct"] / (count * ENTITY_SLOTS),
        "property_accuracy": totals["property_correct"] / (count * ENTITY_SLOTS * len(PROPERTIES) * 2),
        "relationship_type_accuracy": totals["relation_type_correct"] / max(1, totals["valid_relations"]),
        "relationship_source_accuracy": totals["relation_source_correct"] / max(1, totals["valid_relations"]),
        "relationship_target_accuracy": totals["relation_target_correct"] / max(1, totals["valid_relations"]),
        "initial_state_exact": totals["initial_state_correct"] / count,
        "result_state_exact": totals["result_state_correct"] / count,
        "action_type_accuracy": totals["action_correct"] / count,
        "exact_structure": totals["exact_structure"] / count,
        "exact_transition": totals["exact_transition"] / count,
        "constraint_violations_per_example": totals["constraint_violations"] / count,
        "selected_examples_sha256": structural_train.digest([row["example_id"] for row in selected]),
    }


def read_examples(root: Path, split: str) -> list[dict[str, Any]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    info = next(item for item in manifest["files"] if item["split"] == split)
    path = root / info["path"]
    if structural_train.sha(path) != info["sha256"]:
        raise ValueError(f"dataset hash mismatch for {split}")
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != info["examples"]:
        raise ValueError(f"dataset count mismatch for {split}")
    return rows


def run(config: dict[str, Any], initial: Path, root: Path, output: Path,
        report_path: Path, objective: str = "structure",
        stop: int | None = None, resume: Path | None = None) -> dict[str, Any]:
    if objective not in {"structure", "combined"}:
        raise ValueError("objective must be structure or combined")
    if guard.available_mib() < guard.MEMORY_GUARD_MIB:
        raise MemoryError("available memory is below the existing training guard")
    guard.configure_torch()
    seed = int(config["train"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    training = read_examples(root, "train")
    evaluations = {split: read_examples(root, split) for split in
        ("validation", "iid_test", "compositional_test", "pattern_test", "counterfactual_test")}
    token = Tokenizer.load(structural_train.TOKENIZER)
    source = torch.load(initial, map_location="cpu", weights_only=True)
    model = FoundationWithStructuralHead(config["model"])
    model.backbone.load_state_dict(source["model"])
    starting_hash = structural_train.tensor_state_hash(source["model"])
    optimizer_config = config["train"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(optimizer_config["rate"]),
                                  weight_decay=float(optimizer_config["penalty"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: foundation_trainer.rate(step, optimizer_config))
    order = structural_train.Order(len(training), seed + 1)
    steps = int(optimizer_config["steps"])
    end = steps if stop is None else stop
    if not 1 <= end <= steps:
        raise ValueError("stop must be between one and the configured update count")
    eval_limit = int(optimizer_config["eval_examples"])
    eval_seed = int(optimizer_config.get("eval_seed", 2026))
    before = {split: structural_metrics(model, rows, token, eval_limit, eval_seed)
              for split, rows in evaluations.items()}
    history = []
    start = 0
    if resume is not None:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if (saved.get("schema") != 1 or saved.get("objective") != objective
                or saved.get("config") != config or saved.get("seed") != seed
                or saved.get("initial_checkpoint_sha256") != structural_train.sha(initial)
                or saved.get("dataset_manifest_sha256") != structural_train.sha(root / "manifest.json")
                or saved.get("tokenizer_sha256") != token.digest):
            raise ValueError("resume checkpoint does not match objective, config, or inputs")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        order = structural_train.Order(len(training), seed + 1, saved["sampler"])
        random.setstate(saved["python"])
        torch.set_rng_state(saved["torch"])
        history = list(saved["history"])
        start = int(saved["step"])
        before = saved["validation_before"]
    if not 0 <= start < end:
        raise ValueError("resume step must precede the requested stopping update")
    started = time.monotonic()
    for step in range(start, end):
        index = order.take(int(optimizer_config["batch"]))[0]
        row = training[index]
        prefix = token.chat([{"role": "user", "text": row["input"]}],
                           bos=True, eos=False) + [SPECIAL["ASSISTANT"]]
        prediction = model(torch.tensor([prefix], dtype=torch.long))
        struct, components, target_elements = structural_loss(prediction, row)
        objective_loss = struct
        language_loss = None
        if objective == "combined":
            language_loss = structural_train.losses(model.backbone, [row], token)["answer"]
            objective_loss = (struct + language_loss) / 2
        lr_used = float(optimizer.param_groups[0]["lr"])
        optimizer.zero_grad(set_to_none=True)
        objective_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   float(optimizer_config["clip"]))
        optimizer.step()
        scheduler.step()
        history.append({"step": step + 1, "example_id": row["example_id"],
            "input_tokens": len(prefix), "target_elements": target_elements,
            "structural_loss": float(struct.detach()),
            "language_loss": float(language_loss.detach()) if language_loss is not None else None,
            "objective_loss": float(objective_loss.detach()), "components": components,
            "learning_rate_used": lr_used,
            "learning_rate_after": float(optimizer.param_groups[0]["lr"]),
            "gradient_norm": float(grad_norm)})
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"schema": 1, "objective": objective,
        "config": config, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "sampler": order.state(), "step": end,
        "seed": seed, "history": history, "python": random.getstate(),
        "torch": torch.get_rng_state(), "validation_before": before,
        "initial_checkpoint_sha256": structural_train.sha(initial),
        "dataset_manifest_sha256": structural_train.sha(root / "manifest.json"),
        "tokenizer_sha256": token.digest}
    foundation_trainer.save(output, checkpoint)
    after = {split: structural_metrics(model, rows, token, eval_limit, eval_seed)
             for split, rows in evaluations.items()}
    language_after = {split: structural_train.evaluate(
        model.backbone, rows, token, eval_limit, eval_seed, generate_answers=True)
        for split, rows in evaluations.items()}
    spent = time.monotonic() - started
    result = {"objective": objective, "parameters_total": sum(p.numel() for p in model.parameters()),
        "foundation_parameters": model.backbone.params,
        "auxiliary_parameters": sum(p.numel() for p in model.structural.parameters()),
        "updates": end, "resume_update": start, "examples_consumed": len(history),
        "input_tokens_seen": sum(x["input_tokens"] for x in history),
        "prompt_tokens_seen": sum(x["input_tokens"] - 1 for x in history),
        "target_elements_seen": sum(x["target_elements"] for x in history),
        "training_sequence_sha256": structural_train.digest(
            [[x["example_id"]] for x in history]),
        "starting_weights_sha256": starting_hash, "initial_checkpoint_sha256": structural_train.sha(initial),
        "dataset_manifest_sha256": structural_train.sha(root / "manifest.json"),
        "tokenizer_sha256": token.digest, "optimizer_state_restored": False,
        "training": history, "validation_before": before, "validation_after": after,
        "language_after": language_after,
        "seconds": spent, "examples_per_second": len(history) / max(spent, 0.001),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "checkpoint": str(output), "checkpoint_sha256": structural_train.sha(output)}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--objective", choices=("structure", "combined"), default="structure")
    parser.add_argument("--stop", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("choose new checkpoint and report paths")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = run(config, args.initial, args.data, args.output, args.report,
                 args.objective, args.stop, args.resume)
    print(json.dumps({key: report[key] for key in
        ("objective", "updates", "examples_consumed", "input_tokens_seen",
         "exact_structure", "seconds", "checkpoint_sha256") if key in report}, sort_keys=True))


if __name__ == "__main__":
    main()
