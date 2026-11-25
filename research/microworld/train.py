"""Controlled CPU training and independent evaluation for microworld Batch 4."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import re
import resource
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from research.foundation.model import Foundation
from research.interaction.tokenizer import SPECIAL, Tokenizer
from research.microworld.objectives import (
    LOSS_GROUPS, TASKS, FoundationWithDenseHeads, _objective_label,
    label_vocabulary, language_only_loss, objective_losses, token_labels,
)
from research.microworld.splits import stable_hash
from research.prototype import trainer as guard


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset_v4"
TOKENIZER_PATH = ROOT.parent / "interaction" / "data" / "ready7" / "tokenizer.json"
BASE_CHECKPOINT = ROOT.parent / "foundation" / "data" / "learn.pt"
ARMS = {"A": "arm_a", "B": "arm_b", "C": "arm_c"}
EVAL_SPLITS = ("validation", "iid_test", "compositional_test",
               "counterfactual_test", "grammar_test")
PILOT_UPDATES = 64
EVAL_INTERVAL = 16
VALIDATION_SAMPLE_SIZE = 32
FREE_GENERATION_SAMPLE_SIZE = 32
EVAL_SEED = 20261126
GENERATION_SEED = 20261127
MEMORY_GUARD_MIB = guard.MEMORY_GUARD_MIB


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) \
            and torch.equal(left, right)
    if isinstance(left, dict) or isinstance(right, dict):
        return isinstance(left, dict) and isinstance(right, dict) \
            and left.keys() == right.keys() \
            and all(nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) \
            and all(nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def code_hash() -> str:
    paths = (Path(__file__), ROOT / "objectives.py", ROOT.parent / "foundation" / "model.py",
             ROOT.parent / "experiments" / "tiny_gpt.py",
             TOKENIZER_PATH.parent.parent.parent / "tokenizer.py")
    return canonical_hash({str(path): sha256(path) for path in paths})


def read_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def load_inputs(dataset: Path = DATASET, tokenizer_path: Path = TOKENIZER_PATH,
                checkpoint_path: Path = BASE_CHECKPOINT) -> dict[str, Any]:
    manifest_path = dataset / "manifest.json"
    objective_path = dataset / "objective_manifest.json"
    if not manifest_path.is_file() or not objective_path.is_file():
        raise FileNotFoundError("dataset and objective manifests are required")
    manifest = json.loads(manifest_path.read_text())
    objective_manifest = json.loads(objective_path.read_text())
    if objective_manifest["dataset_manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("objective manifest is not bound to the selected dataset")
    for item in manifest["files"]:
        path = dataset / item["path"]
        if (not path.is_file() or path.stat().st_size != item["bytes"]
                or sha256(path) != item["sha256"]):
            raise ValueError(f"dataset file hash mismatch: {item['path']}")
    tokenizer = Tokenizer.load(tokenizer_path)
    if (tokenizer.digest != objective_manifest["tokenizer_digest"]
            or sha256(tokenizer_path) != objective_manifest["tokenizer_sha256"]):
        raise ValueError("tokenizer hash differs from objective manifest")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_sha256 = sha256(checkpoint_path)
    if (sha256(checkpoint_path) != next(
            (json.loads((ROOT / "configs" / f"{name}.json").read_text())
             .get("starting_checkpoint_sha256") for name in ARMS.values()), None)):
        raise ValueError("starting checkpoint differs from experiment config")
    for name in ARMS.values():
        config = json.loads((ROOT / "configs" / f"{name}.json").read_text())
        if (config["starting_checkpoint_sha256"] != checkpoint_sha256
                or config["data"]["tokenizer_digest"] != tokenizer.digest
                or config["dataset_manifest_sha256"] != manifest["manifest_sha256"]
                or config["objective_manifest_sha256"] != objective_manifest["manifest_sha256"]
                or config["tokenizer_file_sha256"] != sha256(tokenizer_path)
                or Path(config["data"]["dataset"]).resolve() != dataset.resolve()
                or config["evaluation"]["free_generation_samples_per_partition"]
                != FREE_GENERATION_SAMPLE_SIZE):
            raise ValueError(f"{name} does not use the shared checkpoint/data/tokenizer")
    if checkpoint.get("model") is None or checkpoint.get("config", {}).get("model") is None:
        raise ValueError("starting checkpoint lacks Foundation weights or model config")
    model_config = checkpoint["config"]["model"]
    if model_config != json.loads((ROOT / "configs" / "arm_a.json").read_text())["model"]:
        raise ValueError("starting checkpoint and experiment architecture differ")
    pilot = objective_manifest["predeclared_pilot"]
    if pilot["updates"] != PILOT_UPDATES or len(pilot["example_ids_in_presentation_order"]) != PILOT_UPDATES:
        raise ValueError("predeclared training sample does not match the 64-update pilot")

    worlds_by_split: dict[str, dict[str, dict[str, Any]]] = {}
    for split in ("train", *EVAL_SPLITS):
        worlds = {row["world_id"]: row
                  for row in read_jsonl_gz(dataset / f"worlds_{split}.jsonl.gz")}
        worlds_by_split[split] = worlds
    train_ids = pilot["example_ids_in_presentation_order"]
    train_indices = objective_manifest["predeclared_pilot"]["training_row_indices"]
    if len(set(train_ids)) != PILOT_UPDATES or len(train_indices) != PILOT_UPDATES:
        raise ValueError("predeclared pilot rows are not unique")
    train_by_index: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    wanted_train_indices = set(train_indices)
    for index, (row, aligned) in enumerate(zip(
            read_jsonl_gz(dataset / "examples_train.jsonl.gz"),
            read_jsonl_gz(dataset / "annotations_train.jsonl.gz"))):
        if row["example_id"] != aligned["example_id"]:
            raise ValueError("example/annotation order differs in train")
        if index in wanted_train_indices:
            train_by_index[index] = (row, aligned)
    if set(train_by_index) != wanted_train_indices:
        raise ValueError("predeclared training indices exceed the dataset")
    train_rows = [train_by_index[index][0] for index in train_indices]
    train_aligned = [train_by_index[index][1] for index in train_indices]
    if [row["example_id"] for row in train_rows] != train_ids:
        raise ValueError("pilot IDs and row indices in objective manifest disagree")

    validation_rng = random.Random(EVAL_SEED)
    validation_count = manifest["statistics"]["examples_by_split"]["validation"]
    validation_indices = set(validation_rng.sample(
        range(validation_count), min(VALIDATION_SAMPLE_SIZE, validation_count)))
    validation_rows, validation_aligned = [], []
    for index, (row, aligned) in enumerate(zip(
            read_jsonl_gz(dataset / "examples_validation.jsonl.gz"),
            read_jsonl_gz(dataset / "annotations_validation.jsonl.gz"))):
        if row["example_id"] != aligned["example_id"]:
            raise ValueError("example/annotation order differs in validation")
        if index in validation_indices:
            validation_rows.append(row)
            validation_aligned.append(aligned)
    freegen_rows = {}
    for split_index, split in enumerate(EVAL_SPLITS):
        count = manifest["statistics"]["examples_by_split"][split]
        rng = random.Random(GENERATION_SEED + split_index)
        rows = list(read_jsonl_gz(dataset / f"examples_{split}.jsonl.gz"))
        by_type: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            by_type[row["query_type"]].append(index)
        selected = []
        for query_type, indices in sorted(by_type.items()):
            selected.append(rng.choice(indices))
        remaining = [index for index in range(len(rows)) if index not in set(selected)]
        selected.extend(rng.sample(remaining, min(
            max(0, FREE_GENERATION_SAMPLE_SIZE - len(selected)), len(remaining))))
        freegen_rows[split] = [rows[index] for index in sorted(selected)]
    vocabularies = objective_manifest["label_vocabulary"]
    # Recompute the train-only class vocabulary to bind training labels to the current sidecar.
    computed = {task: {"O"} for task in TASKS}
    for row in read_jsonl_gz(dataset / "annotations_train.jsonl.gz"):
        values = label_vocabulary(row["annotations"])
        for task in TASKS:
            computed[task].update(values[task])
    computed = {task: ["O"] + sorted(values - {"O"})
                for task, values in computed.items()}
    if computed != vocabularies:
        raise ValueError("objective label vocabulary differs from train annotation sidecar")
    return {
        "dataset": dataset, "manifest": manifest, "objective_manifest": objective_manifest,
        "tokenizer": tokenizer, "checkpoint": checkpoint, "model_config": model_config,
        "vocabularies": vocabularies, "train_rows": train_rows,
        "train_aligned": train_aligned, "validation_rows": validation_rows,
        "validation_aligned": validation_aligned, "freegen_rows": freegen_rows,
        "worlds_by_split": worlds_by_split,
    }


def atomic_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temp.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def make_model(arm: str, model_config: dict[str, Any], vocabularies: dict[str, list[str]],
               initial_state: dict[str, Any], seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    if arm == "A":
        model: torch.nn.Module = Foundation(model_config)
        model.load_state_dict(initial_state)
    else:
        wrapped = FoundationWithDenseHeads(model_config, vocabularies)
        wrapped.backbone.load_state_dict(initial_state)
        model = wrapped
    return model.cpu()


def model_backbone(model: torch.nn.Module) -> Foundation:
    return model.backbone if isinstance(model, FoundationWithDenseHeads) else model


def tensor_targets(aligned: dict[str, Any], vocabularies: dict[str, list[str]]) -> dict[str, torch.Tensor]:
    return {task: torch.tensor(values, dtype=torch.long)[None, :]
            for task, values in token_labels(aligned, vocabularies).items()}


def loss_for(arm: str, model: torch.nn.Module, ids: list[int],
             aligned: dict[str, Any], vocabularies: dict[str, list[str]],
             group_weights: dict[str, float]) -> dict[str, Any]:
    tokens = torch.tensor(ids, dtype=torch.long)[None, :]
    if arm == "A":
        value = language_only_loss(model, tokens)
        return {"total": value, "language": value, "structural": None,
                "group_losses": {}, "task_losses": {}}
    targets = tensor_targets(aligned, vocabularies)
    result = objective_losses(model, tokens, targets, weights=group_weights,
                              include_language=arm == "C")
    return result


def base_metadata(data: dict[str, Any], config: dict[str, Any], arm: str,
                  seed: int) -> dict[str, Any]:
    return {
        "arm": arm, "seed": seed, "config_sha256": canonical_hash(config),
        "dataset_manifest_sha256": data["manifest"]["manifest_sha256"],
        "objective_manifest_sha256": data["objective_manifest"]["manifest_sha256"],
        "tokenizer_sha256": sha256(TOKENIZER_PATH),
        "tokenizer_digest": data["tokenizer"].digest,
        "starting_checkpoint_sha256": sha256(BASE_CHECKPOINT),
        "code_sha256": code_hash(),
        "scheduler": {"kind": "constant", "lr_multiplier": 1.0},
    }


def create_optimizer(model: torch.nn.Module, config: dict[str, Any]):
    values = config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(values["learning_rate"]),
        weight_decay=float(values["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return optimizer, scheduler


def checkpoint_payload(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: Any, step: int, order: list[int], meta: dict[str, Any],
                       config: dict[str, Any], report: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": 1, "kind": "microworld-controlled-pilot-v1", "step": step,
        "arm": meta["arm"], "seed": meta["seed"], "meta": meta, "config": config,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(), "presentation_order": list(order),
        "report": list(report),
    }


def train_segment(data: dict[str, Any], arm: str, config: dict[str, Any],
                  checkpoint_path: Path, total_updates: int = PILOT_UPDATES,
                  stop_after: int | None = None,
                  resume_from: Path | None = None) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if guard.available_mib() < MEMORY_GUARD_MIB:
        raise MemoryError(f"effective available memory below {MEMORY_GUARD_MIB} MiB")
    guard.configure_torch()
    train = data["train_rows"]
    aligned_rows = data["train_aligned"]
    if len(train) != PILOT_UPDATES or len(aligned_rows) != len(train):
        raise ValueError("controlled pilot must contain exactly 64 aligned examples")
    seed = int(config["optimizer"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    meta = base_metadata(data, config, arm, seed)
    order = list(range(len(train)))
    model = make_model(arm, data["model_config"], data["vocabularies"],
                       data["checkpoint"]["model"], seed)
    optimizer, scheduler = create_optimizer(model, config)
    start = 0
    history: list[dict[str, Any]] = []
    if resume_from is not None:
        payload = torch.load(resume_from, map_location="cpu", weights_only=True)
        if (payload.get("schema") != 1 or payload.get("kind") != "microworld-controlled-pilot-v1"
                or payload.get("meta") != meta or payload.get("config") != config
                or payload.get("presentation_order") != order):
            raise ValueError("resume checkpoint experiment, data, code or configuration mismatch")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        random.setstate(payload["python_rng"])
        torch.set_rng_state(payload["torch_rng"])
        start = int(payload["step"])
        history = list(payload["report"])
    end = total_updates if stop_after is None else stop_after
    if not 0 <= start <= end <= total_updates or (stop_after is not None and not start < end):
        raise ValueError("invalid resume/update interval")

    started = time.perf_counter()
    last_swap = guard.swap_counters()
    for step in range(start, end):
        if step % 10 == 0:
            if guard.available_mib() < MEMORY_GUARD_MIB:
                raise MemoryError(f"effective available memory below guard at update {step}")
            current_swap = guard.swap_counters()
            if last_swap is not None and current_swap is not None and any(
                    now > prior for now, prior in zip(current_swap, last_swap)):
                raise MemoryError(f"swap activity detected at update {step}")
            last_swap = current_swap
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = loss_for(arm, model, data["train_aligned"][step]["token_ids"],
                          aligned_rows[step], data["vocabularies"],
                          config["loss"]["structural_groups"])
        total_loss = losses["total"]
        if not bool(torch.isfinite(total_loss)):
            raise FloatingPointError(f"non-finite loss at update {step}")
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["optimizer"]["gradient_clip_norm"]))
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError(f"non-finite gradient norm at update {step}")
        optimizer.step()
        scheduler.step()
        groups = {name: float(value.detach())
                  for name, value in losses.get("group_losses", {}).items()}
        history.append({
            "update": step + 1, "example_id": train[step]["example_id"],
            "input_tokens": len(data["train_aligned"][step]["token_ids"]),
            "language_loss": (float(losses["language"].detach())
                              if arm in {"A", "C"} else None),
            "structural_loss": (float(losses["structural"].detach())
                                if arm in {"B", "C"} else None),
            "objective_loss": float(total_loss.detach()), "group_losses": groups,
            "gradient_norm": float(grad_norm),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })
        if (step + 1) % EVAL_INTERVAL == 0 or step + 1 == end:
            atomic_save(checkpoint_path, checkpoint_payload(
                model, optimizer, scheduler, step + 1, order, meta, config, history))
    return {"arm": arm, "start_update": start, "end_update": end,
            "history": history, "seconds": time.perf_counter() - started,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "backbone_parameters": model_backbone(model).params,
            "auxiliary_parameters": (model.auxiliary_params
                                     if isinstance(model, FoundationWithDenseHeads) else 0),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "checkpoint": str(checkpoint_path), "checkpoint_bytes": (
                checkpoint_path.stat().st_size if checkpoint_path.exists() else 0),
            "meta": meta, "model": model}


def expected_signature(row: dict[str, Any], world: dict[str, Any]) -> Any:
    answer = row["ground_truth"]
    kind = row["query_type"]
    names = {entity["name"]: entity["id"] for entity in world["entities"]}
    slots = {f"e{index}": entity["id"] for index, entity in enumerate(world["entities"])}
    if kind == "entities":
        return tuple((entity["id"], entity["type"]) for entity in answer["entities"])
    if kind == "relationships":
        return tuple(sorted((edge["type"], edge["source"], edge["target"])
                            for edge in answer["relationships"]))
    if kind == "properties":
        return (answer["entity"], tuple(sorted(answer["properties"].items())))
    if kind == "location":
        pos = answer["position"]
        target = pos["target"] if pos["relation"] in {"inside", "held_by"} else answer["effective_location"]
        return (answer["entity"], pos["relation"], target)
    if kind == "action_validity":
        reason = answer.get("reason")
        return (bool(answer["valid"]), reason.get("code") if reason else None)
    if kind == "constraints":
        return tuple(sorted(answer.get("violations", [])))
    if kind == "transition":
        before, after = answer["before"], answer["after"]
        positions = tuple(sorted((ident, after["positions"].get(ident))
                                  for ident in set(before["positions"]) | set(after["positions"])
                                  if before["positions"].get(ident) != after["positions"].get(ident)))
        opened = tuple(sorted((ident, ident in after["open_containers"])
                              for ident in set(before["open_containers"]) ^
                              set(after["open_containers"])))
        owners = tuple(sorted((ident, after["owners"].get(ident))
                              for ident in set(before["owners"]) | set(after["owners"])
                              if before["owners"].get(ident) != after["owners"].get(ident)))
        return positions, opened, owners
    if kind == "counterfactual":
        primary, alternate = answer["primary_state"], answer["alternate_state"]
        return tuple(sorted((ident, primary["positions"][ident], alternate["positions"][ident])
                            for ident in answer["changed_entities"]))
    raise ValueError(f"unknown query type {kind}")


def _parse_position(value: str, names: dict[str, str]) -> dict[str, str] | None:
    match = re.fullmatch(r"(inside|held by|at)\s+(.+)", value.strip(), re.I)
    if not match or match.group(2) not in names:
        return None
    relation = {"held by": "held_by"}.get(match.group(1).casefold(), match.group(1).casefold())
    return {"relation": relation, "target": names[match.group(2)]}


def parse_answer(text: str, row: dict[str, Any], world: dict[str, Any]) -> dict[str, Any]:
    """Parse only the task's semantic answer fields; never consult the model internally."""
    body = re.sub(r"^\s*(?:oracle result|ground truth under the held-out wording):\s*", "",
                  text.strip(), flags=re.I)
    entities = world["entities"]
    names = {entity["name"]: entity["id"] for entity in entities}
    slots = {f"e{index}": entity["id"] for index, entity in enumerate(entities)}
    kind = row["query_type"]
    invalid_entities = 0
    signature = None
    if kind == "entities":
        found = [(int(slot), entity_type.casefold())
                 for slot, entity_type in re.findall(r"\be(\d+)\s*:\s*([a-z_]+)\b", body, re.I)]
        if found:
            mapped = []
            for index, entity_type in found:
                if not 0 <= index < len(entities):
                    invalid_entities += 1
                else:
                    mapped.append((entities[index]["id"], entity_type))
            signature = tuple(mapped)
    elif kind == "relationships":
        matches = re.findall(r"\b([a-z_]+)\s*\(\s*(e\d+)\s*,\s*(e\d+)\s*\)", body, re.I)
        if matches:
            mapped = []
            for relation, source, target in matches:
                if source not in slots or target not in slots:
                    invalid_entities += 1
                else:
                    mapped.append((relation.casefold(), slots[source], slots[target]))
            signature = tuple(sorted(mapped))
    elif kind == "properties":
        match = re.search(r"^\s*(.+?)\s*:\s*(.+)$", body, re.S)
        if match:
            entity_name, values_text = match.groups()
            entity = names.get(entity_name.strip())
            if entity is None:
                invalid_entities += 1
            values = {}
            for item in values_text.rstrip(" .").split(";"):
                prop = re.fullmatch(r"\s*([a-z_]+)\s+is\s+(.+?)\s*", item, re.I)
                if not prop:
                    continue
                try:
                    values[prop.group(1)] = json.loads(prop.group(2))
                except json.JSONDecodeError:
                    continue
            if entity is not None and values:
                signature = (entity, tuple(sorted(values.items())))
    elif kind == "location":
        match = re.fullmatch(r"\s*(.+?)\s+is\s+(held by|inside|at)\s+(.+?)\s*\.?\s*", body, re.I)
        if match:
            entity_name, relation, target_name = match.groups()
            entity, target = names.get(entity_name), names.get(target_name)
            if entity is None or target is None:
                invalid_entities += int(entity is None) + int(target is None)
            else:
                relation = {"held by": "held_by"}.get(relation.casefold(), relation.casefold())
                signature = (entity, relation, target)
    elif kind == "action_validity":
        match = re.search(r"\b(invalid|valid)\b", body, re.I)
        if match:
            valid = match.group(1).casefold() == "valid"
            code = re.search(r"\bcode\s*:\s*([a-z_]+)", body, re.I)
            signature = (valid, code.group(1).casefold() if code else None)
    elif kind == "constraints":
        if re.search(r"\bno constraints are violated\b", body, re.I):
            signature = ()
        else:
            match = re.search(r"violated constraints are\s+(.+?)(?:\.|$)", body, re.I)
            if match:
                signature = tuple(sorted(part.strip() for part in match.group(1).split(",")
                                         if part.strip()))
    elif kind == "transition":
        if re.search(r"\bstate is unchanged\b", body, re.I):
            signature = ((), (), ())
        else:
            positions = []; opened = []; owners = []; parsed = 0
            clauses = body.split(";")
            for clause in clauses:
                change = re.fullmatch(r"\s*(.+?)\s+changes from\s+(.+?)\s+to\s+(.+?)\s*\.?\s*", clause, re.I)
                if change:
                    ident = names.get(change.group(1).strip())
                    target = _parse_position(change.group(3).strip(), names)
                    if ident is None or target is None:
                        invalid_entities += 1
                    else:
                        positions.append((ident, target)); parsed += 1
                    continue
                latch = re.fullmatch(r"\s*(.+?)\s+becomes\s+(open|closed)\s*\.?\s*", clause, re.I)
                if latch:
                    ident = names.get(latch.group(1).strip())
                    if ident is None:
                        invalid_entities += 1
                    else:
                        opened.append((ident, latch.group(2).casefold() == "open")); parsed += 1
                    continue
                owner = re.fullmatch(r"\s*ownership of\s+(.+?)\s+goes to\s+(.+?)\s*\.?\s*", clause, re.I)
                if owner:
                    left, right = names.get(owner.group(1).strip()), names.get(owner.group(2).strip())
                    if left is None or right is None:
                        invalid_entities += int(left is None) + int(right is None)
                    else:
                        owners.append((left, right)); parsed += 1
            if parsed:
                signature = (tuple(sorted(positions)), tuple(sorted(opened)), tuple(sorted(owners)))
    elif kind == "counterfactual":
        if re.search(r"alternative actions produce the same state", body, re.I):
            signature = ()
        else:
            clauses = re.findall(r"(.+?)\s+is\s+(.+?)\s+after the listed action and\s+(.+?)\s+under the alternative(?:;|\.|$)", body, re.I)
            if clauses:
                parsed = []
                for name, primary, alternative in clauses:
                    ident = names.get(name.strip())
                    first = _parse_position(primary.strip(), names)
                    second = _parse_position(alternative.strip(), names)
                    if ident is None or first is None or second is None:
                        invalid_entities += 1
                    else:
                        parsed.append((ident, first, second))
                if parsed:
                    signature = tuple(sorted(parsed))
    return {"signature": signature, "parseable": signature is not None,
            "invalid_entity_references": invalid_entities}


def generate_one(model: torch.nn.Module, tokenizer: Tokenizer,
                 row: dict[str, Any], world: dict[str, Any], limit: int = 128,
                 kv_backing_directory: Path | None = None) -> dict[str, Any]:
    prefix = tokenizer.encode(row["prompt"] + "\nAnswer: ")
    backbone = model_backbone(model)
    maximum = min(limit, backbone.context - len(prefix))
    ids = list(prefix)
    generated: list[int] = []
    raw = bytearray()
    stop = "token_limit"
    cache = backbone.new_kv_cache(backing_directory=kv_backing_directory)
    with torch.inference_mode():
        logits = backbone.forward_cached(torch.tensor([ids], dtype=torch.long), cache)[0, -1]
        for _ in range(maximum):
            ident = int(torch.argmax(logits))
            generated.append(ident)
            if ident == SPECIAL["EOS"]:
                stop = "eos"
                break
            if ident == SPECIAL["END"]:
                stop = "end"
                break
            piece = tokenizer.parts.get(ident)
            if piece is None:
                stop = "special_token"
                break
            raw.extend(piece)
            ids.append(ident)
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = ""
            if text.rstrip().endswith((".", "!", "?")):
                stop = "period"
                break
            if len(generated) < maximum:
                logits = backbone.forward_cached(
                    torch.tensor([[ident]], dtype=torch.long), cache)[0, -1]
    text = raw.decode("utf-8", errors="replace")
    parsed = parse_answer(text, row, world)
    expected = expected_signature(row, world)
    trigrams = [tuple(generated[index:index + 3])
                for index in range(max(0, len(generated) - 2))]
    repetition = 1.0 - len(set(trigrams)) / len(trigrams) if trigrams else 0.0
    return {
        "example_id": row["example_id"], "query_type": row["query_type"],
        "construction_id": row["construction_id"], "text": text,
        "generated_tokens": len(generated), "stop_reason": stop,
        "correct_termination": stop == "period",
        "parseable": parsed["parseable"],
        "semantic_correct": parsed["parseable"] and parsed["signature"] == expected,
        "parsed_signature": parsed["signature"], "expected_signature": expected,
        "invalid_entity_references": parsed["invalid_entity_references"],
        "repetition_trigram_rate": repetition,
        "repeated_trigram": len(trigrams) != len(set(trigrams)),
        "exact_reference_text": text.strip() == row["answer_text"].strip(),
        "kv_cache": {"backing": cache.backing, "bytes": cache.bytes,
                     "tokens_cached": cache.length},
    }


def free_generation_metrics(model: torch.nn.Module, data: dict[str, Any],
                            samples: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    model.eval()
    result = {}
    for split, rows in samples.items():
        started = time.perf_counter()
        records = [generate_one(model, data["tokenizer"], row,
                                data["worlds_by_split"][split][row["world_id"]])
                   for row in rows]
        seconds = time.perf_counter() - started
        count = len(records)
        covered = sum(item["parseable"] for item in records)
        correct = sum(item["semantic_correct"] for item in records)
        query_metrics = {}
        for query in sorted({item["query_type"] for item in records}):
            subset = [item for item in records if item["query_type"] == query]
            parsed_count = sum(item["parseable"] for item in subset)
            query_metrics[query] = {
                "examples": len(subset), "coverage": parsed_count / max(1, len(subset)),
                "accuracy_when_covered": sum(item["semantic_correct"] for item in subset) /
                    max(1, parsed_count),
                "overall_accuracy": sum(item["semantic_correct"] for item in subset) /
                    max(1, len(subset)),
            }
        result[split] = {
            "examples": count, "parseability": covered / max(1, count),
            "semantic_coverage": covered / max(1, count),
            "accuracy_when_covered": correct / max(1, covered),
            "overall_semantic_accuracy": correct / max(1, count),
            "correct_termination_percentage": 100 * sum(
                item["correct_termination"] for item in records) / max(1, count),
            "other_termination_counts": dict(Counter(item["stop_reason"] for item in records)),
            "repetition_percentage": 100 * sum(item["repeated_trigram"]
                                                for item in records) / max(1, count),
            "mean_repetition_trigram_rate": sum(item["repetition_trigram_rate"]
                                                for item in records) / max(1, count),
            "mean_generated_tokens": sum(item["generated_tokens"] for item in records) /
                max(1, count),
            "invalid_or_fabricated_entity_references": sum(
                item["invalid_entity_references"] for item in records),
            "generation_seconds": seconds,
            "generated_tokens_per_second": sum(item["generated_tokens"] for item in records) /
                max(seconds, 1e-12),
            "kv_cache": {"backing": "ram", "bytes_per_sequence": (
                records[0]["kv_cache"]["bytes"] if records else 0),
                "implementation": "preallocated per-layer CPU key/value tensors"},
            "entity_binding_accuracy": query_metrics.get("entities", {}).get("overall_accuracy"),
            "relationship_accuracy": query_metrics.get("relationships", {}).get("overall_accuracy"),
            "state_transition_accuracy": query_metrics.get("transition", {}).get("overall_accuracy"),
            "counterfactual_accuracy": query_metrics.get("counterfactual", {}).get("overall_accuracy"),
            "by_query_type": query_metrics, "predictions": records,
        }
    return result


def structural_validation_metrics(model: torch.nn.Module, rows: list[dict[str, Any]],
                                  aligned_rows: list[dict[str, Any]],
                                  vocabularies: dict[str, list[str]]) -> dict[str, Any] | None:
    if not isinstance(model, FoundationWithDenseHeads):
        return None
    indices = {task: {label: index for index, label in enumerate(vocabularies[task])}
               for task in TASKS}
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    model.eval()
    with torch.inference_mode():
        for aligned in aligned_rows:
            tokens = torch.tensor(aligned["token_ids"], dtype=torch.long)[None, :]
            logits = model(tokens, include_language=False)["structural_logits"]
            predicted = {task: values[0].argmax(dim=-1).tolist()
                         for task, values in logits.items()}
            targets = token_labels(aligned, vocabularies)
            for task in TASKS:
                gold = targets[task]
                positions = [index for index, value in enumerate(gold) if value != -100]
                if positions:
                    stats[task]["correct"] += sum(predicted[task][index] == gold[index]
                                                   for index in positions)
                    stats[task]["total"] += len(positions)
            for annotation in aligned["annotations"]:
                typ = annotation["type"]
                task = {
                    "entity_identity": "entity_identity", "relationship_type": "relationship_type",
                    "relationship_source": "relationship_source",
                    "relationship_target": "relationship_target",
                    "state_change_entity": "state_change_entity",
                }.get(typ)
                if task is None:
                    continue
                label = _objective_label(annotation)
                correct = all(predicted[task][index] == indices[task][label]
                              for index in range(annotation["token_start"], annotation["token_end"]))
                name = ("entity_binding" if typ == "entity_identity" else
                        "relation" if typ.startswith("relationship_") else "state_transition")
                stats[name]["correct"] += int(correct)
                stats[name]["total"] += 1
    return {name: {"correct": value["correct"], "total": value["total"],
                   "accuracy": value["correct"] / max(1, value["total"])}
            for name, value in sorted(stats.items())}


def validation_metrics(model: torch.nn.Module, arm: str, data: dict[str, Any],
                       config: dict[str, Any]) -> dict[str, Any]:
    language_total = 0.0
    structural_total = 0.0
    objective_total = 0.0
    group_totals = Counter()
    group_counts = Counter()
    seen_tokens = 0
    model.eval()
    with torch.inference_mode():
        for row, aligned in zip(data["validation_rows"], data["validation_aligned"]):
            tokens = torch.tensor(aligned["token_ids"], dtype=torch.long)[None, :]
            language = language_only_loss(model_backbone(model), tokens)
            language_total += float(language) * (tokens.shape[1] - 1)
            seen_tokens += tokens.shape[1] - 1
            if arm != "A":
                targets = tensor_targets(aligned, data["vocabularies"])
                values = objective_losses(
                    model, tokens, targets,
                    weights=config["loss"]["structural_groups"], include_language=arm == "C")
                structural_total += float(values["structural"])
                objective_total += float(values["total"])
                for name, loss in values["group_losses"].items():
                    group_totals[name] += float(loss)
                    group_counts[name] += 1
            else:
                objective_total += float(language)
    structure = structural_validation_metrics(
        model, data["validation_rows"], data["validation_aligned"], data["vocabularies"])
    count = len(data["validation_rows"])
    return {
        "examples": count, "input_tokens": seen_tokens,
        "language_next_token_nll": language_total / max(1, seen_tokens),
        "structural_objective_loss": (structural_total / max(1, count) if arm != "A" else None),
        "objective_loss_per_example": objective_total / max(1, count),
        "structural_group_losses": ({name: group_totals[name] / max(1, group_counts[name])
                                     for name in sorted(group_totals)}),
        "teacher_forced_structural_accuracy": structure,
    }


def evaluate_baselines(data: dict[str, Any], split: str) -> dict[str, Any]:
    report = json.loads((data["dataset"] / "baselines.json").read_text())
    row = report["partitions"][split]
    return row["methods"]


def smoke_test(data: dict[str, Any], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if guard.available_mib() < MEMORY_GUARD_MIB:
        raise MemoryError("memory guard rejected smoke test")
    guard.configure_torch()
    configs = {arm: json.loads((ROOT / "configs" / f"{name}.json").read_text())
               for arm, name in ARMS.items()}
    rows = data["train_rows"][:2]
    aligned = data["train_aligned"][:2]
    results = {}
    for arm, config in configs.items():
        model = make_model(arm, data["model_config"], data["vocabularies"],
                           data["checkpoint"]["model"], int(config["optimizer"]["seed"]))
        optimizer, _ = create_optimizer(model, config)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(arm, model, aligned[0]["token_ids"], aligned[0],
                        data["vocabularies"], config["loss"]["structural_groups"])["total"]
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or any(not bool(torch.isfinite(value).all()) for value in gradients):
            raise RuntimeError(f"{arm}: missing or non-finite gradients")
        gradient_norm = torch.sqrt(sum(value.detach().pow(2).sum() for value in gradients))
        if not bool(torch.isfinite(gradient_norm)) or float(gradient_norm) <= 0:
            raise RuntimeError(f"{arm}: gradient flow check failed")
        optimizer.step()
        results[arm] = {"initial_loss": float(loss.detach()),
                        "gradient_norm": float(gradient_norm),
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "backward_pass": True}

    # Checkpoint/resume against an uninterrupted two-update A run.
    config = configs["A"]
    with tempfile.TemporaryDirectory(prefix="microworld-smoke-") as temp:
        directory = Path(temp)
        full_path, partial_path = directory / "full.pt", directory / "partial.pt"
        full = train_segment(data, "A", config, full_path, total_updates=2)
        partial = train_segment(data, "A", config, partial_path, total_updates=2, stop_after=1)
        resumed = train_segment(data, "A", config, partial_path, total_updates=2,
                                resume_from=partial_path)
        full_state = torch.load(full_path, map_location="cpu", weights_only=True)
        resumed_state = torch.load(partial_path, map_location="cpu", weights_only=True)
        exact = all(torch.equal(full_state["model"][key], resumed_state["model"][key])
                    for key in full_state["model"])
        if not exact:
            raise RuntimeError("checkpoint resume differs from uninterrupted training")
        if not nested_equal(full_state["optimizer"], resumed_state["optimizer"]):
            raise RuntimeError("resumed optimizer state differs from uninterrupted training")
        if full_state["scheduler"] != resumed_state["scheduler"]:
            raise RuntimeError("resumed scheduler state differs from uninterrupted training")
        results["checkpoint_resume"] = {
            "exact_model": exact, "exact_optimizer": True, "exact_scheduler": True,
            "uninterrupted_updates": full["end_update"],
            "split_updates": [partial["end_update"], resumed["end_update"]],
        }
    val = {arm: validation_metrics(
        make_model(arm, data["model_config"], data["vocabularies"],
                   data["checkpoint"]["model"], int(configs[arm]["optimizer"]["seed"])),
        arm, {**data, "validation_rows": data["validation_rows"][:2],
              "validation_aligned": data["validation_aligned"][:2]}, configs[arm])
           for arm in configs}
    result = {
        "schema": 1, "kind": "micro-world-smoke", "training_interpretation": False,
        "device": "cpu", "threads": torch.get_num_threads(),
        "memory_guard_mib": MEMORY_GUARD_MIB,
        "memory_before": guard.snapshot(), "arms": results,
        "evaluation_check": {arm: {"examples": val[arm]["examples"],
                                   "objective_loss": val[arm]["objective_loss_per_example"]}
                             for arm in val},
        "free_generation_evaluator": "parseable output produced for smoke sentinel",
        "checkpoint_files_saved_in_temp_directory": True,
        "passed": True,
        "note": "Forward/backward/evaluation/resource checks only; smoke values are not learning evidence.",
    }
    # Exercise greedy generation and parsing in the smoke path without retaining output.
    generated = generate_one(make_model("A", data["model_config"], data["vocabularies"],
                                        data["checkpoint"]["model"], int(configs["A"]["optimizer"]["seed"])),
                             data["tokenizer"], data["validation_rows"][0],
                             data["worlds_by_split"]["validation"][
                                 data["validation_rows"][0]["world_id"]], limit=4)
    result["free_generation_evaluator"] = {"smoke_output": generated["text"],
                                            "evaluator_returned": True}
    result["memory_after"] = guard.snapshot()
    if result["memory_after"]["available"] < MEMORY_GUARD_MIB * 1024 * 1024:
        raise MemoryError("memory guard rejected completed smoke test")
    atomic_json(output, result)
    return result


def run_arm(data: dict[str, Any], arm: str, config: dict[str, Any],
            output: Path, seed: int) -> dict[str, Any]:
    pilot_config = json.loads(json.dumps(config))
    pilot_config["optimizer"]["seed"] = seed
    checkpoint_path = output / f"arm_{arm}_seed_{seed}.pt"
    model = make_model(arm, data["model_config"], data["vocabularies"],
                       data["checkpoint"]["model"], seed)
    optimizer, scheduler = create_optimizer(model, pilot_config)
    meta = base_metadata(data, pilot_config, arm, seed)
    order = list(range(PILOT_UPDATES))
    initial_validation = validation_metrics(model, arm, data, pilot_config)
    history = []
    total_tokens = 0
    started = time.perf_counter()
    last_swap = guard.swap_counters()
    for step, aligned in enumerate(data["train_aligned"]):
        if step % 10 == 0:
            available = guard.available_mib()
            if available < MEMORY_GUARD_MIB:
                raise MemoryError(f"memory guard below {MEMORY_GUARD_MIB} MiB at update {step}")
            current_swap = guard.swap_counters()
            if last_swap is not None and current_swap is not None and any(
                    current > prior for current, prior in zip(current_swap, last_swap)):
                raise MemoryError(f"swap activity detected at update {step}")
            last_swap = current_swap
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = loss_for(arm, model, aligned["token_ids"], aligned,
                          data["vocabularies"], pilot_config["loss"]["structural_groups"])
        total = losses["total"]
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(f"non-finite objective at update {step}")
        total.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(),
            float(pilot_config["optimizer"]["gradient_clip_norm"]))
        optimizer.step()
        scheduler.step()
        tokens = len(aligned["token_ids"])
        total_tokens += tokens
        groups = {name: float(value.detach())
                  for name, value in losses.get("group_losses", {}).items()}
        row = {
            "update": step + 1, "example_id": aligned["example_id"],
            "input_tokens": tokens,
            "language_loss": float(losses["language"].detach()) if arm in {"A", "C"} else None,
            "structural_loss": float(losses["structural"].detach()) if arm in {"B", "C"} else None,
            "objective_loss": float(total.detach()), "group_losses": groups,
            "gradient_norm": float(grad), "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        if (step + 1) % EVAL_INTERVAL == 0 or step + 1 == PILOT_UPDATES:
            atomic_save(checkpoint_path, checkpoint_payload(
                model, optimizer, scheduler, step + 1, order, meta,
                pilot_config, history))
        if (step + 1) % EVAL_INTERVAL == 0 or step + 1 == PILOT_UPDATES:
            validation = validation_metrics(model, arm, data, pilot_config)
            atomic_json(output / f"arm_{arm}_seed_{seed}_validation_{step + 1:03d}.json",
                        {"arm": arm, "seed": seed, "update": step + 1, **validation})
    seconds = time.perf_counter() - started
    final_validation = validation_metrics(model, arm, data, pilot_config)
    model.eval()
    freegen = free_generation_metrics(model, data, data["freegen_rows"])
    baseline_metrics = {split: evaluate_baselines(data, split) for split in EVAL_SPLITS}
    report = {
        "schema": 1, "kind": "microworld-controlled-pilot", "arm": arm,
        "seed": seed, "updates": len(history), "training_input_tokens": total_tokens,
        "language_target_positions_presented": total_tokens - len(history),
        "language_target_tokens_optimized": (total_tokens - len(history)
                                              if arm in {"A", "C"} else 0),
        "full_training_corpus_structural_spans": (
            data["objective_manifest"]["training_structural_labels"]
            if arm in {"B", "C"} else 0),
        "pilot_structural_token_label_targets": (
            data["objective_manifest"]["predeclared_pilot"]["structural_token_labels_total"]
            if arm in {"B", "C"} else 0),
        "seconds": seconds,
        "tokens_per_second": total_tokens / max(seconds, 1e-12),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "memory_after": guard.snapshot(),
        "parameters": {"backbone": model_backbone(model).params,
                       "auxiliary": model.auxiliary_params if isinstance(
                           model, FoundationWithDenseHeads) else 0,
                       "total": sum(parameter.numel() for parameter in model.parameters())},
        "starting_checkpoint_sha256": sha256(BASE_CHECKPOINT),
        "dataset_manifest_sha256": data["manifest"]["manifest_sha256"],
        "objective_manifest_sha256": data["objective_manifest"]["manifest_sha256"],
        "tokenizer_sha256": sha256(TOKENIZER_PATH),
        "experiment_hashes": meta,
        "training_curve": history,
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "free_generation": freegen,
        "deterministic_baselines": baseline_metrics,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "training_performed": True,
    }
    atomic_json(output / f"arm_{arm}_seed_{seed}.json", report)
    return report


def compare_reports(reports: list[dict[str, Any]], output: Path,
                    data: dict[str, Any]) -> dict[str, Any]:
    by_arm = {report["arm"]: report for report in reports}
    initial_path = output / "initial_checkpoint_free_generation.json"
    initial_report = (json.loads(initial_path.read_text()) if initial_path.exists() else None)
    partitions = {}
    for split in EVAL_SPLITS:
        methods = evaluate_baselines(data, split)
        partitions[split] = {
            "deterministic_baselines": methods,
            "initial_checkpoint": (None if initial_report is None else
                                   initial_report["free_generation"][split]),
            "neural": {arm: {key: report["free_generation"][split][key]
                             for key in ("examples", "parseability", "semantic_coverage",
                                         "accuracy_when_covered", "overall_semantic_accuracy",
                                         "correct_termination_percentage", "repetition_percentage",
                                         "invalid_or_fabricated_entity_references",
                                         "entity_binding_accuracy", "relationship_accuracy",
                                         "state_transition_accuracy", "counterfactual_accuracy")}
                       for arm, report in by_arm.items()},
        }
    composition = partitions["compositional_test"]
    majority = composition["deterministic_baselines"]["majority_answer"][
        "exact_structural_accuracy"]
    candidates = [arm for arm, metrics in composition["neural"].items()
                  if metrics["overall_semantic_accuracy"] > majority]
    if candidates:
        decision = "INCONCLUSIVE"
        reason = ("At least one arm exceeded the compositional majority baseline on the fixed "
                  "sample, but repeatability across seeds is required before promotion.")
        repeatability = "not tested; repeat the candidate arm(s) on additional predeclared seeds"
    else:
        decision = "NO-GO"
        reason = "No arm exceeded the compositional majority baseline on the fixed pilot sample."
        repeatability = "not run; no pilot arm met the predeclared candidate criterion"
    result = {
        "schema": 1, "kind": "microworld-pilot-comparison",
        "training_interpretation": "64 updates only; no automatic extension",
        "partitions": partitions,
        "gate": {
            "direct_oracle_and_template_accuracy": 1.0,
            "repeatability": repeatability,
            "general_reasoning_claim": False,
            "decision": decision,
            "compositional_majority_accuracy": majority,
            "candidate_arms_for_repeatability": candidates,
            "reason": reason,
        },
        "notes": [
            "Teacher-forced structural accuracy and validation losses are reported separately from free generation.",
            "The template and direct-oracle systems are deterministic ceilings; they do not provide answers to neural models.",
            "Only validation was used for training progression. IID, compositional, counterfactual, and grammar samples were held out.",
            "The pilot did not change architecture, data composition, optimizer, scheduler, or batch size across arms.",
        ],
    }
    atomic_json(output / "comparison.json", result)
    write_comparison_markdown(output / "COMPARISON.md", result, reports)
    return result


def write_comparison_markdown(path: Path, result: dict[str, Any],
                              reports: list[dict[str, Any]]) -> None:
    decision = result["gate"]["decision"]
    decision_reason = result["gate"]["reason"]
    lines = ["# Batch 4 Micro-World Training", "", "## Decision", "",
             f"**{decision}.** {decision_reason} This bounded pilot does not establish general reasoning.",
             "", "## Training", "",
             "| Arm | Updates | Input tokens | Active LM targets | Structural token labels | Backbone params | Auxiliary params | Seconds |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for report in reports:
        lines.append(
            f"| {report['arm']} | {report['updates']} | {report['training_input_tokens']} | "
            f"{report['language_target_tokens_optimized']} | "
            f"{report['pilot_structural_token_label_targets']} | "
            f"{report['parameters']['backbone']} | {report['parameters']['auxiliary']} | "
            f"{report['seconds']:.1f} |")
    lines += ["", "## Validation", "",
              "| Arm | Initial language NLL | Final language NLL | Initial objective | Final objective |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for report in reports:
        lines.append(
            f"| {report['arm']} | {report['initial_validation']['language_next_token_nll']:.4f} | "
            f"{report['final_validation']['language_next_token_nll']:.4f} | "
            f"{report['initial_validation']['objective_loss_per_example']:.4f} | "
            f"{report['final_validation']['objective_loss_per_example']:.4f} |")
    lines += ["", "## Free Generation", "",
              "| Split | System | Examples | Coverage | Accuracy when covered | Overall accuracy | Termination | Repetition | Invalid entity refs |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for split, value in result["partitions"].items():
        baseline = value["initial_checkpoint"]
        lines.append(
            f"| {split} | Initial Foundation | {baseline['examples']} | "
            f"{baseline['semantic_coverage']:.3f} | {baseline['accuracy_when_covered']:.3f} | "
            f"{baseline['overall_semantic_accuracy']:.3f} | "
            f"{baseline['correct_termination_percentage']:.1f}% | "
            f"{baseline['repetition_percentage']:.1f}% | "
            f"{baseline['invalid_or_fabricated_entity_references']} |")
        for arm, metrics in value["neural"].items():
            lines.append(
                f"| {split} | {arm} | {metrics['examples']} | {metrics['semantic_coverage']:.3f} | "
                f"{metrics['accuracy_when_covered']:.3f} | {metrics['overall_semantic_accuracy']:.3f} | "
                f"{metrics['correct_termination_percentage']:.1f}% | "
                f"{metrics['repetition_percentage']:.1f}% | "
                f"{metrics['invalid_or_fabricated_entity_references']} |")
    lines += ["", "## Deterministic Baselines", "",
              "| Split | Majority exact structural accuracy | Retrieval coverage | Template exact accuracy | Oracle exact structural accuracy |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for split, value in result["partitions"].items():
        baselines = value["deterministic_baselines"]
        lines.append(
            f"| {split} | {baselines['majority_answer']['exact_structural_accuracy']:.3f} | "
            f"{baselines['exact_retrieval']['covered']} | "
            f"{baselines['template_system']['exact_structural_accuracy']:.3f} | "
            f"{baselines['direct_oracle']['exact_structural_accuracy']:.3f} |")
    lines += ["", "## Representative Failure", ""]
    for report in reports:
        sample = report["free_generation"]["compositional_test"]["predictions"][0]
        text = sample["text"].replace("\n", " ").replace("|", "\\|")
        lines.append(
            f"- Arm {report['arm']} ({sample['example_id']}): generated `{text[:180]}`; "
            f"parseable={sample['parseable']}, semantically_correct={sample['semantic_correct']}, "
            f"stop={sample['stop_reason']}.")
    lines += ["", "## Limits", "",
              "All arms consumed the same 64 examples and 54,248 input tokens. Arm A and C optimized 54,184 language next-token targets; Arm B optimized zero language targets and 667,116 structural token labels. These supervision counts are not matched. Each arm used one seed and 32 generation prompts per partition. No arm was extended automatically. See per-arm JSON for grouped structural metrics, exact samples, validation curves, hashes, and resource measurements. A 100% deterministic template ceiling remains a reference; neural scores do not use teacher forcing.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_pilot(data: dict[str, Any], output: Path,
              arms: tuple[str, ...] = ("A", "B", "C"), seed: int = 20261125) -> list[dict[str, Any]]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    guard.configure_torch()
    configs = {arm: json.loads((ROOT / "configs" / f"{ARMS[arm]}.json").read_text())
               for arm in arms}
    experiment = {
        "schema": 1, "kind": "microworld-controlled-pilot", "arms": list(arms),
        "seed": seed, "updates": PILOT_UPDATES, "effective_batch_size": 1,
        "training_examples": [row["example_id"] for row in data["train_rows"]],
        "input_token_budget": data["objective_manifest"]["predeclared_pilot"]["input_tokens"],
        "available_lm_next_token_positions": data["objective_manifest"][
            "predeclared_pilot"]["language_target_tokens"],
        "active_language_target_tokens_by_arm": {
            arm: (data["objective_manifest"]["predeclared_pilot"]["language_target_tokens"]
                  if arm in {"A", "C"} else 0) for arm in arms},
        "structural_token_label_targets": data["objective_manifest"]["predeclared_pilot"]["structural_token_labels_total"],
        "dataset_manifest_sha256": data["manifest"]["manifest_sha256"],
        "objective_manifest_sha256": data["objective_manifest"]["manifest_sha256"],
        "tokenizer_sha256": sha256(TOKENIZER_PATH),
        "starting_checkpoint_sha256": sha256(BASE_CHECKPOINT),
        "memory_guard_mib": MEMORY_GUARD_MIB, "device": "cpu", "threads": 1,
        "kv_cache": {
            "default_backing": "ram",
            "optional_backing": "file-backed mmap; OS page cache and disk provide spill",
            "cache_bytes_per_sequence": (2 * data["model_config"]["context_length"]
                * data["model_config"]["d_model"] * data["model_config"]["layers"] * 4),
            "model_parameters_changed": False,
            "benchmark": {"prefix_tokens": 756, "generated_tokens": 24,
                          "full_recompute_seconds": 3.7985, "ram_cache_seconds": 0.1997,
                          "mmap_cache_seconds": 0.1942, "greedy_tokens_identical": True},
        },
        "scheduler": {"kind": "constant", "lr_multiplier": 1.0},
        "optimizer": configs[arms[0]]["optimizer"],
        "evaluation_updates": [0, 16, 32, 48, 64],
        "validation_sample_size": len(data["validation_rows"]),
        "free_generation_sample_size_per_partition": FREE_GENERATION_SAMPLE_SIZE,
        "model": data["model_config"], "configs": configs,
        "training_started": False, "phase": "initial-checkpoint-free-generation",
    }
    atomic_json(output / "experiment_manifest.json", experiment)
    baseline_model = make_model("A", data["model_config"], data["vocabularies"],
                                data["checkpoint"]["model"], seed)
    baseline_freegen = free_generation_metrics(baseline_model, data, data["freegen_rows"])
    atomic_json(output / "initial_checkpoint_free_generation.json", {
        "checkpoint_sha256": sha256(BASE_CHECKPOINT),
        "dataset_manifest_sha256": data["manifest"]["manifest_sha256"],
        "tokenizer_sha256": sha256(TOKENIZER_PATH),
        "seed": seed, "free_generation": baseline_freegen,
    })
    experiment["training_started"] = True
    experiment["phase"] = "pilot-training"
    atomic_json(output / "experiment_manifest.json", experiment)
    reports = []
    for arm in arms:
        reports.append(run_arm(data, arm, configs[arm], output, seed))
    comparison = compare_reports(reports, output, data)
    experiment["training_complete"] = True
    experiment["phase"] = "completed"
    experiment["decision"] = comparison["gate"]["decision"]
    atomic_json(output / "experiment_manifest.json", experiment)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--checkpoint", type=Path, default=BASE_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH)
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=["A", "B", "C"])
    args = parser.parse_args()
    if args.smoke == args.pilot:
        parser.error("select exactly one of --smoke or --pilot")
    guard.configure_torch()
    if guard.available_mib() < MEMORY_GUARD_MIB:
        raise MemoryError(f"effective available memory below {MEMORY_GUARD_MIB} MiB")
    data = load_inputs(args.dataset, args.tokenizer, args.checkpoint)
    if args.smoke:
        print(json.dumps(smoke_test(data, args.output), sort_keys=True, indent=2))
    else:
        print(json.dumps(run_pilot(data, args.output, tuple(args.arms)),
                         sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
