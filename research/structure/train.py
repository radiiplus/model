"""Matched language, structure, and combined objectives on generated worlds."""

from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import random
import re
import resource
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from research.foundation import trainer as foundation_trainer
from research.foundation.model import Foundation
from research.interaction.tokenizer import SPECIAL, Tokenizer
from research.prototype import trainer as guard
from research.structure.sampler import CandidateSampler


HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset_v5"
TOKENIZER = HERE.parent / "interaction" / "data" / "ready7" / "tokenizer.json"
ARMS = {"a": "language", "b": "structure", "c": "combined"}
OBJECTIVES = {"a": "answer_language_modeling", "b": "structural_recovery",
              "c": "combined_answer_and_structure"}


def sha(path: Path) -> str:
    return foundation_trainer.file(path)


def digest(value: Any) -> str:
    return foundation_trainer.digest(value)


def tensor_state_hash(state: dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        hasher.update(key.encode())
        hasher.update(ctypes.string_at(tensor.data_ptr(), tensor.numel() * tensor.element_size()))
    return hasher.hexdigest()


def code_hash() -> str:
    return digest({str(path): sha(path) for path in
                   (HERE / "schema.py", HERE / "splits.py", HERE / "generator.py",
                    HERE / "train.py", HERE / "sampler.py",
                    HERE / "value_curriculum.py", HERE.parent / "foundation" / "model.py")})


def read_examples(root: Path, split: str) -> list[dict[str, Any]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    item = next((row for row in manifest["files"] if row["split"] == split), None)
    if item is None:
        raise ValueError(f"dataset has no {split} partition")
    path = root / item["path"]
    if sha(path) != item["sha256"]:
        raise ValueError(f"dataset hash mismatch for {split}")
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] != split:
                raise ValueError("dataset row is in the wrong split file")
            rows.append({key: row[key] for key in
                         ("example_id", "world_id", "input", "answer_target",
                          "structure_target", "answer_value", "oracle", "pattern",
                          "surface_id", "query_type")})
    if len(rows) != item["examples"]:
        raise ValueError("dataset example count differs from manifest")
    return rows


def encode_pair(row: dict[str, Any], token: Tokenizer,
                context: int) -> tuple[list[int], list[int], list[bool], list[bool], list[bool]]:
    prefix = token.chat([{"role": "user", "text": row["input"]}], bos=True, eos=False)
    outputs = []
    targets = []
    for value in (row["answer_target"], row["structure_target"]):
        ids = prefix + [SPECIAL["ASSISTANT"]] + token.encode(value) + [SPECIAL["END"], SPECIAL["EOS"]]
        target_start = len(prefix) + 1
        if len(ids) > context:
            raise ValueError(f"example {row['example_id']} exceeds model context ({len(ids)} > {context})")
        mask = [False] * len(ids)
        mask[target_start:] = [True] * (len(ids) - target_start)
        outputs.append(ids)
        targets.append(mask)
    return outputs[0], outputs[1], [True] * len(outputs[0]), targets[0], targets[1]


def tensors(rows: list[dict[str, Any]], token: Tokenizer, context: int
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                       torch.Tensor]:
    answer_ids = []
    structure_ids = []
    answer_valid = []
    answer_targets = []
    structure_targets = []
    for row in rows:
        answer, structure, valid, answer_target, structure_target = encode_pair(row, token, context)
        answer_ids.append(answer)
        structure_ids.append(structure)
        answer_valid.append(valid)
        answer_targets.append(answer_target)
        structure_targets.append(structure_target)
    width = max(max(map(len, answer_ids)), max(map(len, structure_ids)))
    width = min(width, context)
    values = []
    valid_values = []
    answer_masks = []
    structure_masks = []
    for ids, valid, mask in zip(answer_ids + structure_ids,
                                answer_valid + [[True] * len(item) for item in structure_ids],
                                answer_targets + [([False] * len(item)) for item in structure_ids]):
        pad = width - len(ids)
        values.append(ids + [SPECIAL["EOS"]] * pad)
        valid_values.append(valid + [False] * pad)
        answer_masks.append(mask + [False] * pad)
    for ids, mask in zip(answer_ids + structure_ids,
                         [([False] * len(item)) for item in answer_ids] + structure_targets):
        structure_masks.append(mask + [False] * (width - len(ids)))
    return (torch.tensor(values, dtype=torch.long),
            torch.tensor(valid_values, dtype=torch.bool),
            torch.tensor(answer_masks, dtype=torch.bool),
            torch.tensor(structure_masks, dtype=torch.bool),
            torch.tensor([len(item) for item in answer_ids + structure_ids], dtype=torch.long))


def losses(model: Foundation, rows: list[dict[str, Any]], token: Tokenizer
           ) -> dict[str, torch.Tensor]:
    values, valid, answer_mask, structure_mask, lengths = tensors(rows, token, model.context)
    logits = model(values[:, :-1])
    targets = values[:, 1:]
    valid_targets = valid[:, 1:]
    flat = F.cross_entropy(logits.reshape(-1, model.vocab), targets.reshape(-1),
                           reduction="none").reshape_as(targets)
    count = len(rows)
    language_mask = valid_targets[:count]
    answer_target_mask = answer_mask[:count, 1:]
    structure_target_mask = structure_mask[count:, 1:]
    structural = flat[count:].masked_select(structure_target_mask).mean()
    language = flat[:count].masked_select(language_mask).mean()
    answer = flat[:count].masked_select(answer_target_mask).mean()
    return {"language": language, "answer": answer, "structure": structural,
            "language_tokens": language_mask.sum(),
            "answer_tokens": answer_target_mask.sum(),
            "structure_tokens": structure_target_mask.sum(),
            "input_tokens": valid_targets.sum(), "logits": logits,
            "values": values, "lengths": lengths,
            "language_mask": language_mask, "answer_mask": answer_target_mask,
            "structure_mask": structure_target_mask}


class Order:
    def __init__(self, count: int, seed: int, state: dict | None = None):
        self.count = count
        self.rng = random.Random(seed)
        self.epoch = 0
        self.position = 0
        self.order = list(range(count))
        self.rng.shuffle(self.order)
        if state is not None:
            self.epoch = int(state["epoch"])
            self.position = int(state["position"])
            self.order = list(state["order"])
            self.rng.setstate(state["rng"])
            if sorted(self.order) != list(range(count)) or not 0 <= self.position <= count:
                raise ValueError("training example order state is invalid")

    def take(self, count: int) -> list[int]:
        result = []
        while len(result) < count:
            if self.position >= self.count:
                self.epoch += 1
                self.position = 0
                self.order = list(range(self.count))
                self.rng.shuffle(self.order)
            amount = min(count - len(result), self.count - self.position)
            result.extend(self.order[self.position:self.position + amount])
            self.position += amount
        return result

    def state(self) -> dict:
        return {"epoch": self.epoch, "position": self.position,
                "order": list(self.order), "rng": self.rng.getstate()}


def select_evaluation_rows(rows: list[dict[str, Any]], limit: int,
                           seed: int) -> list[dict[str, Any]]:
    if not rows or limit < 1:
        raise ValueError("evaluation rows and limit must be nonempty")
    surfaces = ("structured", "natural", "paraphrased", "reordered", "irrelevant",
                "counterfactual")
    questions = ("influence", "relation_after", "invariant", "action_effect")
    buckets = {(surface, question): [] for surface in surfaces for question in questions}
    for row in rows:
        key = (row.get("surface_id"), row.get("query_type"))
        if key in buckets:
            buckets[key].append(row)
    rng = random.Random(seed)
    for values in buckets.values():
        rng.shuffle(values)
    selected = []
    for index in range(limit):
        surface = surfaces[index % len(surfaces)]
        question = questions[(index // len(surfaces) + index) % len(questions)]
        values = buckets[(surface, question)]
        if values:
            selected.append(values[(index // len(surfaces)) % len(values)])
        else:
            selected.append(rows[index % len(rows)])
    return selected


def evaluate(model: Foundation, rows: list[dict[str, Any]], token: Tokenizer,
             limit: int, seed: int, generate_answers: bool = True,
             generate_structures: bool = False) -> dict[str, Any]:
    surfaces = ("structured", "natural", "paraphrased", "reordered", "irrelevant",
                "counterfactual")
    selected = select_evaluation_rows(rows, limit, seed)
    language_loss = answer_loss = structure_loss = 0.0
    language_tokens = answer_tokens = structure_tokens = structure_correct = 0
    answer_correct = 0
    answer_exact = structure_exact = structure_semantic_exact = json_valid = 0
    generation_samples = []
    model.eval()
    with torch.no_grad():
        for row in selected:
            result = losses(model, [row], token)
            language_count = int(result["language_tokens"])
            answer_count = int(result["answer_tokens"])
            structure_count = int(result["structure_tokens"])
            language_loss += float(result["language"]) * language_count
            answer_loss += float(result["answer"]) * answer_count
            structure_loss += float(result["structure"]) * structure_count
            logits = result["logits"]
            predictions = logits.argmax(dim=-1)
            values = result["values"]
            answer_len = int(result["lengths"][0])
            structure_len = int(result["lengths"][1])
            answer_mask = result["answer_mask"][0]
            answer_target = values[0, 1:answer_len][answer_mask[:answer_len - 1]]
            answer_pred = predictions[0, :answer_len - 1][answer_mask[:answer_len - 1]]
            answer_exact += int(torch.equal(answer_target, answer_pred))
            answer_correct += int((answer_target == answer_pred).sum())
            structure_target = values[1, 1:structure_len]
            structure_pred = predictions[1, :structure_len - 1]
            structure_exact += int(torch.equal(structure_target, structure_pred))
            structure_correct += int((structure_target == structure_pred).sum())
            try:
                generated = []
                for ident in structure_pred.tolist():
                    if ident in (SPECIAL["END"], SPECIAL["EOS"]):
                        break
                    if ident not in token.parts:
                        raise ValueError("generated a structural token inside JSON")
                    generated.append(ident)
                text = token.decode(generated)
                parsed = json.loads(text)
                expected = json.loads(row["structure_target"])
                json_valid += 1
                structure_semantic_exact += int(parsed == expected)
            except (ValueError, json.JSONDecodeError):
                pass
            language_tokens += language_count
            answer_tokens += answer_count
            structure_tokens += structure_count
        for row in selected[:min(2, len(selected))] if generate_answers else ():
            prefix = token.chat([{"role": "user", "text": row["input"]}],
                                bos=True, eos=False) + [SPECIAL["ASSISTANT"]]
            generated = list(prefix)
            generated_body = []
            stop_reason = "token_limit"
            invalid_special = None
            for _ in range(min(32, model.context - len(prefix))):
                next_id = int(model(torch.tensor([generated], dtype=torch.long))
                              [0, -1].argmax())
                if next_id in (SPECIAL["END"], SPECIAL["EOS"]):
                    stop_reason = "end_token" if next_id == SPECIAL["END"] else "eos_token"
                    break
                if next_id not in token.parts:
                    invalid_special = next_id
                    stop_reason = "unexpected_special_token"
                    break
                generated_body.append(next_id)
                generated.append(next_id)
            try:
                generated_text = (f"<special:{invalid_special}>" if invalid_special is not None
                                  else token.decode(generated_body))
            except (ValueError, UnicodeDecodeError):
                generated_text = ""
            cleaned = re.sub(
                r"^(?:Oracle result|The computed result|Result in this state|"
                r"Following the stated rules|After applying the graph rules|"
                r"Under the selected alternative)\s*[:,]?\s*",
                "", generated_text.strip(), flags=re.IGNORECASE)
            semantic_value = "".join(character.casefold() for character in cleaned
                                     if character.isalnum())
            expected_value = "".join(character.casefold() for character in row["answer_value"]
                                      if character.isalnum())
            generation_samples.append({
                "example_id": row["example_id"],
                "expected": row["answer_target"],
                "expected_value": row["answer_value"],
                "generated": generated_text,
                "stop_reason": stop_reason,
                "exact_match": generated_text.strip() == row["answer_target"].strip(),
                "semantic_match": bool(expected_value) and semantic_value == expected_value})
    structure_generation_samples = []
    if generate_answers and generate_structures:
        for row in selected[:1]:
            prefix = token.chat([{"role": "user", "text": row["input"]}],
                                bos=True, eos=False) + [SPECIAL["ASSISTANT"]]
            generated = list(prefix)
            body = []
            stop_reason = "token_limit"
            invalid_special = None
            for _ in range(min(480, model.context - len(prefix))):
                next_id = int(model(torch.tensor([generated], dtype=torch.long))
                              [0, -1].argmax())
                if next_id in (SPECIAL["END"], SPECIAL["EOS"]):
                    stop_reason = "end_token" if next_id == SPECIAL["END"] else "eos_token"
                    break
                if next_id not in token.parts:
                    invalid_special = next_id
                    stop_reason = "unexpected_special_token"
                    break
                body.append(next_id)
                generated.append(next_id)
            try:
                generated_text = (f"<special:{invalid_special}>" if invalid_special is not None
                                  else token.decode(body))
                parsed = json.loads(generated_text)
                expected = json.loads(row["structure_target"])
                json_valid_generated = True
                exact_generated = parsed == expected
            except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
                generated_text = "" if invalid_special is None else f"<special:{invalid_special}>"
                json_valid_generated = False
                exact_generated = False
            structure_generation_samples.append({
                "example_id": row["example_id"],
                "generated": generated_text,
                "stop_reason": stop_reason,
                "json_valid": json_valid_generated,
                "exact_structure": exact_generated,
                "expected": row["structure_target"]})
    return {"examples": len(selected),
            "language_nll": language_loss / max(1, language_tokens),
            "answer_nll": answer_loss / max(1, answer_tokens),
            "answer_token_accuracy": answer_correct / max(1, answer_tokens),
            "answer_exact": answer_exact / len(selected),
            "answer_greedy_examples": len(generation_samples),
            "answer_greedy_exact": sum(row["exact_match"] for row in generation_samples)
            / max(1, len(generation_samples)),
            "answer_semantic_accuracy": sum(row["semantic_match"] for row in generation_samples)
            / max(1, len(generation_samples)),
            "answer_generation_samples": generation_samples,
            "structure_nll": structure_loss / max(1, structure_tokens),
            "structure_token_accuracy": structure_correct / max(1, structure_tokens),
            "structure_teacher_forced_exact": structure_exact / len(selected),
            "structure_teacher_forced_semantic_accuracy":
            structure_semantic_exact / len(selected),
            "structure_teacher_forced_json_valid": json_valid / len(selected),
            "structure_greedy_examples": len(structure_generation_samples),
            "structure_greedy_json_valid": sum(row["json_valid"] for row in
                structure_generation_samples) / max(1, len(structure_generation_samples)),
            "structure_greedy_exact": sum(row["exact_structure"] for row in
                structure_generation_samples) / max(1, len(structure_generation_samples)),
            "structure_generation_samples": structure_generation_samples,
            "language_target_tokens": language_tokens,
            "answer_target_tokens": answer_tokens,
            "structure_target_tokens": structure_tokens,
            "selected_examples_sha256": digest([row["example_id"] for row in selected]),
            "surfaces": {surface: sum(row["surface_id"] == surface for row in selected)
                         for surface in surfaces},
            "evaluation_mode": ("teacher-forced token scoring plus bounded greedy generation"
                                if generate_structures else
                                "teacher-forced token scoring plus bounded greedy answer generation")}


def checkpoint(path: Path, arm: str, config: dict, model: Foundation,
               optimizer: torch.optim.Optimizer, scheduler: Any, order: Order,
               step: int, meta: dict, seed: int, history: list[dict[str, Any]]) -> None:
    foundation_trainer.save(path, {"schema": 1, "arm": arm, "config": config,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "sampler": order.state(),
        "step": step, "seed": seed, "meta": meta,
        "python": random.getstate(), "torch": torch.get_rng_state(),
        "history": history})


def run(config: dict, arm: str, initial: Path, output: Path,
        root: Path = DATA, resume: Path | None = None,
        stop: int | None = None) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError("arm must be a, b or c")
    if config.get("condition") not in {ARMS[arm], "smoke-objective-selected-by-arm"}:
        raise ValueError("configuration condition does not match the selected arm")
    model_config = config["model"]
    train = config["train"]
    seed = int(train["seed"])
    updates = int(train["steps"])
    if not 1 <= int(train["batch"]) or int(train["eval_examples"]) < 1:
        raise ValueError("batch size and evaluation example count must be positive")
    if guard.available_mib() < guard.MEMORY_GUARD_MIB:
        raise MemoryError("effective available memory below the existing 2560 MiB guard")
    guard.configure_torch()
    torch.manual_seed(seed)
    random.seed(seed)
    train_rows = read_examples(root, "train")
    evaluation_rows = {split: read_examples(root, split) for split in
                       ("validation", "iid_test", "compositional_test",
                        "pattern_test", "counterfactual_test")}
    token = Tokenizer.load(TOKENIZER)
    sampling_config = config.get("sampling")
    selection_plan = initial_scores = reference_scores = None
    sampling_meta = None
    if sampling_config:
        plan_path = Path(sampling_config["plan_path"])
        initial_path = Path(sampling_config["initial_losses_path"])
        selection_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if selection_plan.get("updates") != updates:
            raise ValueError("candidate plan update budget does not match the training config")
        initial_scores = json.loads(initial_path.read_text(encoding="utf-8"))
        reference_path_value = sampling_config.get("reference_losses_path")
        reference_path = Path(reference_path_value) if reference_path_value else None
        if sampling_config["mode"] == "rho_loss":
            if reference_path is None:
                raise ValueError("rho_loss requires a training-only reference loss file")
            reference_scores = json.loads(reference_path.read_text(encoding="utf-8"))
        sampling_meta = {"mode": sampling_config["mode"], "plan": sha(plan_path),
                         "initial_losses": sha(initial_path),
                         "reference_losses": sha(reference_path) if reference_path else None}
    dataset_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    meta = {"dataset_manifest": sha(root / "manifest.json"),
            "train": next(row["sha256"] for row in dataset_manifest["files"]
                          if row["split"] == "train"),
            "validation": next(row["sha256"] for row in dataset_manifest["files"]
                               if row["split"] == "validation"),
            "tokenizer": token.digest, "model": digest(model_config),
            "code": code_hash(), "initial": sha(initial),
            "arm": arm, "objective": ARMS[arm], "sampling": sampling_meta}
    source = torch.load(initial, map_location="cpu", weights_only=True)
    source_config = source.get("config", {}).get("model")
    if source_config != model_config or "model" not in source:
        raise ValueError("initial checkpoint architecture differs from the configured Foundation")
    model = Foundation(model_config)
    model.load_state_dict(source["model"])
    starting_weights_hash = tensor_state_hash(source["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train["rate"]),
                                  weight_decay=float(train["penalty"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: foundation_trainer.rate(step, train))
    optimizer_state_restored = bool(config.get("restore_optimizer", False))
    scheduler_state_restored = optimizer_state_restored
    if optimizer_state_restored:
        if "optimizer" not in source or "scheduler" not in source:
            raise ValueError("configured optimizer continuation requires checkpoint state")
        optimizer.load_state_dict(source["optimizer"])
        scheduler.load_state_dict(source["scheduler"])
    def new_sampler(saved_state: dict | None = None):
        if sampling_config:
            return CandidateSampler(train_rows, selection_plan, sampling_config["mode"],
                seed + 1, initial_scores, reference_scores, saved_state)
        return Order(len(train_rows), seed + 1, saved_state)

    order = new_sampler()
    start = 0
    history = []
    if resume is not None:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if (saved.get("schema") != 1 or saved.get("arm") != arm
                or saved.get("config") != config or saved.get("meta") != meta
                or saved.get("seed") != seed):
            raise ValueError("resume checkpoint does not match arm, config, data, or initialization")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        order = new_sampler(saved["sampler"])
        random.setstate(saved["python"])
        torch.set_rng_state(saved["torch"])
        history = list(saved.get("history", []))
        start = int(saved["step"])
    if start > updates or (stop is not None and not start < stop <= updates):
        raise ValueError("invalid stop update")
    if resume is None and output.exists():
        raise FileExistsError(output)
    train_count = int(train["batch"])
    eval_limit = int(train["eval_examples"])
    eval_seed = int(train.get("eval_seed", 2026))
    scheduler_start_update = int(scheduler.last_epoch) - start
    before = {split: evaluate(model, rows, token, eval_limit, eval_seed,
                              generate_answers=False)
              for split, rows in evaluation_rows.items()}
    started = time.monotonic()
    end = updates if stop is None else stop
    swap = guard.swap_counters()
    selection_seconds = 0.0
    for index in range(start, end):
        if index % 10 == 0:
            if guard.available_mib() < guard.MEMORY_GUARD_MIB:
                raise MemoryError(f"effective available memory fell below the guard at update {index}")
            current = guard.swap_counters()
            if swap is not None and current is not None and any(
                    now > prior for now, prior in zip(current, swap)):
                raise MemoryError(f"swap activity detected at update {index}")
            swap = current
        selection_start = time.monotonic()
        selection_info = None
        if sampling_config:
            score_fn = None
            if sampling_config["mode"] != "uniform":
                def score_fn(candidates):
                    was_training = model.training
                    model.eval()
                    with torch.no_grad():
                        values = [float(losses(model, [candidate], token)["structure"])
                                  for candidate in candidates]
                    model.train(was_training)
                    return values
            indices, selection_info = order.take(train_count, score_fn)
        else:
            indices = order.take(train_count)
        selection_elapsed = time.monotonic() - selection_start
        selection_seconds += selection_elapsed
        selected = [train_rows[item] for item in indices]
        model.train()
        losses_value = losses(model, selected, token)
        if selection_info is not None and int(losses_value["input_tokens"]) != selection_info["token_cost"]:
            raise ValueError("selected example differs from the predeclared per-update token budget")
        lang = losses_value["language"]
        structure = losses_value["structure"]
        if arm == "a":
            objective = losses_value["answer"]
            active_tokens = int(losses_value["answer_tokens"])
        elif arm == "b":
            objective = structure
            active_tokens = int(losses_value["structure_tokens"])
        else:
            objective = 0.5 * (losses_value["answer"] + structure)
            active_tokens = (int(losses_value["answer_tokens"])
                             + int(losses_value["structure_tokens"]))
        used_rate = float(optimizer.param_groups[0]["lr"])
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), float(train["clip"]))
        optimizer.step()
        scheduler.step()
        history.append({"step": index + 1, "language_loss": float(lang.detach()),
                        "answer_loss": float(losses_value["answer"].detach()),
                        "structure_loss": float(structure.detach()),
                        "objective_loss": float(objective.detach()),
                        "language_tokens": int(losses_value["language_tokens"]),
                        "answer_tokens": int(losses_value["answer_tokens"]),
                        "structure_tokens": int(losses_value["structure_tokens"]),
                        "active_objective_tokens": active_tokens,
                        "input_tokens": int(losses_value["input_tokens"]),
                        "examples": train_count, "learning_rate_used": used_rate,
                        "learning_rate_after": float(optimizer.param_groups[0]["lr"]),
                        "gradient_norm": float(grad),
                        "example_ids": [row["example_id"] for row in selected],
                        "selection_seconds": selection_elapsed,
                        "selection": selection_info})
        current = index + 1
        if current % int(train["checkpoint_interval"]) == 0 or current == end:
            checkpoint(output, arm, config, model, optimizer, scheduler, order,
                       current, meta, seed, history)
    fit_end = time.monotonic()
    after = {split: evaluate(model, rows, token, eval_limit, eval_seed,
                             generate_structures=bool(train.get("generate_structures", False)))
             for split, rows in evaluation_rows.items()}
    spent = fit_end - started
    return {"arm": arm, "objective": OBJECTIVES[arm], "architecture": model_config,
            "parameters": model.params, "auxiliary_parameters": 0,
            "updates": int(history[-1]["step"]) if history else start,
            "resume_update": start,
            "optimizer_state_restored": optimizer_state_restored,
            "scheduler_state_restored": scheduler_state_restored,
            "scheduler_start_update": scheduler_start_update,
            "interrupted": end < updates, "initial_checkpoint_sha256": meta["initial"],
            "dataset": meta["dataset_manifest"], "training_split": meta["train"],
            "validation_split": meta["validation"], "tokenizer": meta["tokenizer"],
            "code": meta["code"], "starting_weights_sha256": starting_weights_hash,
            "examples_consumed": sum(row["examples"] for row in history),
            "segment_examples_consumed": len(history) - start,
            "input_tokens_seen": sum(row["input_tokens"] for row in history),
            "language_target_tokens_seen": sum(row["language_tokens"] for row in history),
            "answer_target_tokens_seen": sum(row["answer_tokens"] for row in history),
            "structure_target_tokens_seen": sum(row["structure_tokens"] for row in history),
            "active_objective_tokens_seen": sum(row["active_objective_tokens"] for row in history),
            "training_sequence_sha256": digest([item["example_ids"] for item in history]),
            "sampling": sampling_meta,
            "selection_seconds": sum(row.get("selection_seconds", 0.0) for row in history),
            "selection_plan_token_budget": (sum(selection_plan["budget_tokens_per_update"][:len(history)])
                if selection_plan else None),
            "training": history, "validation_before": before,
            "validation_after": after, "seconds": time.monotonic() - started,
            "fit_seconds": spent, "examples_per_second": sum(row["examples"] for row in history) / max(spent, 0.001),
            "input_tokens_per_second": sum(row["input_tokens"] for row in history) / max(spent, 0.001),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "resources": guard.snapshot(), "checkpoint": str(output),
            "checkpoint_sha256": sha(output) if output.exists() else None}


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    result = run(load_config(args.config), args.arm, args.initial, args.output,
                 args.data, args.resume, args.stop)
    args.report.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("arm", "updates", "examples_consumed", "input_tokens_seen",
                       "seconds", "checkpoint_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
