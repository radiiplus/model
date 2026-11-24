"""Bounded supervised experiment on generated interaction corrections."""

from __future__ import annotations

import argparse
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

from research.foundation.model import Foundation
from research.foundation import trainer as foundation_trainer
from research.interaction.compact import Compact, Learn
from research.interaction.correction import Corrections
from research.interaction.semantic_benchmark import (
    DATASET, semantic_label, selected_evaluation_rows,
)
from research.interaction.tokenizer import SPECIAL, Tokenizer
from research.memory.store import Memory
from research.prototype import trainer as guard


HERE = Path(__file__).resolve().parent
OUT = HERE / "correction_pilot"
START = HERE.parent / "structure" / "pilot_v5" / "b.pt"
TOKENIZER = HERE / "data" / "ready7" / "tokenizer.json"
UPDATES = 24
CORRECTION_EXAMPLES = 24
SEED = 7122026
LR = 0.00005


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_data(root: Path = DATASET) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    claimed_manifest_hash = manifest.get("manifest_sha256")
    manifest_body = {key: value for key, value in manifest.items()
                     if key != "manifest_sha256"}
    calculated_manifest_hash = hashlib.sha256(json.dumps(
        manifest_body, sort_keys=True, indent=2).encode("utf-8")).hexdigest()
    if claimed_manifest_hash != calculated_manifest_hash:
        raise ValueError("semantic benchmark manifest hash mismatch")
    rows = {}
    for item in manifest["files"]:
        path = root / item["path"]
        if sha(path) != item["sha256"]:
            raise ValueError(f"semantic benchmark hash mismatch: {item['split']}")
        with path.open(encoding="utf-8") as stream:
            rows[item["split"]] = [json.loads(line) for line in stream]
    return rows, manifest


def load_model(path: Path) -> tuple[Foundation, dict[str, Any]]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = Foundation(state["config"]["model"])
    model.load_state_dict(state["model"])
    return model, state


def tensor_hash(model: Foundation) -> str:
    hasher = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        hasher.update(name.encode("utf-8"))
        hasher.update(ctypes.string_at(tensor.data_ptr(),
                                       tensor.numel() * tensor.element_size()))
    return hasher.hexdigest()


def generate(model: Foundation, token: Tokenizer, prompt: str,
             limit: int = 40) -> dict[str, Any]:
    prefix = token.chat([{"role": "user", "text": prompt}], bos=True, eos=False)
    ids = prefix + [SPECIAL["ASSISTANT"]]
    body = []
    reason = "token_limit"
    with torch.inference_mode():
        for _ in range(min(limit, model.context - len(ids))):
            next_id = int(model(torch.tensor([ids], dtype=torch.long))[0, -1].argmax())
            if next_id in {SPECIAL["END"], SPECIAL["EOS"]}:
                reason = "end" if next_id == SPECIAL["END"] else "eos"
                break
            if next_id not in token.parts:
                reason = "special_token"
                break
            body.append(next_id)
            ids.append(next_id)
    return {"text": token.decode(body), "tokens": len(body), "stop": reason}


def score(model: Foundation, token: Tokenizer,
          split_rows: dict[str, list[dict[str, Any]]], label: str) -> dict[str, Any]:
    metrics = {}
    for split, all_rows in split_rows.items():
        if split == "train":
            continue
        rows = selected_evaluation_rows(all_rows)
        records = []
        parsed = 0
        correct = 0
        exact = 0
        numeric_details = 0
        for row in rows:
            result = generate(model, token, row["input"])
            predicted = semantic_label(result["text"])
            parsed += predicted != "unknown"
            correct += predicted == row["answer_semantic"]
            exact += result["text"].strip().casefold().rstrip(".!?") == \
                     row["answer_target"].strip().casefold().rstrip(".!?")
            values = {int(value) for value in re.findall(r"\b\d+\b", result["text"])}
            numeric_details += {row["effective_capacity"], row["demand"]} <= values
            records.append({"example_id": row["example_id"], "world_id": row["world_id"],
                "gold": row["answer_semantic"], "predicted": predicted,
                "correct": predicted == row["answer_semantic"], "text": result["text"],
                "stop": result["stop"], "exact_response_match": bool(
                    result["text"].strip().casefold().rstrip(".!?") ==
                    row["answer_target"].strip().casefold().rstrip(".!?")),
                "numeric_details_correct": {row["effective_capacity"], row["demand"]} <= values})
        metrics[split] = {"examples": len(rows), "semantic_accuracy": correct / len(rows),
            "semantic_coverage": parsed / len(rows),
            "accuracy_when_parsed": correct / parsed if parsed else 0.0,
            "exact_response_accuracy": exact / len(rows),
            "numeric_detail_accuracy": numeric_details / len(rows),
            "predictions": records}
    return {"condition": label, "per_split": metrics,
        "semantic_accuracy": sum(item["semantic_accuracy"] * item["examples"]
                                  for item in metrics.values()) /
                             sum(item["examples"] for item in metrics.values()),
        "exact_response_accuracy": sum(item["exact_response_accuracy"] * item["examples"]
                                       for item in metrics.values()) /
                                  sum(item["examples"] for item in metrics.values())}


def encoded(token: Tokenizer, prompt: str, target: str, context: int) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = token.chat([{"role": "user", "text": prompt}], bos=True, eos=False)
    ids = prefix + [SPECIAL["ASSISTANT"]] + token.encode(target) + [SPECIAL["END"], SPECIAL["EOS"]]
    if len(ids) > context:
        raise ValueError(f"correction example exceeds context: {len(ids)} > {context}")
    start = len(prefix)
    inputs = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0)
    targets = torch.tensor(ids[1:], dtype=torch.long)
    mask = torch.zeros_like(targets, dtype=torch.bool)
    mask[start:] = True
    return inputs, (targets, mask)


def select_corrections(rows: list[dict[str, Any]], limit: int = UPDATES) -> list[dict[str, Any]]:
    by_world = {}
    for row in rows:
        if row["surface"] == "explicit" and row["action"] == "none":
            by_world.setdefault(row["world_id"], {})[row["context"]] = row
    result = []
    for index, world_id in enumerate(sorted(by_world)):
        context = "cold" if index % 2 == 0 else "warm"
        result.append(by_world[world_id][context])
    if len(result) != limit:
        raise ValueError("correction corpus does not match the predeclared update count")
    return result


def run(output: Path = OUT, start: Path = START, updates: int = UPDATES,
        learning_rate: float = LR) -> dict[str, Any]:
    if updates < CORRECTION_EXAMPLES or updates % CORRECTION_EXAMPLES:
        raise ValueError("updates must be a positive whole number of correction-corpus passes")
    if guard.available_mib() < guard.MEMORY_GUARD_MIB:
        raise MemoryError("available memory is below the existing training guard")
    guard.configure_torch()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"pilot output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    rows, manifest = read_data()
    token = Tokenizer.load(TOKENIZER)
    model, initial = load_model(start)
    if model.vocab != token.vocab or model.context < 128:
        raise ValueError("starting checkpoint and tokenizer are incompatible")
    baseline_hash = sha(start)
    starting_weights_hash = tensor_hash(model)
    baseline_result = score(model, token, rows, "frozen_start")
    train_rows = select_corrections(rows["train"], CORRECTION_EXAMPLES)

    # Simulate consented correction events from this generated fixture only.
    memory = Memory(":memory:", fts=False)
    ledger = Learn(Compact(memory))
    capture = Corrections(memory, ledger)
    policy = {"persist": True, "scope": "synthetic-correction-pilot",
              "owner": "generated-fixture", "private": True}
    captured = []
    kind_counts = {"meaning": 0, "expression": 0}
    base_samples = []
    for row in train_rows:
        draft = generate(model, token, row["input"], limit=24)
        parsed = semantic_label(draft["text"])
        meaning_confirmed = parsed == row["answer_semantic"]
        kind = "expression" if meaning_confirmed else "meaning"
        result = capture.capture(row["input"], draft["text"], row["answer_target"],
            kind=kind, meaning_confirmed=meaning_confirmed,
            scope=policy["scope"], owner=policy["owner"], policy=policy,
            model="foundation-structural-b", version=baseline_hash,
            dataset=manifest["manifest_sha256"], origin="synthetic_fixture")
        episode = ledger.fetch(result["episode"], policy["scope"], policy["owner"])
        captured.append({"row": row, "episode": episode, "event": result["event"]})
        kind_counts[kind] += 1
        base_samples.append({"example_id": row["example_id"], "draft": draft["text"],
            "draft_semantic": parsed, "gold_semantic": row["answer_semantic"],
            "correction_kind": kind, "correction": row["answer_target"]})

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    index_rng = random.Random(SEED)
    indices = []
    while len(indices) < updates:
        epoch_indices = list(range(len(captured)))
        index_rng.shuffle(epoch_indices)
        indices.extend(epoch_indices)
    model.train()
    history = []
    start_time = time.monotonic()
    token_count = 0
    losses = []
    for step, index in enumerate(indices[:updates], 1):
        item = captured[index]["episode"]
        inputs, (targets, mask) = encoded(token, item["input_text"] if "input_text" in item
                                          else captured[index]["row"]["input"],
                                          item["answer"], model.context)
        logits = model(inputs)
        shifted_logits = logits[0, mask]
        shifted_targets = targets[mask]
        loss = F.cross_entropy(shifted_logits, shifted_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        count = int(mask.sum())
        token_count += count
        losses.append(float(loss.detach()))
        history.append({"update": step, "example_id": captured[index]["row"]["example_id"],
                        "target_tokens": count, "loss": float(loss.detach()),
                        "gradient_norm": float(grad), "learning_rate": learning_rate})
    fit_seconds = time.monotonic() - start_time
    model.eval()
    corrected_result = score(model, token, rows, "supervised_corrections")
    no_split_regression = all(
        corrected_result["per_split"][split]["semantic_accuracy"]
        >= baseline_result["per_split"][split]["semantic_accuracy"]
        for split in baseline_result["per_split"])
    if (corrected_result["semantic_accuracy"] > baseline_result["semantic_accuracy"]
            and no_split_regression):
        decision = "INCONCLUSIVE: one-seed semantic gain needs independent replication"
    else:
        decision = "NO-GO: correction fine-tuning did not improve held-out semantic accuracy"
    checkpoint = output / "corrected.pt"
    foundation_trainer.save(checkpoint, {"schema": 1, "config": initial["config"],
        "model": model.state_dict(), "starting_checkpoint_sha256": baseline_hash,
        "updates": updates, "seed": SEED, "objective": "supervised corrected-response SFT"})
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    capture_report = {"origin": "synthetic_fixture", "persisted_user_data": False,
        "episodes_captured": len(captured), "correction_kind_counts": kind_counts,
        "episode_ids": [item["episode"]["instance"] for item in captured],
        "draft_samples": base_samples}
    comparison = {"schema": 1, "experiment": "correction-driven-response-pilot-v1",
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "tokenizer_file_sha256": sha(TOKENIZER), "tokenizer_digest": token.digest,
        "initial_checkpoint_sha256": baseline_hash,
        "identical_starting_weights": True,
        "starting_weights_sha256": starting_weights_hash,
        "updates": updates, "unique_correction_examples": len(captured),
        "examples_seen": updates, "correction_epochs": updates / len(captured),
        "target_tokens_seen": token_count, "fit_seconds": fit_seconds,
        "peak_rss_bytes": peak_rss, "optimizer": {"name": "AdamW", "state": "reset",
            "learning_rate": learning_rate, "weight_decay": 0.01, "batch_size": 1,
            "gradient_clip_norm": 1.0},
        "objective": "answer semantic label first; corrected natural response as target",
        "capture": capture_report, "training_losses": history,
        "baseline": baseline_result, "corrected": corrected_result,
        "decision": decision,
        "corrected_checkpoint_sha256": sha(checkpoint)}
    memory.close()
    for name, value in (("baseline.json", baseline_result),
                        ("corrected.json", corrected_result),
                        ("comparison.json", comparison)):
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--start", type=Path, default=START)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--learning-rate", type=float, default=LR)
    args = parser.parse_args()
    report = run(args.output, args.start, args.updates, args.learning_rate)
    print(json.dumps({"output": str(args.output), "decision": report["decision"],
        "baseline": report["baseline"]["semantic_accuracy"],
        "corrected": report["corrected"]["semantic_accuracy"],
        "fit_seconds": report["fit_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
