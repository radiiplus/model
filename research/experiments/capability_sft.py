#!/usr/bin/env python3
"""Tiny verifier-backed SFT demo with assistant-only loss and exact abstention."""

from __future__ import annotations

import argparse
import json
import os
import random
import resource
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from tiny_gpt import TinyGPT


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "experiments" / "baseline_config.json"
PAD, BOS, USER, ASSISTANT, ADD, MISSING, EOS, ABSTAIN = range(8)
OPERAND_BASE = 8
ANSWER_BASE = 18
ONES_TAG, CARRY_TAG, TENS_TAG = 37, 38, 39
MAX_SEQUENCE = 13


def available_mib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable unavailable; refusing to run without a memory guard")


def make_record(x: int | None, y: int | None, style: str, split: str) -> dict[str, Any]:
    if x is None and y is None:
        raise ValueError("both operands cannot be missing")
    if any(value is not None and not 0 <= value <= 9 for value in (x, y)):
        raise ValueError("known operands must be digits 0..9")
    if style not in {"concise", "worked"}:
        raise ValueError(style)
    return {"x": x, "y": y, "style": style, "split": split,
            "answerable": x is not None and y is not None}


def encode_record(record: dict[str, Any]) -> tuple[list[int], int]:
    x, y, style = record["x"], record["y"], record["style"]
    operand_x = OPERAND_BASE + x if x is not None else MISSING
    operand_y = OPERAND_BASE + y if y is not None else MISSING
    prompt = [BOS, USER, ADD, operand_x, operand_y, ASSISTANT]
    if x is None or y is None:
        completion = [ABSTAIN, EOS]
    elif style == "concise":
        completion = [ANSWER_BASE + x + y, EOS]
    else:
        total = x + y
        completion = [ONES_TAG, OPERAND_BASE + (total % 10), CARRY_TAG,
                      OPERAND_BASE + (total // 10), TENS_TAG,
                      OPERAND_BASE + (total // 10), EOS]
    return prompt + completion, len(prompt) - 1


def make_splits(style: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pairs = [(x, y) for x in range(10) for y in range(10)]
    # Hold out 20 pairs while keeping every answer value represented in training.
    heldout = {(x, y) for x, y in pairs if (7 * x + 11 * y) % 5 == 0 and x + y > 0}
    heldout.add((0, 1))
    train = [make_record(x, y, style, "train") for x, y in pairs if (x, y) not in heldout]
    train.extend(make_record(x, None, style, "train_missing") for x in range(10))
    test = [make_record(x, y, style, "heldout") for x, y in pairs if (x, y) in heldout]
    missing = [make_record(None, y, style, "missing_eval") for y in range(10)]
    return train, test, missing


def collate(records: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.full((len(records), MAX_SEQUENCE), PAD, dtype=torch.long)
    target_mask = torch.zeros((len(records), MAX_SEQUENCE - 1), dtype=torch.bool)
    for row, record in enumerate(records):
        seq, assistant_pos = encode_record(record)
        if len(seq) > MAX_SEQUENCE:
            raise ValueError("encoded example exceeds context budget")
        tokens[row, :len(seq)] = torch.tensor(seq)
        # Position j of the target is trained only after the assistant role marker.
        for target_position in range(assistant_pos + 1, len(seq)):
            target_mask[row, target_position - 1] = True
    return tokens, target_mask


def masked_sft_loss(model: TinyGPT, tokens: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:]
    flat = F.cross_entropy(logits.reshape(-1, model.vocab_size), targets.reshape(-1), reduction="none")
    mask = target_mask.reshape(-1)
    if not bool(mask.any()):
        raise ValueError("batch has no assistant completion targets")
    return flat[mask].mean()


def expected_completion(record: dict[str, Any]) -> list[int]:
    return encode_record(record)[0][6:]


@torch.no_grad()
def generate_completion(model: TinyGPT, record: dict[str, Any], max_new_tokens: int = 7) -> list[int]:
    seq, _ = encode_record({**record, "style": record["style"]})
    prefix = torch.tensor([seq[:6]], dtype=torch.long)
    out = []
    for _ in range(max_new_tokens):
        token = int(model(prefix)[:, -1].argmax(dim=-1).item())
        out.append(token)
        prefix = torch.cat([prefix, torch.tensor([[token]], dtype=torch.long)], dim=1)
        if token == EOS:
            break
    return out


def exact_verifier(record: dict[str, Any], completion: list[int]) -> dict[str, Any]:
    if not record["answerable"]:
        expected = [ABSTAIN, EOS]
        return {"correct": completion == expected, "abstained": completion[:1] == [ABSTAIN],
                "expected": expected, "verifier_status": "missing_operand"}
    total = int(record["x"]) + int(record["y"])
    if record["style"] == "concise":
        expected = [ANSWER_BASE + total, EOS]
    else:
        expected = [ONES_TAG, OPERAND_BASE + total % 10, CARRY_TAG,
                    OPERAND_BASE + total // 10, TENS_TAG, OPERAND_BASE + total // 10, EOS]
    return {"correct": completion == expected, "abstained": completion[:1] == [ABSTAIN],
            "expected": expected, "verifier_status": "verified_sum"}


def evaluate(model: TinyGPT, records: list[dict[str, Any]]) -> dict[str, Any]:
    model.eval()
    correct, false_abstain, true_abstain, missing_total = 0, 0, 0, 0
    errors = []
    for record in records:
        completion = generate_completion(model, record)
        scored = exact_verifier(record, completion)
        correct += int(scored["correct"])
        if record["answerable"]:
            false_abstain += int(scored["abstained"])
        else:
            missing_total += 1
            true_abstain += int(scored["abstained"] and scored["correct"])
        if not scored["correct"] and len(errors) < 5:
            errors.append({"x": record["x"], "y": record["y"], "completion": completion, **scored})
    answerable = sum(record["answerable"] for record in records)
    return {"n": len(records), "exact_task_accuracy": correct / len(records) if records else None,
            "answerable_exact_accuracy": (correct - true_abstain) / answerable if answerable else None,
            "abstention_recall": true_abstain / missing_total if missing_total else None,
            "false_abstention_rate": false_abstain / answerable if answerable else None,
            "correct": correct, "answerable_count": answerable,
            "missing_count": missing_total, "correct_abstentions": true_abstain,
            "false_abstentions": false_abstain, "first_incorrect_examples": errors}


def train_one(base_config: dict[str, Any], style: str, seed: int, epochs: int, batch_size: int) -> dict[str, Any]:
    if available_mib() < 2560:
        raise MemoryError("MemAvailable below 2560 MiB guard")
    config = dict(base_config)
    config.update({"vocab_size": 40, "context_length": 16, "seed": seed})
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = TinyGPT(config).cpu()
    train_data, heldout_data, missing_data = make_splits(style)
    eval_data = heldout_data + missing_data
    initial_eval = evaluate(model, eval_data)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    rng = random.Random(seed + 1000)
    curve, target_tokens = [], 0
    start = time.perf_counter()
    update = 0
    for epoch in range(epochs):
        order = list(range(len(train_data)))
        rng.shuffle(order)
        losses = []
        for offset in range(0, len(order), batch_size):
            if update % 20 == 0 and available_mib() < 2560:
                raise MemoryError(f"MemAvailable below 2560 MiB at update {update}")
            batch = [train_data[i] for i in order[offset:offset + batch_size]]
            tokens, mask = collate(batch)
            target_tokens += int(mask.sum())
            optimizer.zero_grad(set_to_none=True)
            loss = masked_sft_loss(model, tokens, mask)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite SFT loss at update {update}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update += 1
            losses.append(float(loss.detach()))
        if epoch in {0, 1, 4, 9} or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            curve.append({"epoch": epoch + 1, "mean_batch_loss_nats": sum(losses) / len(losses)})
    seconds = time.perf_counter() - start
    result = {"style": style, "seed": seed, "epochs": epochs, "updates": update,
              "target_tokens_processed": target_tokens,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "initial_eval": initial_eval, "train_eval": evaluate(model, train_data),
              "heldout_and_missing_eval": evaluate(model, eval_data),
              "training_curve": curve, "training_seconds": seconds,
              "target_tokens_per_second": target_tokens / max(seconds, 1e-12),
              "optimizer_state_bytes": sum(t.numel() * t.element_size()
                  for state in optimizer.state.values() for t in state.values() if isinstance(t, torch.Tensor)),
              "format_target_tokens_per_training_epoch": sum(len(encode_record(item)[0]) - 6 for item in train_data)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    args = parser.parse_args()
    if not 1 <= args.epochs <= 120 or not 1 <= args.batch_size <= 32:
        raise SystemExit("epochs must be 1..120 and batch size 1..32")
    if available_mib() < 2560:
        raise SystemExit("MemAvailable below 2560 MiB guard")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base = json.loads(args.config.read_text(encoding="utf-8"))
    methods = ["concise", "worked"]
    all_runs = [train_one(base, style, seed, args.epochs, args.batch_size)
                for style in methods for seed in args.seeds]
    train_pairs = {(r["x"], r["y"]) for r in make_splits("concise")[0] if r["answerable"]}
    heldout_pairs = {(r["x"], r["y"]) for r in make_splits("concise")[1]}
    all_pairs = [(x, y) for x in range(10) for y in range(10)]
    train_sums = {x + y for x, y in train_pairs}
    heldout_sums = {x + y for x, y in heldout_pairs}
    result = {"schema_version": 1, "experiment": "tiny_synthetic_arithmetic_sft_and_abstention",
              "token_ids": {"PAD": PAD, "BOS": BOS, "USER": USER, "ASSISTANT": ASSISTANT,
                            "ADD": ADD, "MISSING": MISSING, "EOS": EOS, "ABSTAIN": ABSTAIN,
                            "OPERAND_BASE": OPERAND_BASE, "ANSWER_BASE": ANSWER_BASE,
                            "ONES_TAG": ONES_TAG, "CARRY_TAG": CARRY_TAG, "TENS_TAG": TENS_TAG},
              "data_audit": {"unique_pair_count": len(all_pairs), "train_pair_count": len(train_pairs),
                             "heldout_pair_count": len(heldout_pairs), "pair_overlap": len(train_pairs & heldout_pairs),
                             "duplicate_train_pairs": len(train_pairs) - len(set(train_pairs)),
                             "unseen_heldout_sum_values": sorted(heldout_sums - train_sums),
                             "training_missing_count": 10, "heldout_missing_count": 10,
                             "answer_distribution": dict(Counter(x + y for x, y in train_pairs)),
                             "license": "hand-authored/generated synthetic arithmetic; no third-party data"},
              "verifier_audit": {"all_100_sums_verified": all(
                                     exact_verifier(make_record(x, y, "concise", "audit"),
                                                    [ANSWER_BASE + sum((x, y)), EOS])["correct"]
                                     for x, y in all_pairs),
                                 "all_missing_inputs_require_abstention": all(r["x"] is None or r["y"] is None
                                     for r in make_splits("concise")[2])},
              "config": {"base": base, "vocab_size": 40, "context_length": 16,
                         "loss": "cross entropy over assistant completion tokens only; EOS included",
                         "optimizer": "AdamW lr 0.003, weight_decay 0, global grad clip 1.0",
                         "training_examples_per_epoch": 90, "batch_size": args.batch_size,
                         "epochs": args.epochs, "seeds": args.seeds},
              "runs": all_runs,
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                              "cuda_available": torch.cuda.is_available(), "threads": torch.get_num_threads(),
                              "mem_available_mib_at_start": available_mib(),
                              "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None},
              "limitations": ["synthetic one-digit arithmetic is not natural-language reasoning",
                              "step-labelled digit/carry tokens are an abstract output format, not natural-language chain-of-thought",
                              "no code synthesis, preference optimization or online RL run",
                              "three seeds and small split are preliminary; no frontier capability claim"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "runs": [
        {"style": r["style"], "seed": r["seed"], "updates": r["updates"],
         "train_accuracy": r["train_eval"]["exact_task_accuracy"],
         "heldout_accuracy": r["heldout_and_missing_eval"]["answerable_exact_accuracy"],
         "abstention_recall": r["heldout_and_missing_eval"]["abstention_recall"],
         "false_abstention_rate": r["heldout_and_missing_eval"]["false_abstention_rate"]}
        for r in all_runs], "peak_rss_bytes": result["peak_rss_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
