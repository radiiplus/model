"""Prepare and run a matched value-based structural sampler experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import torch

from research.foundation import trainer as foundation_trainer
from research.foundation.model import Foundation
from research.interaction.tokenizer import Tokenizer
from research.prototype import trainer as guard
from research.structure import train
from research.structure.sampler import MODES, build_plan


HERE = Path(__file__).resolve().parent
INITIAL = HERE.parent / "foundation" / "data" / "learn.pt"
OUT = HERE / "value_curriculum"
CONFIG = HERE / "configs" / "arm_b.json"
UPDATES = 24
POOL_SIZE = 8
HOLDOUT_WORLDS = 6
SEED = 2025
REFERENCE_SEED = 64109


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
    temp.replace(path)


def hash_ids(ids: list[str]) -> str:
    raw = json.dumps(ids, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def structural_loss(model: Foundation, row: dict[str, Any], token: Tokenizer) -> float:
    with torch.no_grad():
        return float(train.losses(model, [row], token)["structure"])


def prepare(output: Path, initial: Path = INITIAL, updates: int = UPDATES,
            pool_size: int = POOL_SIZE, holdout_worlds: int = HOLDOUT_WORLDS,
            seed: int = SEED) -> dict[str, Any]:
    phase_start = time.monotonic()
    if guard.available_mib() < guard.MEMORY_GUARD_MIB:
        raise MemoryError("effective available memory below the existing 2560 MiB guard")
    guard.configure_torch()
    torch.manual_seed(seed)
    train_rows = train.read_examples(train.DATA, "train")
    worlds = sorted({row["world_id"] for row in train_rows})
    if not 1 <= holdout_worlds < len(worlds):
        raise ValueError("reference holdout world count is outside the training corpus")
    rng = random.Random(seed + 7701)
    holdout = sorted(rng.sample(worlds, holdout_worlds))
    holdout_set = set(holdout)
    holdout_rows = [row for row in train_rows if row["world_id"] in holdout_set]
    allowed = [index for index, row in enumerate(train_rows) if row["world_id"] not in holdout_set]
    token = Tokenizer.load(train.TOKENIZER)
    costs = [0] * len(train_rows)
    for index in allowed:
        answer, structure, _, _, _ = train.encode_pair(train_rows[index], token, 1024)
        costs[index] = len(answer) + len(structure) - 2
    plan = build_plan(costs, train_rows, allowed, updates, pool_size, seed + 1)
    plan["holdout_world_ids"] = holdout
    plan["available_training_world_ids"] = sorted(set(worlds) - holdout_set)
    plan["holdout_example_count"] = len(holdout_rows)
    plan["train_pool_example_count"] = len(allowed)
    plan["input_token_budget_total"] = sum(plan["budget_tokens_per_update"])
    plan["dataset_manifest_sha256"] = train.sha(train.DATA / "manifest.json")
    plan["initial_checkpoint_sha256"] = train.sha(initial)
    source = torch.load(initial, map_location="cpu", weights_only=True)
    model_config = source["config"]["model"]
    model = Foundation(model_config)
    model.load_state_dict(source["model"])
    model.eval()
    candidate_indices = sorted({index for item in plan["pools"] for index in item["indices"]})
    initial_losses = {train_rows[index]["example_id"]:
                      structural_loss(model, train_rows[index], token)
                      for index in candidate_indices}
    plan_path = output / "candidate_plan.json"
    initial_path = output / "initial_losses.json"
    atomic_json(plan_path, plan)
    atomic_json(initial_path, initial_losses)
    report = {"schema": 1, "updates": updates, "pool_size": pool_size,
        "seed": seed, "objective": "structural_recovery", "training_corpus_rows": len(allowed),
        "training_world_count": len(worlds) - holdout_worlds,
        "reference_holdout_rows": len(holdout_rows), "reference_holdout_world_ids": holdout,
        "candidate_unique_rows": len(candidate_indices),
        "candidate_plan_sha256": train.sha(plan_path),
        "initial_loss_sha256": train.sha(initial_path),
        "input_token_budget_per_update": plan["budget_tokens_per_update"],
        "input_token_budget_total": plan["input_token_budget_total"],
        "matched_exactly_per_update": True,
        "preparation_wall_seconds": time.monotonic() - phase_start,
        "tokenizer_sha256": train.sha(train.TOKENIZER),
        "dataset_manifest_sha256": plan["dataset_manifest_sha256"],
        "initial_checkpoint_sha256": plan["initial_checkpoint_sha256"]}
    atomic_json(output / "prepare_report.json", report)
    return {**report, "holdout_rows": holdout_rows, "rows": train_rows,
            "candidate_indices": candidate_indices, "source": source,
            "tokenizer": token, "output": output}


def fit_rho_reference(prepared: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    phase_start = time.monotonic()
    output: Path = prepared["output"]
    initial = prepared["source"]
    model = Foundation(config["model"])
    model.load_state_dict(initial["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["train"]["rate"]),
                                  weight_decay=float(config["train"]["penalty"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: foundation_trainer.rate(step, config["train"]))
    optimizer.load_state_dict(initial["optimizer"])
    scheduler.load_state_dict(initial["scheduler"])
    holdout = prepared["holdout_rows"]
    order = train.Order(len(holdout), REFERENCE_SEED)
    history = []
    model.train()
    for step in range(UPDATES):
        selected = [holdout[index] for index in order.take(1)]
        result = train.losses(model, selected, prepared["tokenizer"])
        objective = result["structure"]
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["clip"]))
        lr = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
        history.append({"step": step + 1, "example_id": selected[0]["example_id"],
            "structure_loss": float(objective.detach()), "input_tokens": int(result["input_tokens"]),
            "structure_tokens": int(result["structure_tokens"]), "gradient_norm": float(grad),
            "learning_rate_used": lr})
    model.eval()
    candidate_rows = [prepared["rows"][index] for index in prepared["candidate_indices"]]
    reference_losses = {row["example_id"]: structural_loss(model, row, prepared["tokenizer"])
                        for row in candidate_rows}
    checkpoint_path = output / "rho_reference.pt"
    foundation_trainer.save(checkpoint_path, {"schema": 1, "model": model.state_dict(),
        "model_config": config["model"], "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "sampler": order.state(),
        "step": UPDATES, "seed": REFERENCE_SEED, "history": history,
        "holdout_world_ids": prepared["reference_holdout_world_ids"],
        "candidate_ids_sha256": hash_ids([row["example_id"] for row in candidate_rows])})
    loss_path = output / "rho_reference_losses.json"
    atomic_json(loss_path, reference_losses)
    report = {"schema": 1, "method": "RHO-LOSS-style reducible structural holdout loss",
        "reference": "same Foundation architecture, initialized from update-800 weights, fine-tuned on reserved training worlds only",
        "reference_updates": UPDATES, "reference_parameters": model.params,
        "reference_fit_and_scoring_wall_seconds": time.monotonic() - phase_start,
        "reference_examples_seen": len(history),
        "reference_input_tokens_seen": sum(row["input_tokens"] for row in history),
        "holdout_world_ids": prepared["reference_holdout_world_ids"],
        "checkpoint_sha256": train.sha(checkpoint_path),
        "reference_losses_sha256": train.sha(loss_path),
        "training_history": history,
        "faithfulness_note": "Uses the published RHO score current structural loss minus an irreducible-loss estimate from a separate model trained on a disjoint training-only holdout. The reference model is only 24 updates and is not claimed to have converged; this is reported as RHO-LOSS-style, not an exact replication."}
    atomic_json(output / "rho_reference_report.json", report)
    return report


def run_modes(prepared: dict[str, Any], reference_report: dict[str, Any],
              config_path: Path = CONFIG, initial: Path = INITIAL) -> dict[str, Any]:
    output: Path = prepared["output"]
    base = train.load_config(config_path)
    base["condition"] = "structure"
    base["restore_optimizer"] = True
    base["train"]["steps"] = UPDATES
    base["train"]["batch"] = 1
    result = {"schema": 1, "experiment": "value-driven-structural-selection-v1",
        "starting_checkpoint_sha256": train.sha(initial),
        "dataset_manifest_sha256": train.sha(train.DATA / "manifest.json"),
        "tokenizer_sha256": train.sha(train.TOKENIZER),
        "updates": UPDATES, "input_token_budget_total": prepared["input_token_budget_total"],
        "sampler_modes": {}, "reference_report": "rho_reference_report.json",
        "initial_loss_file": "initial_losses.json", "candidate_plan": "candidate_plan.json"}
    for mode in MODES:
        arm_config = json.loads(json.dumps(base))
        arm_config["sampling"] = {"mode": mode,
            "plan_path": str(output / "candidate_plan.json"),
            "initial_losses_path": str(output / "initial_losses.json"),
            "reference_losses_path": str(output / "rho_reference_losses.json") if mode == "rho_loss" else None}
        config_file = output / f"{mode}.config.json"
        atomic_json(config_file, arm_config)
        checkpoint = output / f"{mode}.pt"
        report_file = output / f"{mode}.json"
        started = time.monotonic()
        report = train.run(arm_config, "b", initial, checkpoint, resume=None,
                           stop=None)
        report["wall_seconds_including_selection"] = time.monotonic() - started
        atomic_json(report_file, report)
        result["sampler_modes"][mode] = {"checkpoint": checkpoint.name,
            "checkpoint_sha256": report["checkpoint_sha256"], "report": report_file.name,
            "starting_weights_sha256": report["starting_weights_sha256"],
            "input_tokens_seen": report["input_tokens_seen"],
            "active_objective_tokens_seen": report["active_objective_tokens_seen"],
            "updates": report["updates"], "selection_seconds": report["selection_seconds"],
            "wall_seconds_including_selection": report["wall_seconds_including_selection"],
            "validation_before": report["validation_before"]["validation"],
            "validation_after": report["validation_after"]["validation"],
            "training_sequence_sha256": report["training_sequence_sha256"],
            "training_ids": [ident for row in report["training"] for ident in row["example_ids"]]}
    atomic_json(output / "comparison.json", result)
    analyze(output)
    return result


def analyze(output: Path) -> dict[str, Any]:
    comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    preparation = json.loads((output / "prepare_report.json").read_text(encoding="utf-8"))
    reference = json.loads((output / "rho_reference_report.json").read_text(encoding="utf-8"))
    modes = {}
    for mode in MODES:
        report = json.loads((output / f"{mode}.json").read_text(encoding="utf-8"))
        metrics = {}
        for split, after in report["validation_after"].items():
            before = report["validation_before"][split]
            gain = before["structure_nll"] - after["structure_nll"]
            mode_wall = report["wall_seconds_including_selection"]
            amortized_wall = mode_wall + preparation.get("preparation_wall_seconds", 0.0) / len(MODES)
            if mode == "rho_loss":
                amortized_wall += reference.get("reference_fit_and_scoring_wall_seconds", 0.0)
            metrics[split] = {"structure_nll_before": before["structure_nll"],
                "structure_nll_after": after["structure_nll"], "structure_nll_gain": gain,
                "gain_per_1k_input_tokens": gain / report["input_tokens_seen"] * 1000,
                "gain_per_update": gain / report["updates"],
                "gain_per_target_run_wall_second": gain / mode_wall,
                "gain_per_amortized_compute_second": gain / amortized_wall,
                "structure_token_accuracy_before": before["structure_token_accuracy"],
                "structure_token_accuracy_after": after["structure_token_accuracy"],
                "teacher_forced_json_valid": after["structure_teacher_forced_json_valid"],
                "answer_semantic_accuracy": after["answer_semantic_accuracy"]}
        coverage = [row["selection"]["coverage_group"] for row in report["training"]]
        modes[mode] = {"checkpoint_sha256": report["checkpoint_sha256"],
            "starting_weights_sha256": report["starting_weights_sha256"],
            "updates": report["updates"], "examples_consumed": report["examples_consumed"],
            "input_tokens_seen": report["input_tokens_seen"],
            "active_objective_tokens_seen": report["active_objective_tokens_seen"],
            "fit_seconds": report["fit_seconds"],
            "wall_seconds_including_selection": report["wall_seconds_including_selection"],
            "selection_seconds": report["selection_seconds"],
            "peak_rss_bytes": report["peak_rss_bytes"],
            "unique_training_examples": len({ident for row in report["training"]
                                               for ident in row["example_ids"]}),
            "unique_worlds": len({row["example_ids"][0].split(":")[0]
                                   for row in report["training"]}),
            "query_surface_groups_covered": len(set(coverage)),
            "per_split": metrics}
    value = {"schema": 1, "experiment": comparison["experiment"],
        "starting_checkpoint_sha256": comparison["starting_checkpoint_sha256"],
        "dataset_manifest_sha256": comparison["dataset_manifest_sha256"],
        "tokenizer_sha256": comparison["tokenizer_sha256"],
        "identical_starting_weights": len({item["starting_weights_sha256"]
                                            for item in modes.values()}) == 1,
        "identical_updates": len({item["updates"] for item in modes.values()}) == 1,
        "identical_input_token_budgets": len({item["input_tokens_seen"]
                                              for item in modes.values()}) == 1,
        "input_token_budget_total": modes["uniform"]["input_tokens_seen"],
        "shared_preparation_wall_seconds": preparation.get("preparation_wall_seconds"),
        "rho_reference_fit_and_scoring_wall_seconds": reference.get(
            "reference_fit_and_scoring_wall_seconds"),
        "total_experiment_wall_seconds": (preparation.get("preparation_wall_seconds", 0.0)
            + reference.get("reference_fit_and_scoring_wall_seconds", 0.0)
            + sum(item["wall_seconds_including_selection"] for item in modes.values())),
        "samplers": modes,
        "rho_reference_extra_compute": {"reference_parameters": reference["reference_parameters"],
            "reference_updates": reference["reference_examples_seen"],
            "candidate_reference_scored_examples": len(json.loads(
                (output / "rho_reference_losses.json").read_text(encoding="utf-8"))),
            "reference_wall_seconds_included_in_rho_target_run_wall": False,
            "checkpoint_sha256": reference["checkpoint_sha256"]},
        "uncertainty": "one paired seed only; Batch 4 showed no positive capability signal; validation samples share underlying worlds",
        "source_reports": {mode: f"{mode}.json" for mode in MODES}}
    atomic_json(output / "comparison_metrics.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--initial", type=Path, default=INITIAL)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    prepared = prepare(args.output, args.initial)
    base = train.load_config(CONFIG)
    rho = fit_rho_reference(prepared, base)
    if args.prepare_only:
        print(json.dumps({"prepared": str(args.output), "rho_reference": rho["checkpoint_sha256"]}, indent=2))
        return
    result = run_modes(prepared, rho, CONFIG, args.initial)
    metrics = json.loads((args.output / "comparison_metrics.json").read_text(encoding="utf-8"))
    print(json.dumps({"output": str(args.output), "modes": {
        mode: {"input_tokens": value["input_tokens_seen"],
               "selection_seconds": value["selection_seconds"],
               "validation_structure_nll": value["validation_after"]["structure_nll"]}
        for mode, value in result["sampler_modes"].items()},
        "identical_input_token_budgets": metrics["identical_input_token_budgets"]}, indent=2))


if __name__ == "__main__":
    main()
