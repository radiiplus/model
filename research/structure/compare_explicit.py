"""Audit controls and compare free JSON output with fixed-slot structure output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from research.foundation.model import Foundation
from research.interaction.tokenizer import Tokenizer
from research.structure import explicit_head, train


SPLITS = ("validation", "iid_test", "compositional_test", "pattern_test",
          "counterfactual_test")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--head-structure", type=Path, required=True)
    parser.add_argument("--head-combined", type=Path, required=True)
    parser.add_argument("--head-structure-checkpoint", type=Path, required=True)
    parser.add_argument("--head-combined-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = {
        "json": load(args.baseline),
        "slots": load(args.head_structure),
        "combined": load(args.head_combined),
    }
    same_weights = len({item["starting_weights_sha256"] for item in reports.values()}) == 1
    same_order = len({item["training_sequence_sha256"] for item in reports.values()}) == 1
    json_report = reports["json"]
    json_history = json_report["training"]
    json_prompt_tokens = sum(
        row["input_tokens"] - row["answer_tokens"] - row["structure_tokens"]
        for row in json_history) // 2
    prompt_tokens = {"json": json_prompt_tokens,
                     "slot_structure": reports["slots"]["prompt_tokens_seen"],
                     "slot_combined": reports["combined"]["prompt_tokens_seen"]}
    same_prompt_budget = len(set(prompt_tokens.values())) == 1

    baseline_config = load(args.config)
    token = Tokenizer.load(train.TOKENIZER)
    direct_head_metrics = {}
    for key, checkpoint_path in (("slots", args.head_structure_checkpoint),
                                 ("combined", args.head_combined_checkpoint)):
        model = explicit_head.FoundationWithStructuralHead(baseline_config["model"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        report = {}
        for split in SPLITS:
            rows = explicit_head.read_examples(args.data, split)
            report[split] = explicit_head.structural_metrics(model, rows, token, 1,
                int(baseline_config["train"].get("eval_seed", 2026)))
        direct_head_metrics[key] = report

    paired = {}
    selection_matches = True
    full_selection_matches = True
    for split in SPLITS:
        json_full_hash = json_report["validation_after"][split]["selected_examples_sha256"]
        generated = json_report["validation_after"][split]["structure_generation_samples"]
        json_sample = generated[0] if generated else None
        per_arm = {}
        for key in ("slots", "combined"):
            metric = direct_head_metrics[key][split]
            full_hash = reports[key]["validation_after"][split]["selected_examples_sha256"]
            full_selection_matches &= json_full_hash == full_hash
            matches = bool(json_sample) and metric["selected_examples_sha256"] == train.digest(
                [json_sample["example_id"]])
            selection_matches &= matches
            per_arm[key] = {"selected_same_example_as_json": matches,
                "exact_structure": metric["exact_structure"],
                "exact_transition": metric["exact_transition"],
                "constraint_violations_per_example": metric["constraint_violations_per_example"]}
        paired[split] = {"json_generation": json_sample, "slot_heads": per_arm,
                         "entity_and_edge_metrics_12_examples": {
            key: reports[key]["validation_after"][split] for key in ("slots", "combined")}}

    schedules = {key: [row["learning_rate_used"] for row in item["training"]]
                 for key, item in reports.items()}
    same_schedule = len({tuple(values) for values in schedules.values()}) == 1
    dataset_hashes = {"json": json_report["dataset"],
        "slot_structure": reports["slots"]["dataset_manifest_sha256"],
        "slot_combined": reports["combined"]["dataset_manifest_sha256"]}
    tokenizer_hashes = {"json": json_report["tokenizer"],
        "slot_structure": reports["slots"]["tokenizer_sha256"],
        "slot_combined": reports["combined"]["tokenizer_sha256"]}
    controls = {"same_starting_weights": same_weights, "same_training_order": same_order,
        "same_prompt_token_budget": same_prompt_budget, "prompt_tokens": prompt_tokens,
        "same_learning_rate_trajectory": same_schedule, "learning_rates": schedules,
        "same_dataset_manifest": len(set(dataset_hashes.values())) == 1,
        "dataset_manifest_hashes": dataset_hashes,
        "same_tokenizer": len(set(tokenizer_hashes.values())) == 1,
        "tokenizer_hashes": tokenizer_hashes,
        "same_full_evaluation_examples": full_selection_matches,
        "same_paired_evaluation_examples": selection_matches,
        "optimizer_state_restored": False,
        "updates_per_arm": {key: item["updates"] for key, item in reports.items()},
        "examples_per_arm": {key: item["examples_consumed"] for key, item in reports.items()},
        "active_supervision": {
            "json_structure_tokens": json_report["structure_target_tokens_seen"],
            "slot_structure_elements": reports["slots"]["target_elements_seen"],
            "slot_combined_elements_plus_answer_tokens": {
                "slot_elements": reports["combined"]["target_elements_seen"],
                "answer_tokens": sum(item["answer_tokens"] for item in
                                      reports["json"]["training"])}}}

    summary = {"schema": 1, "experiment": "foundation-explicit-structural-head-pilot",
        "decision": "NO-GO for further training at this scale",
        "conclusion": "The explicit head did not achieve exact structure or transition recovery in this 12-update, one-seed pilot. JSON free generation produced no valid structure. The bounded result does not support scaling either objective.",
        "parameters": {"foundation": reports["slots"]["foundation_parameters"],
            "explicit_head": reports["slots"]["auxiliary_parameters"],
            "slot_total": reports["slots"]["parameters_total"]},
        "controls": controls,
        "arms": {
            "json": {"validation_answer_nll": json_report["validation_after"]["validation"]["answer_nll"],
                "validation_structure_nll": json_report["validation_after"]["validation"]["structure_nll"],
                "teacher_forced_json_valid": json_report["validation_after"]["validation"]["structure_teacher_forced_json_valid"],
                "free_generated_json_valid": json_report["validation_after"]["validation"]["structure_greedy_json_valid"],
                "free_generated_exact_structure": json_report["validation_after"]["validation"]["structure_greedy_exact"]},
            "slot_structure": {"validation": reports["slots"]["validation_after"]["validation"],
                "answer_nll": reports["slots"]["language_after"]["validation"]["answer_nll"],
                "answer_semantic_accuracy": reports["slots"]["language_after"]["validation"]["answer_semantic_accuracy"]},
            "slot_combined": {"validation": reports["combined"]["validation_after"]["validation"],
                "answer_nll": reports["combined"]["language_after"]["validation"]["answer_nll"],
                "answer_semantic_accuracy": reports["combined"]["language_after"]["validation"]["answer_semantic_accuracy"]}},
        "paired_free_generation_examples": paired}
    if not all((same_weights, same_order, same_prompt_budget, same_schedule,
                controls["same_dataset_manifest"], controls["same_tokenizer"],
                full_selection_matches, selection_matches)):
        raise ValueError("pilot controls do not match")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"decision": summary["decision"], "controls_pass": True,
        "comparison": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
