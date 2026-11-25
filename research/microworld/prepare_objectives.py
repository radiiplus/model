"""Prepare train-only label vocabularies and comparable objective manifests."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from research.interaction.tokenizer import Tokenizer
from research.microworld.objectives import TASKS, label_vocabulary, token_labels


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def prepare(dataset: Path, tokenizer_path: Path, output: Path | None = None) -> dict[str, Any]:
    token = Tokenizer.load(tokenizer_path)
    train_path = dataset / "annotations_train.jsonl.gz"
    vocab_sets = {task: {"O"} for task in TASKS}
    annotation_counts: Counter[str] = Counter()
    structural_labels = 0
    train_examples = 0
    train_tokens = 0
    training_vocab_sets = {task: {"O"} for task in TASKS}
    for row in read_jsonl_gz(train_path):
        vocab = label_vocabulary(row["annotations"])
        for task in TASKS:
            vocab_sets[task].update(vocab[task])
            training_vocab_sets[task].update(vocab[task])
        annotation_counts.update(item["type"] for item in row["annotations"])
        structural_labels += len(row["annotations"])
        train_examples += 1
        train_tokens += len(row["token_ids"])
    vocabularies = {task: ["O"] + sorted(values - {"O"})
                    for task, values in vocab_sets.items()}
    token_supervision = Counter()
    examples_by_split = {"train": train_examples}
    max_sequence_tokens = 0
    pilot_input_tokens = 0
    pilot_language_targets = 0
    pilot_token_supervision = Counter()
    pilot_examples = 64
    pilot_seed = 20261125
    pilot_indices = random.Random(pilot_seed).sample(range(train_examples), pilot_examples)
    pilot_index_set = set(pilot_indices)
    pilot_example_ids = {}
    for train_index, row in enumerate(read_jsonl_gz(train_path)):
        if len(row["token_ids"]) > 1024:
            raise ValueError("train example exceeds Foundation context length")
        max_sequence_tokens = max(max_sequence_tokens, len(row["token_ids"]))
        labels = token_labels(row, vocabularies)
        row_counts = {task: sum(value != -100 for value in values)
                      for task, values in labels.items()}
        token_supervision.update(row_counts)
        if train_index in pilot_index_set:
            pilot_input_tokens += len(row["token_ids"])
            pilot_language_targets += max(0, len(row["token_ids"]) - 1)
            pilot_token_supervision.update(row_counts)
            pilot_example_ids[train_index] = row["example_id"]
    for split in ("validation", "iid_test", "compositional_test",
                  "counterfactual_test", "grammar_test"):
        examples_by_split[split] = 0
        for row in read_jsonl_gz(dataset / f"annotations_{split}.jsonl.gz"):
            if row["tokenizer_digest"] != token.digest:
                raise ValueError(f"tokenizer digest mismatch in {split}")
            if len(row["token_ids"]) > 1024:
                raise ValueError(f"{split} example exceeds Foundation context length")
            max_sequence_tokens = max(max_sequence_tokens, len(row["token_ids"]))
            examples_by_split[split] += 1
            heldout_vocab = label_vocabulary(row["annotations"])
            for task in TASKS:
                unseen = set(heldout_vocab[task]) - training_vocab_sets[task]
                if unseen:
                    raise ValueError(
                        f"{split} has labels absent from train for {task}: {sorted(unseen)}")
            labels = token_labels(row, vocabularies)
            token_supervision.update({task: sum(value != -100 for value in values)
                                      for task, values in labels.items()})

    width = 256
    projection_parameters = width * width + width
    task_head_parameters = sum(width * len(labels) + len(labels)
                               for labels in vocabularies.values())
    auxiliary_parameters = projection_parameters + task_head_parameters
    manifest = {
        "schema": 1, "dataset": str(dataset),
        "dataset_manifest_sha256": json.loads(
            (dataset / "manifest.json").read_text())["manifest_sha256"],
        "tokenizer_path": str(tokenizer_path), "tokenizer_digest": token.digest,
        "tokenizer_sha256": file_sha(tokenizer_path),
        "training_examples": train_examples, "training_input_tokens": train_tokens,
        "training_structural_labels": structural_labels,
        "predeclared_pilot": {
            "updates": pilot_examples, "examples": pilot_examples,
            "sampling": "uniform sample of distinct training row indices, fixed seed",
            "seed": pilot_seed,
            "training_row_indices": pilot_indices,
            "example_ids_in_presentation_order": [pilot_example_ids[index]
                                                   for index in pilot_indices],
            "input_tokens": pilot_input_tokens,
            "language_target_tokens": pilot_language_targets,
            "structural_token_labels_by_task": dict(sorted(pilot_token_supervision.items())),
            "structural_token_labels_total": sum(pilot_token_supervision.values()),
        },
        "token_level_supervision_by_task_all_splits": dict(sorted(token_supervision.items())),
        "examples_by_split": examples_by_split,
        "maximum_sequence_tokens": max_sequence_tokens,
        "structural_labels_by_type": dict(sorted(annotation_counts.items())),
        "label_classes_by_task": {task: len(labels)
                                  for task, labels in vocabularies.items()},
        "label_vocabulary": vocabularies,
        "backbone_parameters": 4_470_272,
        "auxiliary_projection_parameters": projection_parameters,
        "auxiliary_task_head_parameters": task_head_parameters,
        "auxiliary_parameters": auxiliary_parameters,
        "total_parameters_with_dense_heads": 4_470_272 + auxiliary_parameters,
        "sequence_limit": 1024,
        "all_sequences_fit_context": True,
        "heldout_label_vocab_covered_by_train": True,
        "token_label_conflicts_found": False,
        "training_signal_comparison": {
            "arm_a_language_tokens": "one next-token target per non-final input token",
            "arm_b_structural_labels": structural_labels,
            "arm_c_both": "both counts above; not matched label quantity",
        },
        "training_performed": False,
    }
    raw = json.dumps(manifest, sort_keys=True, indent=2).encode()
    manifest["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    if output:
        output.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("research/microworld/dataset_v1"))
    parser.add_argument("--tokenizer", type=Path, default=Path(
        "research/interaction/data/ready7/tokenizer.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset, args.tokenizer, args.output),
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
