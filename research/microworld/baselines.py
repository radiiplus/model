"""Small deterministic baselines for generated microworld examples."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from research.microworld.language import answer_text
from research.microworld.oracle import answer_query
from research.microworld.splits import stable_hash


def direct_oracle(row: dict[str, Any], world: dict[str, Any]) -> dict[str, Any]:
    return answer_query(world, row["oracle_state"], row["query"])


def template_system(row: dict[str, Any], world: dict[str, Any]) -> dict[str, Any]:
    answer = direct_oracle(row, world)
    return {"ground_truth": answer,
            "answer_text": answer_text(answer, row["query"], world, row["construction_id"])}


def build_retrieval_index(training_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["prompt"]: {"ground_truth": row["ground_truth"],
                            "answer_text": row["answer_text"]}
            for row in training_rows}


def build_majority_index(training_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    values: dict[tuple[str, str], dict[str, Any]] = {}
    for row in training_rows:
        key = row["query_type"]
        signature = stable_hash(row["ground_truth"])
        counts[key][signature] += 1
        global_counts[signature] += 1
        values[(key, signature)] = row["ground_truth"]
        values[("__global__", signature)] = row["ground_truth"]
    result = {key: values[(key, counts[key].most_common(1)[0][0])]
              for key in counts}
    result["__global__"] = values[("__global__", global_counts.most_common(1)[0][0])]
    return result


def evaluate(worlds: list[dict[str, Any]], examples: list[dict[str, Any]]) -> dict[str, Any]:
    worlds_by_id = {row["world_id"]: row for row in worlds}
    train = [row for row in examples if row["split"] == "train"]
    retrieval = build_retrieval_index(train)
    majority = build_majority_index(train)
    summary: dict[str, Any] = {}

    for split in sorted({row["split"] for row in examples}):
        rows = [row for row in examples if row["split"] == split]
        methods: dict[str, list[tuple[dict[str, Any] | None, str | None]]] = {
            "direct_oracle": [], "template_system": [], "exact_retrieval": [],
            "majority_answer": [],
        }
        for row in rows:
            oracle_value = direct_oracle(row, worlds_by_id[row["world_id"]])
            methods["direct_oracle"].append((oracle_value, None))
            template = template_system(row, worlds_by_id[row["world_id"]])
            methods["template_system"].append((template["ground_truth"], template["answer_text"]))
            found = retrieval.get(row["prompt"])
            methods["exact_retrieval"].append(
                (found["ground_truth"], found["answer_text"]) if found else (None, None))
            methods["majority_answer"].append(
                (majority.get(row["query_type"], majority["__global__"]), None))

        split_result = {"examples": len(rows), "methods": {}}
        for name, predictions in methods.items():
            structural = sum(prediction == row["ground_truth"]
                             for row, (prediction, _) in zip(rows, predictions))
            text_examples = sum(answer is not None for _, answer in predictions)
            text = sum(answer == row["answer_text"]
                       for row, (_, answer) in zip(rows, predictions) if answer is not None)
            split_result["methods"][name] = {
                "exact_structural_accuracy": structural / max(1, len(rows)),
                "exact_answer_text_accuracy": text / text_examples if text_examples else None,
                "answer_text_examples": text_examples,
                "covered": sum(prediction is not None for prediction, _ in predictions),
            }
        summary[split] = split_result
    return {"evaluation": "exact oracle output equality; no neural model involved",
            "training_rows": len(train), "partitions": summary}
