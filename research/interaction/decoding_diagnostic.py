"""Inference-only decoding experiments on the semantic correction benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import re
import resource
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from research.foundation.model import Foundation
from research.interaction import correction_pilot as pilot
from research.interaction.tokenizer import SPECIAL, Tokenizer
from research.prototype import trainer as guard


HERE = Path(__file__).resolve().parent
OUT = HERE / "decoding_diagnostic"
PARTITIONS = ("validation", "iid_test", "compositional_test", "counterfactual_test")
MAX_NEW_TOKENS = 40
SAMPLE_SEED = 9142026
MIN_NEW_TOKENS = 4
REPEAT_PENALTY = 1.2
NO_REPEAT_NGRAM = 3
TEMPERATURE = 0.8
TOP_P = 0.9
CONTRASTIVE_ALPHA = 0.6
CONTRASTIVE_K = 4

INDEPENDENT = {
    "greedy": {},
    "temperature_0.8_top_p_0.9": {"temperature": TEMPERATURE, "top_p": TOP_P},
    "repetition_penalty_1.2": {"repetition_penalty": REPEAT_PENALTY},
    "no_repeat_trigram": {"no_repeat_ngram": NO_REPEAT_NGRAM},
    "minimum_4_mask_end_eos": {"minimum_length": MIN_NEW_TOKENS},
    "contrastive_alpha_0.6_k4": {"contrastive_alpha": CONTRASTIVE_ALPHA,
                                  "contrastive_k": CONTRASTIVE_K},
}


def file_sha(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def stable_seed(example_id: str, decoder: str) -> int:
    raw = f"{SAMPLE_SEED}:{decoder}:{example_id}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)


def _apply_repeat_penalty(logits: torch.Tensor, history: list[int], penalty: float) -> None:
    if penalty <= 0:
        raise ValueError("repetition penalty must be positive")
    for ident in set(history):
        if logits[ident] >= 0:
            logits[ident] /= penalty
        else:
            logits[ident] *= penalty


def _blocked_ngram(history: list[int], n: int) -> set[int]:
    if n < 1:
        raise ValueError("no-repeat n-gram size must be positive")
    if len(history) + 1 < n:
        return set()
    prefix = tuple(history[-(n - 1):]) if n > 1 else ()
    banned = set()
    for start in range(len(history) - n + 1):
        gram = history[start:start + n]
        if tuple(gram[:-1]) == prefix:
            banned.add(gram[-1])
    return banned


def _sample_top_p(logits: torch.Tensor, temperature: float, top_p: float,
                  generator: torch.Generator) -> tuple[int, float]:
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("temperature and top-p are outside valid bounds")
    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    remove = torch.cumsum(sorted_probs, dim=-1) - sorted_probs >= top_p
    sorted_probs[remove] = 0
    filtered = torch.zeros_like(probs)
    filtered.scatter_(0, sorted_ids, sorted_probs)
    filtered /= filtered.sum()
    ident = int(torch.multinomial(filtered, 1, generator=generator))
    return ident, float(filtered[ident])


def _contrastive_choice(model: Foundation, ids: list[int], logits: torch.Tensor,
                        alpha: float, k: int) -> tuple[int, float]:
    if not 0 <= alpha <= 1 or k < 1:
        raise ValueError("contrastive search parameters are invalid")
    current = torch.tensor([ids], dtype=torch.long)
    previous = model.hidden(current)[0]
    probabilities = torch.softmax(logits, dim=-1)
    top_prob, top_ids = torch.topk(probabilities, min(k, probabilities.numel()))
    scores = []
    for probability, ident in zip(top_prob, top_ids):
        candidate_ids = torch.cat((current, ident.reshape(1, 1)), dim=1)
        candidate = model.hidden(candidate_ids)[0, -1]
        similarities = F.cosine_similarity(previous, candidate[None, :], dim=-1)
        degeneration = similarities.max()
        scores.append((1 - alpha) * probability - alpha * degeneration)
    index = int(torch.stack(scores).argmax())
    return int(top_ids[index]), float(top_prob[index])


def decode(model: Foundation, token: Tokenizer, prompt: str, example_id: str,
           name: str, config: dict[str, Any], limit: int = MAX_NEW_TOKENS) -> dict[str, Any]:
    prefix = token.chat([{"role": "user", "text": prompt}], bos=True, eos=False)
    ids = prefix + [SPECIAL["ASSISTANT"]]
    generated: list[int] = []
    content: list[int] = []
    traces = []
    stop = "token_limit"
    special_tokens = 0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(example_id, name))
    started = time.monotonic()
    with torch.inference_mode():
        for step in range(min(limit, model.context - len(ids))):
            current = torch.tensor([ids], dtype=torch.long)
            if config.get("contrastive_alpha") is not None:
                hidden = model.hidden(current)
                logits = F.linear(hidden[:, -1], model.head)[0]
            else:
                logits = model(current)[0, -1]
            raw_prob = torch.softmax(logits, dim=-1)
            eos_p = float(raw_prob[SPECIAL["EOS"]])
            end_p = float(raw_prob[SPECIAL["END"]])
            entropy = float(-(raw_prob * raw_prob.clamp_min(1e-30).log()).sum())
            top_id = int(logits.argmax())
            adjusted = logits.clone()
            if config.get("repetition_penalty") is not None:
                _apply_repeat_penalty(adjusted, generated,
                                      float(config["repetition_penalty"]))
            ngram = config.get("no_repeat_ngram")
            if ngram is not None:
                blocked = _blocked_ngram(generated, int(ngram))
                if blocked:
                    adjusted[list(blocked)] = -torch.inf
            if config.get("minimum_length") is not None and len(generated) < int(config["minimum_length"]):
                adjusted[SPECIAL["EOS"]] = -torch.inf
                adjusted[SPECIAL["END"]] = -torch.inf

            if config.get("contrastive_alpha") is not None:
                picked, picked_probability = _contrastive_choice(model, ids, logits,
                    float(config["contrastive_alpha"]), int(config["contrastive_k"]))
            elif config.get("temperature") is not None:
                picked, picked_probability = _sample_top_p(adjusted,
                    float(config["temperature"]), float(config["top_p"]), generator)
            else:
                picked = int(adjusted.argmax())
                picked_probability = float(torch.softmax(adjusted, dim=-1)[picked])
            traces.append({"step": step, "raw_top1": top_id,
                "raw_top1_probability": float(raw_prob[top_id]),
                "chosen_token": picked, "chosen_probability_after_decoder": picked_probability,
                "p_eos": eos_p, "p_end": end_p, "entropy_nats": entropy})
            generated.append(picked)
            if picked == SPECIAL["EOS"]:
                stop = "eos"
                break
            if picked == SPECIAL["END"]:
                stop = "end"
                break
            ids.append(picked)
            if picked in token.parts:
                content.append(picked)
            else:
                special_tokens += 1

    raw_text = b"".join(token.parts[item] for item in content).decode("utf-8", errors="replace")
    semantic = pilot.semantic_label(raw_text)
    exact = raw_text.strip().casefold().rstrip(".!?") == ""  # replaced at scoring time
    trigrams = [tuple(content[index:index + 3]) for index in range(max(0, len(content) - 2))]
    repeated_3gram = len(trigrams) != len(set(trigrams))
    adjacent_repeat = any(left == right for left, right in zip(content, content[1:]))
    values = {int(value) for value in re.findall(r"\b\d+\b", raw_text)}
    return {"example_id": example_id, "text": raw_text, "semantic": semantic,
        "generated_token_count": len(generated), "content_token_count": len(content),
        "termination_token_count": int(stop in {"eos", "end"}),
        "stop": stop, "special_tokens_in_body": special_tokens,
        "empty": not raw_text.strip(), "repeated_3gram": repeated_3gram,
        "adjacent_repeat": adjacent_repeat,
        "repetition_rate_3gram": (1 - len(set(trigrams)) / len(trigrams)) if trigrams else 0.0,
        "exact_response_match": exact, "numeric_details_correct": False,
        "gold": None, "correct": False, "trace": traces,
        "seconds": time.monotonic() - started}


def score_records(records: list[dict[str, Any]], rows_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    for record in records:
        row = rows_by_id[record["example_id"]]
        record["gold"] = row["answer_semantic"]
        record["correct"] = record["semantic"] == row["answer_semantic"]
        record["exact_response_match"] = record["text"].strip().casefold().rstrip(".!?") == \
            row["answer_target"].strip().casefold().rstrip(".!?")
        numbers = {int(value) for value in re.findall(r"\b\d+\b", record["text"])}
        record["numeric_details_correct"] = {row["effective_capacity"], row["demand"]} <= numbers
    count = len(records)
    covered = sum(record["semantic"] != "unknown" for record in records)
    correct = sum(record["correct"] for record in records)
    ended = sum(record["termination_token_count"] for record in records)
    stop_counts = {name: sum(record["stop"] == name for record in records)
                   for name in ("eos", "end", "token_limit")}
    return {"examples": count, "semantic_coverage": covered / max(1, count),
        "accuracy_when_covered": correct / max(1, covered),
        "overall_semantic_accuracy": correct / max(1, count),
        "empty_output_percentage": 100 * sum(record["empty"] for record in records) / max(1, count),
        "repetition_percentage": 100 * sum(record["repeated_3gram"] for record in records) / max(1, count),
        "adjacent_repeat_percentage": 100 * sum(record["adjacent_repeat"] for record in records) / max(1, count),
        "mean_repeated_3gram_rate": sum(record["repetition_rate_3gram"] for record in records) / max(1, count),
        "mean_generated_tokens": sum(record["generated_token_count"] for record in records) / max(1, count),
        "mean_content_tokens": sum(record["content_token_count"] for record in records) / max(1, count),
        "correct_termination_percentage": 100 * ended / max(1, count),
        "eos_termination_percentage": 100 * stop_counts["eos"] / max(1, count),
        "end_termination_percentage": 100 * stop_counts["end"] / max(1, count),
        "token_limit_percentage": 100 * stop_counts["token_limit"] / max(1, count),
        "exact_response_accuracy": sum(record["exact_response_match"] for record in records) / max(1, count),
        "numeric_detail_accuracy": sum(record["numeric_details_correct"] for record in records) / max(1, count),
        "stop_counts": stop_counts, "predictions": records}


def teacher_forced_specials(model: Foundation, token: Tokenizer,
                            splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    output = {}
    model.eval()
    with torch.inference_mode():
        for split in PARTITIONS:
            rows = pilot.selected_evaluation_rows(splits[split])
            first = {"p_eos": [], "p_end": [], "gold_first": [], "top1_special": []}
            at_end = {"p_end": [], "p_eos": [], "correct_top1": []}
            at_eos = {"p_eos": [], "p_end": [], "correct_top1": []}
            for row in rows:
                prefix = token.chat([{"role": "user", "text": row["input"]}],
                                    bos=True, eos=False)
                target = token.encode(row["answer_target"])
                if not target:
                    continue
                full = prefix + [SPECIAL["ASSISTANT"]] + target + [SPECIAL["END"], SPECIAL["EOS"]]
                logits = model(torch.tensor([full[:-1]], dtype=torch.long))[0]

                def probs(position: int) -> torch.Tensor:
                    return torch.softmax(logits[position], dim=-1)

                p = probs(len(prefix))
                first["p_eos"].append(float(p[SPECIAL["EOS"]]))
                first["p_end"].append(float(p[SPECIAL["END"]]))
                first["gold_first"].append(float(p[target[0]]))
                first["top1_special"].append(int(int(logits[len(prefix)].argmax())
                    in {SPECIAL["EOS"], SPECIAL["END"]}))
                end_position = len(prefix) + len(target)
                p_end = probs(end_position)
                at_end["p_end"].append(float(p_end[SPECIAL["END"]]))
                at_end["p_eos"].append(float(p_end[SPECIAL["EOS"]]))
                at_end["correct_top1"].append(int(int(logits[end_position].argmax())
                                                   == SPECIAL["END"]))
                eos_position = end_position + 1
                p_eos = probs(eos_position)
                at_eos["p_eos"].append(float(p_eos[SPECIAL["EOS"]]))
                at_eos["p_end"].append(float(p_eos[SPECIAL["END"]]))
                at_eos["correct_top1"].append(int(int(logits[eos_position].argmax())
                                                   == SPECIAL["EOS"]))
            output[split] = {
                "examples": len(rows),
                "first_answer_position": {key: sum(values) / max(1, len(values))
                    for key, values in first.items() if key != "top1_special"},
                "first_answer_top1_is_eos_or_end_percentage":
                    100 * sum(first["top1_special"]) / max(1, len(first["top1_special"])),
                "teacher_forced_end_position": {key: sum(values) / max(1, len(values))
                    for key, values in at_end.items()},
                "teacher_forced_eos_position": {key: sum(values) / max(1, len(values))
                    for key, values in at_eos.items()},
            }
    return output


def checkpoint_inventory(paths: list[Path]) -> list[dict[str, Any]]:
    inventory = []
    for path in paths:
        if path.is_file():
            inventory.append({"path": str(path), "bytes": path.stat().st_size,
                              "sha256": file_sha(path)})
    return inventory


def checkpoint_validation_scores(paths: list[Path], token: Tokenizer,
                                 rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores = []
    for path in paths:
        if not path.is_file():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        config = payload["config"]["model"]
        model = Foundation(config)
        model.load_state_dict(payload["model"])
        model.eval()
        loss = 0.0
        token_count = correct = examples = first_correct = 0
        first_gold_probability = first_eos_probability = first_end_probability = 0.0
        with torch.inference_mode():
            for row in rows:
                prefix = token.chat([{"role": "user", "text": row["input"]}],
                                    bos=True, eos=False)
                target = token.encode(row["answer_target"])
                if not target:
                    continue
                full = prefix + [SPECIAL["ASSISTANT"]] + target + [SPECIAL["END"], SPECIAL["EOS"]]
                logits = model(torch.tensor([full[:-1]], dtype=torch.long))[0]
                start = len(prefix)
                selected = logits[start:start + len(target)]
                labels = torch.tensor(target, dtype=torch.long)
                loss += float(F.cross_entropy(selected, labels, reduction="sum"))
                token_count += len(target)
                correct += int((selected.argmax(dim=-1) == labels).sum())
                first = torch.softmax(logits[start], dim=-1)
                first_correct += int(int(logits[start].argmax()) == target[0])
                first_gold_probability += float(first[target[0]])
                first_eos_probability += float(first[SPECIAL["EOS"]])
                first_end_probability += float(first[SPECIAL["END"]])
                examples += 1
        scores.append({"path": str(path), "checkpoint_sha256": file_sha(path),
            "stage_updates": payload.get("updates", payload.get("step")),
            "starting_checkpoint_sha256": payload.get("starting_checkpoint_sha256"),
            "validation_examples": examples, "answer_target_tokens": token_count,
            "answer_target_nll": loss / max(1, token_count),
            "answer_token_accuracy": correct / max(1, token_count),
            "first_answer_token_accuracy": first_correct / max(1, examples),
            "mean_first_gold_token_probability": first_gold_probability / max(1, examples),
            "mean_first_step_p_eos": first_eos_probability / max(1, examples),
            "mean_first_step_p_end": first_end_probability / max(1, examples)})
    return scores


def run(output: Path = OUT, checkpoint: Path = pilot.START) -> dict[str, Any]:
    if guard.available_mib() < guard.MEMORY_GUARD_MIB:
        raise MemoryError("effective available memory below the existing 2560 MiB guard")
    guard.configure_torch()
    torch.set_grad_enabled(False)
    model, state = pilot.load_model(checkpoint)
    model.eval()
    token = Tokenizer.load(pilot.TOKENIZER)
    splits, manifest = pilot.read_data()
    rows_by_id = {row["example_id"]: row for name in PARTITIONS for row in
                  pilot.selected_evaluation_rows(splits[name])}
    previous = json.loads((HERE / "correction_pilot" / "comparison.json").read_text())
    previous_greedy = previous["baseline"]["per_split"]
    per_decoder = {}
    selection = {}
    started = time.monotonic()

    for name, settings in INDEPENDENT.items():
        per_split = {}
        for split in PARTITIONS:
            rows = pilot.selected_evaluation_rows(splits[split])
            records = [decode(model, token, row["input"], row["example_id"], name, settings)
                       for row in rows]
            metrics = score_records(records, rows_by_id)
            if name == "greedy":
                old_rows = previous_greedy[split]["predictions"]
                matches = all(new["text"] == old["text"] and new["stop"] == old["stop"]
                              for new, old in zip(records, old_rows))
                metrics["reproduces_previous_text_and_stop"] = matches
            per_split[split] = metrics
        per_decoder[name] = {"settings": settings, "per_split": per_split}

    validation_base = per_decoder["greedy"]["per_split"]["validation"]
    validation_min = per_decoder["minimum_4_mask_end_eos"]["per_split"]["validation"]
    candidates = []
    if validation_min["semantic_coverage"] > validation_base["semantic_coverage"]:
        for name in ("repetition_penalty_1.2", "no_repeat_trigram"):
            metrics = per_decoder[name]["per_split"]["validation"]
            if (metrics["overall_semantic_accuracy"] >= validation_base["overall_semantic_accuracy"]
                    and metrics["repetition_percentage"] < validation_base["repetition_percentage"]):
                candidates.append(name)
    if len(candidates) > 2:
        candidates = candidates[:2]
    combos = {}
    for component in candidates:
        combo_name = f"minimum_4_plus_{component}"
        settings = {"minimum_length": MIN_NEW_TOKENS}
        settings.update(INDEPENDENT[component])
        per_split = {}
        for split in PARTITIONS:
            rows = pilot.selected_evaluation_rows(splits[split])
            records = [decode(model, token, row["input"], row["example_id"], combo_name, settings)
                       for row in rows]
            per_split[split] = score_records(records, rows_by_id)
        combos[combo_name] = {"settings": settings, "per_split": per_split}

    eos_analysis = teacher_forced_specials(model, token, splits)
    candidate_paths = [Path("research/foundation/data/learn.pt"),
        Path("research/foundation/data/a1.pt"),
        Path("research/structure/pilot_v5/a.pt"), Path("research/structure/pilot_v5/b.pt"),
        Path("research/structure/pilot_v5/c.pt"),
        Path("research/interaction/correction_pilot/corrected.pt"),
        Path("research/interaction/correction_pilot_96/corrected.pt")]
    inventory_paths = sorted(Path("research").rglob("*.pt"))
    validation_rows = pilot.selected_evaluation_rows(splits["validation"])
    result = {"schema": 1, "experiment": "foundation-decoding-diagnostic-v1",
        "inference_only": True, "weights_modified": False, "training_performed": False,
        "repository": {"branch": "adaptive", "head": "bd61b24ace8b23290fb3a8e546e73c75107ff535"},
        "checkpoint": {"path": str(checkpoint), "sha256": file_sha(checkpoint),
            "starting_checkpoint_sha256": state.get("starting_checkpoint_sha256") or
                state.get("meta", {}).get("initial"),
            "updates_in_checkpoint": state.get("updates", state.get("step")),
            "model_config": state["config"]["model"],
            "parameters": model.params, "seed": state.get("seed")},
        "tokenizer": {"path": str(pilot.TOKENIZER), "file_sha256": file_sha(pilot.TOKENIZER),
                      "digest": token.digest, "version": 1, "special_ids": SPECIAL},
        "dataset": {"path": str(pilot.DATASET),
            "manifest_sha256": manifest["manifest_sha256"],
            "selected_examples": {split: len(pilot.selected_evaluation_rows(splits[split]))
                                  for split in PARTITIONS},
            "prompt_surface": "explicit", "selection": "correction_pilot.selected_evaluation_rows"},
        "previous_report_reproduction": {"report": "research/interaction/correction_pilot/comparison.json",
            "checkpoint_sha256": previous["initial_checkpoint_sha256"],
            "match_checks_in_greedy_arm": {
                split: per_decoder["greedy"]["per_split"][split]["reproduces_previous_text_and_stop"]
                for split in PARTITIONS}},
        "decoding": {"max_new_tokens": MAX_NEW_TOKENS, "seed": SAMPLE_SEED,
            "configs": per_decoder, "validation_selected_combinations": candidates,
            "combinations": combos},
        "teacher_forced_special_token_analysis": eos_analysis,
        "checkpoint_inventory": checkpoint_inventory(inventory_paths),
        "checkpoint_validation_scores": checkpoint_validation_scores(candidate_paths, token,
                                                                        validation_rows),
        "runtime": {"device": "cpu", "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(), "peak_rss_bytes":
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "seconds": time.monotonic() - started}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                                        encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=pilot.START)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    result = run(args.output, args.checkpoint)
    print(json.dumps({"output": str(args.output / "report.json"),
                      "seconds": result["runtime"]["seconds"],
                      "independent_decoders": len(result["decoding"]["configs"]),
                      "combinations": len(result["decoding"]["combinations"])}, sort_keys=True))
