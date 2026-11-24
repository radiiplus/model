"""Probe the existing structural model on a retrieved concept-composition prompt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from research.foundation.model import Foundation
from research.interaction.tokenizer import SPECIAL, Tokenizer


HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE.parent / "structure" / "pilot_v5" / "b.pt"
TOKENIZER = HERE.parent / "interaction" / "data" / "ready7" / "tokenizer.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(model: Foundation, token: Tokenizer, prompt: str, limit: int = 48) -> dict[str, Any]:
    prefix = token.chat([{"role": "user", "text": prompt}], bos=True, eos=False)
    ids = prefix + [SPECIAL["ASSISTANT"]]
    output = []
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
            output.append(next_id)
            ids.append(next_id)
    return {"text": token.decode(output), "stop": reason, "tokens": len(output)}


def evaluate() -> dict[str, Any]:
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = Foundation(saved["config"]["model"])
    model.load_state_dict(saved["model"])
    model.eval()
    token = Tokenizer.load(TOKENIZER)
    prompts = [
        {"context": "cold", "expected": "no"},
        {"context": "warm", "expected": "yes"},
    ]
    results = []
    for row in prompts:
        prompt = ("Concept record: cell baseline stores electrical energy. "
            f"Context qualification: in {row['context']} conditions, cell supports lamp = "
            f"{'false' if row['context'] == 'cold' else 'true'}. "
            "Relationship: lamp requires cell. Question: Can cell power lamp in "
            f"{row['context']} conditions? Answer with yes or no: ")
        result = generate(model, token, prompt)
        normalized = result["text"].strip().lower().split()
        answer = normalized[0].strip(".,:;!?") if normalized else ""
        results.append({**row, "answer": answer or None,
                        "correct": answer == row["expected"], **result})
    return {"schema": 1, "evaluation": "foundation-concept-context-probe-v1",
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha(CHECKPOINT),
        "tokenizer_sha256": sha(TOKENIZER), "tokenizer_digest": token.digest,
        "model_parameters": model.params, "decode": "greedy, max 48 tokens",
        "cases": results, "accuracy": sum(row["correct"] for row in results) / len(results),
        "limitation": "Two synthetic prompts only; no model claim is made from this sample."}


def main() -> None:
    torch.set_num_threads(1)
    report = evaluate()
    path = HERE / "concept_model_metrics.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
