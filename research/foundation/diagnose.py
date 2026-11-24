"""Controlled optimizer/schedule continuation comparison for the Stage A pilot."""

from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import torch

from research.foundation import trainer
from research.foundation.model import Foundation
from research.interaction.tokenizer import SPECIAL, Tokenizer


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "data" / "learn.pt"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def state_hash(state: dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        hasher.update(key.encode())
        hasher.update(ctypes.string_at(tensor.data_ptr(), tensor.numel() * tensor.element_size()))
    return hasher.hexdigest()


def batch_hash(tokens: torch.Tensor, mask: torch.Tensor) -> str:
    def content(value: torch.Tensor) -> bytes:
        tensor = value.contiguous()
        return ctypes.string_at(tensor.data_ptr(), tensor.numel() * tensor.element_size())
    return digest(content(tokens) + content(mask))


def generation(model: Foundation, token: Tokenizer, count: int = 12) -> dict[str, Any]:
    prompt = [SPECIAL["BOS"], SPECIAL["USER"], *token.encode("Hi"),
              SPECIAL["END"], SPECIAL["ASSISTANT"]]
    generated = []
    model.eval()
    with torch.no_grad():
        for _ in range(count):
            context = torch.tensor([prompt[-model.context:]], dtype=torch.long)
            next_id = int(model(context)[0, -1].argmax())
            if next_id in (SPECIAL["END"], SPECIAL["EOS"]):
                break
            generated.append(next_id)
            prompt.append(next_id)
    pieces = []
    plain = []
    names = {value: f"<{key.lower()}>" for key, value in SPECIAL.items()}
    for value in generated:
        if value in token.parts:
            plain.append(value)
        else:
            if plain:
                pieces.append(token.decode(plain))
                plain = []
            pieces.append(names.get(value, f"<id:{value}>"))
    if plain:
        pieces.append(token.decode(plain))
    return {"prompt": "Hi", "token_ids": generated, "text": "".join(pieces)}


def prepare_batches(payload: dict, config: dict, updates: int) -> tuple[list, dict]:
    sampler = trainer.SequentialSampler(trainer.DATA, "train", int(config["train"]["seed"]) + 1,
                                        payload["sampler"])
    batches = []
    for _ in range(updates):
        tokens, mask = trainer.batch(sampler, int(config["train"]["batch"]),
                                     int(config["train"]["sequence"]))
        batches.append((tokens, mask))
    return batches, sampler.state()


def arm(name: str, payload: dict, config: dict, batches: list,
        validation_seed: int, token: Tokenizer) -> dict[str, Any]:
    model = Foundation(config["model"])
    model.load_state_dict(payload["model"])
    starting_hash = state_hash(model.state_dict())
    train = config["train"]
    opt = torch.optim.AdamW(model.parameters(), lr=float(train["rate"]),
                            weight_decay=float(train["penalty"]))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: trainer.rate(step, train))
    optimizer_restored = name == "A"
    scheduler_restored = name in {"A", "B"}
    if optimizer_restored:
        opt.load_state_dict(payload["optimizer"])
    if scheduler_restored:
        schedule.load_state_dict(copy.deepcopy(payload["scheduler"]))
        current = float(payload["scheduler"]["_last_lr"][0])
        for group in opt.param_groups:
            group["lr"] = current
    optimizer_entries_at_start = len(opt.state)
    random.setstate(payload["python"])
    torch.set_rng_state(payload["torch"])
    before = trainer.evaluate(model, trainer.DATA, validation_seed, train,
                              time.monotonic() + max(30, int(train["time"])))
    rates = []
    hashes = []
    started = time.monotonic()
    for tokens, mask in batches:
        hashes.append(batch_hash(tokens, mask))
        used_rate = float(opt.param_groups[0]["lr"])
        model.train()
        opt.zero_grad(set_to_none=True)
        losses = model.losses(tokens, mask)
        objective = (float(train["lang"]) * losses["language"] +
                     float(train["sft"]) * losses["interaction"])
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train["clip"]))
        opt.step()
        schedule.step()
        rates.append({"used": used_rate, "after": float(opt.param_groups[0]["lr"])})
    elapsed = time.monotonic() - started
    after = trainer.evaluate(model, trainer.DATA, validation_seed, train,
                             time.monotonic() + max(30, int(train["time"])))
    return {"arm": name,
            "condition": {"A": "restore optimizer and scheduler state",
                          "B": "reset optimizer; restore matched scheduler state",
                          "C": "reset optimizer and scheduler"}[name],
            "starting_weights_sha256": starting_hash,
            "initial_position_update": int(payload["step"]),
            "source_optimizer_state_available": bool(payload.get("optimizer")),
            "optimizer_restored": optimizer_restored,
            "scheduler_restored": scheduler_restored,
            "validation_seed": validation_seed,
            "validation_sampler": trainer.Sampler.MODE,
            "scheduler_start_update": int(schedule.last_epoch),
            "optimizer_start_state_entries": optimizer_entries_at_start,
            "updates": len(batches), "examples_consumed": len(batches),
            "positions_trained": sum(int(tokens.shape[1] - 1) for tokens, _ in batches),
            "training_batch_sha256": hashes,
            "training_sequence_sha256": digest("".join(hashes).encode()),
            "learning_rates": rates, "validation_before": before,
            "validation_after": after, "generation_after": generation(model, token),
            "seconds_fit": elapsed, "final_weights_sha256": state_hash(model.state_dict())}


def run(output: Path, updates: int = 32) -> dict[str, Any]:
    if updates < 1 or (output / "manifest.json").exists():
        raise ValueError("updates must be positive and a previous experiment is present")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(trainer.load()["train"]["seed"]))
    trainer.guard.configure_torch()
    payload = torch.load(SOURCE, map_location="cpu", weights_only=True)
    config = trainer.load()
    config["train"].update({"steps": updates, "sequence": 125, "eval": 512,
                            "lang": 1.0, "sft": 0.0, "time": 90})
    source_config = payload.get("config", {})
    if (payload.get("schema") != trainer.ARCH or "optimizer" not in payload
            or "scheduler" not in payload or "sampler" not in payload
            or source_config.get("model") != config["model"]):
        raise ValueError("source checkpoint lacks compatible optimizer, scheduler or sampler state")
    batches, cursor = prepare_batches(payload, config, updates)
    token = Tokenizer.load(trainer.HERE.parent / "interaction" / "data" / "ready7" / "tokenizer.json")
    shared = digest("".join(batch_hash(t, m) for t, m in batches).encode())
    results = {}
    for name in ("A", "B", "C"):
        value = arm(name, payload, config, batches, int(config["train"]["seed"]) + 2, token)
        value["shared_training_sequence_sha256"] = shared
        path = output / f"arm_{name.lower()}.json"
        path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        results[name] = {"validation_before": value["validation_before"],
                         "validation_after": value["validation_after"],
                         "optimizer_restored": value["optimizer_restored"],
                         "scheduler_restored": value["scheduler_restored"],
                         "positions_trained": value["positions_trained"],
                         "rates": value["learning_rates"]}
    reference = results["A"]["validation_after"].get("language")
    delta_b = results["B"]["validation_after"].get("language")
    delta_c = results["C"]["validation_after"].get("language")
    conclusion = ("supported" if reference is not None and delta_c is not None
                  and delta_c > reference and delta_b is not None
                  and abs(delta_b - reference) < abs(delta_c - reference)
                  else "unsupported" if reference is not None and delta_c is not None
                  and delta_c <= reference and delta_b is not None
                  and abs(delta_b - reference) <= abs(delta_c - reference)
                  else "inconclusive")
    manifest = {"schema": 1, "source_checkpoint": str(SOURCE),
                "source_checkpoint_sha256": trainer.file(SOURCE),
                "source_update": int(payload["step"]), "updates_per_arm": updates,
                "positions_per_arm": updates * int(config["train"]["sequence"]),
                "shared_training_sequence_sha256": shared,
                "source_sampler_position": {
                    "mode": "legacy-sequential",
                    "cursor": payload["sampler"]["cursor"],
                    "buffered_tokens": len(payload["sampler"]["ids"])},
                "resume_cursor_after": {
                    "cursor": cursor["cursor"], "buffered_tokens": len(cursor["ids"]),
                    "state_sha256": trainer.digest(cursor)},
                "architecture": config["model"],
                "dataset": trainer.pack(trainer.DATA, config)["dataset"],
                "split": trainer.pack(trainer.DATA, config)["split"],
                "tokenizer": trainer.pack(trainer.DATA, config)["tokenizer"],
                "evaluation": "same deterministic validation sampler, seed, and 512-token budget for every arm",
                "arms": results, "restart_contribution": conclusion,
                "interpretation": "Controlled schedule restart contrast on identical post-800-update examples; not a full reproduction of the historical pilot's data reset."}
    manifest["learning_rate_trajectory_matches"] = {
        "A_vs_B": results["A"]["rates"] == results["B"]["rates"]}
    manifest["training_sequence_matches"] = all(
        json.loads((output / f"arm_{name.lower()}.json").read_text())["training_sequence_sha256"]
        == shared for name in ("A", "B", "C"))
    if not manifest["learning_rate_trajectory_matches"]["A_vs_B"]:
        raise AssertionError("Arm B learning-rate trajectory does not match Arm A")
    if not manifest["training_sequence_matches"]:
        raise AssertionError("diagnostic arms consumed different training examples")
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n",
                                           encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.updates), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
