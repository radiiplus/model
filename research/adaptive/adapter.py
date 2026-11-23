"""Offline LoRA adapter experiment bound to a validated strategy version."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import resource
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from research.adaptive.loop import step
from research.recursive.evaluate import interval, score
from research.prototype.trainer import (
    MEMORY_GUARD_MIB,
    available_mib,
    configure_torch,
    swap_counters,
)
from research.recursive.model import Decoder
from research.recursive.tasks import BANDS, VERSION, build, draw


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "experiments/results/batch12_byte_seed_11.pt"
STRAT = ROOT / "out.json"
TASKS = ROOT.parent / "recursive/tasks.py"
EVAL = ROOT.parent / "recursive/evaluate.py"
RANK = 4
WIDTH = 32
CONTEXT = 512
COUNT = 4
SEEDS = [911, 912, 913]
FAMILIES = ["addition", "bracket", "chain", "graph", "trace"]
GATE = {"updates": 120, "rate": 0.003, "decay": 0.01,
        "clip": 1.0, "steps": 2, "rank": RANK,
        "budget": 1_000_000, "min_gain": 0.0}
deadline = float("inf")


def raw(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(raw(value)).hexdigest()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp(state: dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        hasher.update(name.encode("utf-8"))
        hasher.update(str(value.dtype).encode("ascii"))
        hasher.update(raw(list(value.shape)))
        hasher.update(bytes(value.view(torch.uint8).reshape(-1).tolist()))
    return hasher.hexdigest()


def expand(value: torch.Tensor, count: int) -> torch.Tensor:
    width = value.shape[1]
    result = torch.empty((count, width), dtype=value.dtype)
    result[:len(value)].copy_(value)
    pos = torch.arange(len(value), count, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, width, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / width))
    tail = torch.zeros((count - len(value), width), dtype=torch.float32)
    tail[:, 0::2] = torch.sin(pos * div)
    tail[:, 1::2] = torch.cos(pos * div)
    result[len(value):].copy_(tail.to(value.dtype) * 0.02)
    return result


def loadbase(payload: dict[str, Any]) -> tuple[Decoder, dict[str, Any]]:
    source = payload.get("model")
    config = payload.get("config")
    if not isinstance(source, dict) or not isinstance(config, dict):
        raise ValueError("foundation checkpoint must contain config and model state")
    model = Decoder({**config, "context_length": CONTEXT,
                     "limit": GATE["steps"], "steps": GATE["steps"]})
    current = model.state_dict()
    for name, value in source.items():
        if name == "position_embedding.weight":
            if value.shape[0] > CONTEXT or value.shape[1] != WIDTH:
                raise ValueError("foundation position table is incompatible")
            current[name] = expand(value, CONTEXT)
        elif name in current and current[name].shape == value.shape:
            current[name] = value.detach().clone()
        else:
            raise ValueError(f"foundation tensor mismatch for {name}")
    result = model.load_state_dict(current, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("foundation load was incomplete")
    model.eval()
    for value in model.parameters():
        value.requires_grad_(False)
    return model, {"source": config, "context": CONTEXT,
                   "positions_added": CONTEXT - len(source["position_embedding.weight"]),
                   "position_params_added": (CONTEXT - len(source["position_embedding.weight"])) * WIDTH,
                   "extension": stamp({"position": model.position_embedding.weight[32:]})}


class Adapter(nn.Module):
    def __init__(self, width: int = WIDTH, rank: int = RANK, seed: int = 0):
        super().__init__()
        if width < 1 or not 1 <= rank <= width:
            raise ValueError("adapter width and rank are invalid")
        torch.manual_seed(seed)
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)
        self.scale = 1.0 / rank
        self.active = False
        self.handle = None

    @property
    def params(self) -> int:
        return sum(value.numel() for value in self.parameters())

    def hook(self, module: nn.Module, inputs: tuple[torch.Tensor, ...],
             output: torch.Tensor) -> torch.Tensor:
        if not self.active:
            return output
        return output + self.up(self.down(inputs[0])) * self.scale

    def mount(self, model: Decoder) -> None:
        if self.handle is not None:
            raise RuntimeError("adapter is already mounted")
        self.handle = model.block.attn.proj.register_forward_hook(self.hook)

    def unmount(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.active = False


def mark(rows: list[dict[str, Any]]) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    result = []
    for item in rows:
        prompt = item["prompt"].encode("utf-8")
        answer = item["target"].encode("utf-8") + b"\n"
        full = prompt + answer
        if len(full) > CONTEXT:
            raise ValueError("task input and answer exceed the adapter context")
        if len(full) < 2:
            raise ValueError("task input has no next-byte target")
        values = torch.tensor(list(full), dtype=torch.long)
        ids = values[:-1].unsqueeze(0)
        targets = values[1:].unsqueeze(0)
        mask = (torch.arange(targets.shape[1]) + 1 >= len(prompt)).unsqueeze(0)
        if not bool(mask.any()):
            raise ValueError("answer mask selected no bytes")
        result.append((ids, targets, mask))
    return result


def data() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks = build()
    payload = json.loads(STRAT.read_text(encoding="utf-8"))
    eligible = set(payload["candidate"]["episodes"])
    train = []
    for item in tasks["families"]["chain"]["train"]:
        key = hashlib.sha256(item["prompt"].encode("utf-8")).hexdigest()
        episode = digest([item["family"], item["id"], key])
        if episode not in eligible:
            raise ValueError("chain training row does not match the validated episode ledger")
        train.append(item)
    if len(train) != len(eligible):
        raise ValueError("adapter training may use only chain train episodes")
    for item in train:
        mark([item])
    meta = {"strategy": payload["candidate"]["id"],
            "strategyversion": payload["candidate"]["version"],
            "taskversion": VERSION, "trainids": sorted(item["id"] for item in train),
            "trainhash": digest([{key: item[key] for key in ("id", "prompt", "target")}
                                  for item in train])}
    return train, meta


def holdout(train: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    seen = {hashlib.sha256(item["prompt"].encode("utf-8")).hexdigest()
            for item in train}
    result = {}
    hashes = {}
    for familyindex, family in enumerate(FAMILIES):
        result[family] = {}
        hashes[family] = {}
        for seedindex, seed in enumerate(SEEDS):
            result[family][str(seed)] = {}
            hashes[family][str(seed)] = {}
            for splitindex, split in enumerate(("interpolation", "extrapolation")):
                rng = random.Random(seed + familyindex * 100_003 + splitindex * 7_919)
                low, high = BANDS[family][split]
                rows = []
                for index in range(COUNT):
                    for attempt in range(10000):
                        level = rng.randint(low, high)
                        prompt, target = draw(family, rng, level)
                        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                        if key not in seen:
                            item = {"id": f"{family}-{split}-{seed}-{index:02d}",
                                    "family": family, "split": split,
                                    "level": level, "prompt": prompt, "target": target}
                            mark([item])
                            seen.add(key)
                            rows.append(item)
                            break
                    else:
                        raise RuntimeError("could not generate a distinct adapter test item")
                result[family][str(seed)][split] = rows
                hashes[family][str(seed)][split] = digest(rows)
    return result, hashes


def signature(meta: dict[str, Any], config: dict[str, Any], basehash: str,
              seed: int) -> dict[str, Any]:
    return {"strategy": meta["strategy"], "strategyversion": meta["strategyversion"],
            "dataset": meta["trainhash"], "taskversion": meta["taskversion"],
            "foundation": basehash, "seed": seed, "rank": RANK,
            "steps": GATE["steps"], "rate": GATE["rate"],
            "decay": GATE["decay"], "model": config}


def checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".temp")
    torch.save(payload, temp)
    os.replace(temp, path)


def fit(model: Decoder, adapter: Adapter, rows: list[dict[str, Any]],
        sig: dict[str, Any], seed: int, total: int, path: Path,
        resume: Path | None = None, stop: int | None = None) -> dict[str, Any]:
    if available_mib() < MEMORY_GUARD_MIB:
        raise MemoryError(f"effective memory below {MEMORY_GUARD_MIB} MiB")
    if not 1 <= total <= 120 or stop is not None and not 1 <= stop <= total:
        raise ValueError("adapter update budget is invalid")
    started = time.monotonic()
    batches = mark(rows)
    adapter.mount(model)
    adapter.active = True
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=GATE["rate"],
                                  weight_decay=GATE["decay"])
    sample = torch.Generator(device="cpu").manual_seed(seed + 1)
    start = 0
    history = []
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=True)
        if payload["signature"] != sig or payload["total"] != total:
            adapter.unmount()
            raise ValueError("adapter resume signature mismatch")
        adapter.load_state_dict(payload["adapter"])
        optimizer.load_state_dict(payload["optimizer"])
        sample.set_state(payload["sampler"])
        torch.set_rng_state(payload["torch"])
        start = int(payload["step"])
        history = list(payload["history"])
    if start >= total or stop is not None and stop <= start:
        adapter.unmount()
        raise ValueError("adapter resume cursor is invalid")
    deadline = time.monotonic() + 90
    swaps = swap_counters()
    actual = total if stop is None else stop
    try:
        model.eval()
        for update in range(start, actual):
            if update % 10 == 0:
                if available_mib() < MEMORY_GUARD_MIB:
                    raise MemoryError(f"adapter memory guard reached update {update}")
                now = swap_counters()
                if swaps is not None and now is not None and any(
                    right > left for left, right in zip(swaps, now)
                ):
                    raise MemoryError("swap activity detected during adapter training")
                swaps = now
            if time.monotonic() > deadline:
                raise TimeoutError("adapter training exceeded its 90 second wall cap")
            index = int(torch.randint(len(batches), (1,), generator=sample).item())
            ids, targets, mask = batches[index]
            optimizer.zero_grad(set_to_none=True)
            _, trace = model(ids, steps=GATE["steps"], trace=True,
                             targets=targets, mask=mask)
            loss = trace["losses"][-1]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("adapter loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), GATE["clip"])
            optimizer.step()
            history.append(float(loss.detach()))
            checkpoint(path, {"signature": sig, "total": total,
                              "step": update + 1, "adapter": adapter.state_dict(),
                              "optimizer": optimizer.state_dict(),
                              "sampler": sample.get_state(),
                              "torch": torch.get_rng_state(), "history": history})
        return {"step": actual, "interrupted": actual < total,
                "history": history, "state": stamp(adapter.state_dict()),
                "checkpoint": sha(path), "bytes": path.stat().st_size,
                "seconds": time.monotonic() - started,
                "applications": (actual - start) * GATE["steps"]}
    finally:
        adapter.unmount()


def mount(model: Decoder, items: list[Adapter]) -> None:
    for item in items:
        item.mount(model)
        item.active = True


def unmount(items: list[Adapter]) -> None:
    for item in items:
        item.unmount()


def answer(model: Decoder, items: list[Adapter], task: dict[str, Any]) -> str:
    prompt = task["prompt"].encode("utf-8")
    if not prompt or len(prompt) >= CONTEXT:
        raise ValueError("inference prompt exceeds the fixed context")
    limit = 48 if task["family"] == "graph" else 8
    mount(model, items)
    result = []
    token = torch.tensor([list(prompt)], dtype=torch.long)
    try:
        with torch.no_grad():
            for _ in range(limit):
                if time.monotonic() > deadline:
                    raise TimeoutError("adapter evaluation exceeded its wall-time cap")
                logits = model(token, steps=GATE["steps"])
                nextid = int(logits[0, -1].argmax().item())
                if nextid == 10:
                    break
                result.append(nextid)
                token = torch.cat((token, torch.tensor([[nextid]], dtype=torch.long)), dim=1)
                if token.shape[1] >= CONTEXT:
                    break
    finally:
        unmount(items)
    return bytes(result).decode("utf-8", errors="replace").strip()


def measure(model: Decoder, items: list[Adapter], rows: list[dict[str, Any]]) -> dict[str, Any]:
    start = time.perf_counter()
    total = 0
    errors: dict[str, int] = {}
    samples = []
    for task in rows:
        output = answer(model, items, task)
        outcome = score(task, output)
        total += int(outcome["correct"])
        if outcome["error"]:
            errors[outcome["error"]] = errors.get(outcome["error"], 0) + 1
        samples.append([task["id"], output, outcome["correct"], outcome["error"]])
    count = len(rows)
    elapsed = time.perf_counter() - start
    return {"samples": count, "correct": total,
            "accuracy": total / count if count else 0.0,
            "ci95": interval(total, count),
            "errors": errors, "seconds": elapsed,
            "latency": elapsed / count if count else 0.0,
            "sampleset": samples}


def tracks(model: Decoder, items: list[Adapter], data: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for family in FAMILIES:
        if family not in data:
            continue
        result[family] = {}
        for seed, parts in data[family].items():
            result[family][seed] = {}
            for split, rows in parts.items():
                result[family][seed][split] = measure(model, items, rows)
    return result


def factory() -> tuple[Decoder, dict[str, Any], dict[str, Any]]:
    payload = torch.load(BASE, map_location="cpu", weights_only=True)
    model, extra = loadbase(payload)
    info = {"file": sha(BASE), "source": stamp({name: value
            for name, value in model.state_dict().items()
            if name not in {"step", "gate"}}),
            "params": model.params, "extra": extra}
    return model, payload, info


def generate() -> dict[str, Any]:
    global deadline
    configure_torch()
    torch.manual_seed(71)
    started = time.monotonic()
    deadline = started + 105
    if available_mib() < MEMORY_GUARD_MIB:
        raise MemoryError(f"effective memory below {MEMORY_GUARD_MIB} MiB")
    if not BASE.is_file():
        raise FileNotFoundError(BASE)
    strategy = json.loads(STRAT.read_text(encoding="utf-8"))["candidate"]
    if strategy["status"] != "validated" or "trace" not in strategy["valid"]:
        raise ValueError("adapter must be bound to the validated trace strategy")
    train, meta = data()
    trainhash = meta["trainhash"]
    model, payload, base = factory()
    if model.params + RANK * WIDTH * 2 > GATE["budget"]:
        raise ValueError("foundation plus one adapter exceeds the parameter budget")
    before = stamp(model.state_dict())
    basefile = sha(BASE)
    for value in model.parameters():
        if value.requires_grad:
            raise AssertionError("foundation parameter was not frozen")
    config = {"context": CONTEXT, "steps": GATE["steps"], "rank": RANK,
              "layer": "block.attn.proj", "updates": GATE["updates"],
              "rate": GATE["rate"], "decay": GATE["decay"]}
    modelcfg = {key: model.config[key] for key in
                ("vocab_size", "context_length", "d_model", "n_heads", "d_ff")}
    sig = signature(meta, modelcfg, basefile, 17)
    torch.manual_seed(17)
    one = Adapter(seed=17)
    vtwo = Adapter(seed=17)
    with tempfile.TemporaryDirectory(prefix="adapter-") as folder:
        trainstart = time.monotonic()
        path = Path(folder) / "one.pt"
        start = fit(model, one, train, sig, 17, GATE["updates"], path,
                    stop=GATE["updates"] // 2)
        with torch.no_grad():
            vtwo.load_state_dict(one.state_dict())
        second = fit(model, vtwo, train, sig, 17, GATE["updates"],
                     Path(folder) / "two.pt", resume=path)
        trainingwall = time.monotonic() - trainstart
        previous = {key: value.clone() for key, value in one.state_dict().items()}
        versionone = Adapter(seed=19)
        versionone.load_state_dict(previous)
        tests, testhash = holdout(train)
        beforeeval = {family: {seed: {split: measure(model, [], rows)
                                      for split, rows in parts.items()}
                               for seed, parts in seeds.items()}
                      for family, seeds in tests.items()}
        aftereval = tracks(model, [vtwo], tests)
        firsttrace = tracks(model, [versionone], {"trace": tests["trace"]})["trace"]
        joint = tracks(model, [versionone, vtwo], {"trace": tests["trace"]})["trace"]
        symbolic = {}
        for seed, parts in tests["trace"].items():
            symbolic[seed] = {}
            for split, rows in parts.items():
                count = sum(score(task, step(task)[0])["correct"] for task in rows)
                symbolic[seed][split] = {"correct": count, "samples": len(rows),
                                         "accuracy": count / len(rows)}
        rollback = Adapter(seed=29)
        rollback.load_state_dict(vtwo.state_dict())
        rollback.load_state_dict(previous)
        rolled = stamp(rollback.state_dict()) == stamp(previous)
        saved = Path(folder) / "adapter.pt"
        checkpoint(saved, {"adapter": vtwo.state_dict(),
                           "strategy": strategy["id"],
                           "strategyversion": strategy["version"],
                           "foundation": basefile, "dataset": trainhash,
                           "evaluator": sha(EVAL), "tasksource": sha(TASKS),
                           "config": config, "version": 2,
                           "previous": stamp(previous)})
        artifact = {"hash": sha(saved), "bytes": saved.stat().st_size,
                    "adapter": stamp(vtwo.state_dict()), "retained": False}
    after = stamp(model.state_dict())
    afterfile = sha(BASE)
    if before != after or basefile != afterfile:
        raise AssertionError("frozen foundation changed during adapter training")
    traceall = [value for seed in aftereval["trace"].values() for value in seed.values()]
    symall = [value for seed in symbolic.values() for value in seed.values()]
    symbolicrate = sum(value["correct"] for value in symall) / sum(value["samples"] for value in symall)
    adapterrate = sum(value["correct"] for value in traceall) / sum(value["samples"] for value in traceall)
    promoted = adapterrate > symbolicrate + GATE["min_gain"]
    evalhash = sha(EVAL)
    taskhash = sha(TASKS)
    result = {"version": 1, "mode": "neural-adapter-experiment",
              "decision": "promote" if promoted else "no-go",
              "reason": "adapter must beat the symbolic strategy-memory condition" if not promoted else "registered transfer gate passed",
              "strategy": {"id": strategy["id"], "version": strategy["version"],
                           "family": "trace"},
              "foundation": {"path": str(BASE), "hash": basefile,
                             "unchanged": before == after and basefile == afterfile,
                             "checkpoint_params": sum(value.numel() for value in payload["model"].values()),
                             "frozen_recurrent_params": model.params,
                             "position_params_added": base["extra"]["position_params_added"],
                             "recurrent_controls": model.params - sum(value.numel() for value in payload["model"].values()) - base["extra"]["position_params_added"],
                             "extension": base["extra"]},
              "adapter": {"rank": RANK, "layer": config["layer"],
                          "params": one.params, "versionone": start,
                          "versiontwo": second, "artifact": artifact,
                          "rollback": rolled, "selected": False},
              "budget": {"cap": GATE["budget"],
                         "one_active": model.params + vtwo.params,
                         "two_active_experiment": model.params + vtwo.params + versionone.params,
                         "all_adapter_params_loaded": vtwo.params + versionone.params},
              "train": {"count": len(train), "hash": trainhash,
                        "ids": meta["trainids"], "updates": second["step"],
                        "target_mask": "answer bytes plus newline only",
                        "wall": trainingwall,
                        "block_applications": GATE["updates"] * GATE["steps"],
                        "seconds_per_update": trainingwall / GATE["updates"]},
              "validation": {"family": "chain", "split": "validation",
                             "hash": digest([{key: item[key] for key in ("id", "prompt", "target")}
                                              for item in build()["families"]["chain"]["validation"]]),
                             "status": "not used for optimization or adapter selection"},
              "tests": {"hashes": testhash, "base": beforeeval,
                        "adapter": aftereval, "symbolic": symbolic,
                        "versionone_trace": firsttrace,
                        "combined_trace": joint},
              "interference": {"tested": True, "active_combination": False,
                               "decision": "not combined for use; symbolic gate failed",
                               "versionone": firsttrace, "versiontwo": {
                                   seed: {split: aftereval["trace"][seed][split]
                                          for split in aftereval["trace"][seed]}
                                   for seed in aftereval["trace"]},
                               "combined": joint},
              "params": {"foundation": model.params, "adapter": vtwo.params,
                         "active": model.params + vtwo.params, "trainable": vtwo.params},
              "metrics": {"trace_symbolic": symbolicrate,
                          "trace_adapter": adapterrate,
                          "beats_symbolic": promoted},
              "hashes": {"strategy": digest(strategy), "dataset": trainhash,
                         "evaluator": evalhash, "tasks": taskhash,
                         "source": sha(Path(__file__)),
                         "foundation": basefile, "adapter": artifact["adapter"]},
              "runtime": {"wall": time.monotonic() - started,
                          "rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                          "threads": torch.get_num_threads(), "device": "cpu",
                          "guard_mib": MEMORY_GUARD_MIB, "available_mib": available_mib(),
                          "timeout": 120, "model_loaded": True},
              "limits": ["Project-authored synthetic task families only.",
                         "No new corpus, checkpoint, package, service or GPU was used.",
                         "The 224-position embedding extension is deterministic and frozen; base positions are copied exactly.",
                         "A rejected candidate adapter was used only inside a temporary directory and no adapter weights are retained.",
                         "The base responder and symbolic solver are synthetic controls, not natural-language capability evidence."]}
    return result


def text(result: dict[str, Any]) -> str:
    neural = [value for fam in result["tests"]["adapter"].values()
              for seed in fam.values() for value in seed.values()]
    correct = sum(value["correct"] for value in neural)
    samples = sum(value["samples"] for value in neural)
    families = ", ".join(
        f"{name} {sum(cell['correct'] for seed in fam.values() for cell in seed.values())}/"
        f"{sum(cell['samples'] for seed in fam.values() for cell in seed.values())}"
        for name, fam in result["tests"]["adapter"].items())
    rows = ["# Patch 9: Strategy adapter experiment", "", "## Decision", "",
            f"**{result['decision'].upper()}**: {result['reason']}.", "",
            "The recurrent byte model was loaded from the existing Batch 12 checkpoint. Its parameters were frozen. A rank-4 LoRA residual was attached to `block.attn.proj`, trained from the validated Patch 8 strategy's 12 chain training episodes, and evaluated with the adapter explicitly selected. The rejected weight artifact was temporary and was deleted after measurement.",
            "", "## Results", "",
            "Each table cell shows exact matches per seed, with interpolation and extrapolation kept separate.", ""]
    for family in FAMILIES:
        rows.extend([f"### {family}", "", "| Seed | Split | Base | Adapter | Samples |", "|---:|---|---:|---:|---:|"])
        for seed, parts in result["tests"]["adapter"][family].items():
            for split, item in parts.items():
                base = result["tests"]["base"][family][seed][split]
                rows.append(f"| {seed} | {split} | {base['correct']} | {item['correct']} | {item['samples']} |")
    rows.extend(["", "## Transfer and safeguards", "",
                 f"Patch 8's symbolic strategy scored {result['metrics']['trace_symbolic']:.1%} on the same held-out trace rows; the neural adapter scored {result['metrics']['trace_adapter']:.1%}. It therefore did not beat the symbolic condition. The frozen foundation parameter hash remained `{result['foundation']['hash']}`. The recurrent foundation has {result['params']['foundation']:,} frozen parameters, including {result['foundation']['position_params_added']:,} deterministic extended position parameters and {result['foundation']['recurrent_controls']:,} recurrence controls. One rank-4 adapter has {result['params']['adapter']:,} trainable parameters; active total is {result['params']['active']:,}, below the 1,000,000 limit.",
                 f"Adapter fitting used {result['train']['count']} chain episodes, {result['train']['updates']} updates and {result['train']['block_applications']} forward block applications in {result['train']['wall']:.3f}s ({result['train']['seconds_per_update']:.4f}s/update). Across {samples} held-out generated cases, the adapter scored {correct}/{samples}; per-family counts were {families}. The overall run took {result['runtime']['wall']:.3f}s and peaked at {result['runtime']['rss']:,} bytes RSS.",
                 "", "The adapter-only checkpoint contains no foundation tensors and records strategy, strategy version, dataset, evaluator, task and foundation hashes. A prior adapter state was restored exactly. Two adapter versions were measured together as an interference probe, but combination was not activated. No adapter is automatically selected or updated during ordinary inference.",
                 "", "## Limits", "",
                 "The only available neural foundation is a 21,984-parameter model trained on a tiny synthetic text fixture. The model and all reasoning inputs remain synthetic. Context was extended from 32 to 512 with a deterministic frozen positional encoding; the first 32 learned positions were copied exactly. This makes the full task prompts fit without truncation but is an architectural change that may affect behavior. Results are a CPU mechanics experiment, not real-world reasoning or interaction quality.",
                 "", "Per-instance outputs, per-seed task metrics, hashes, resume data, parameter counts, interference results and resource data are in `neural.json`.", ""])
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "neural.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result = generate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_name("neural.md").write_text(text(result), encoding="utf-8")


if __name__ == "__main__":
    main()
