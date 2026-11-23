"""Seeded CPU ablations for shared and dual-state recurrent task models."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from research.prototype.data import load_splits
from research.prototype.interfaces import ByteTokenizer
from research.prototype.trainer import MEMORY_GUARD_MIB, available_mib
from research.recursive.dual import Refiner
from research.recursive.evaluate import interval, score
from research.recursive.model import Decoder
from research.recursive.stack import Stack
from research.recursive.tasks import CROSS, build, manifest
from research.recursive.train import digest, run
from research.experiments.tiny_gpt import TinyGPT


ROOT = Path(__file__).resolve().parent
HOME = ROOT.parents[1]
ARCHS = ("plain-v1", "loop-v1", "dual-v1", "stack-v1")


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def guard(deadline: float) -> None:
    if time.perf_counter() >= deadline:
        raise TimeoutError("ablation wall-time limit reached")
    if available_mib() < MEMORY_GUARD_MIB:
        raise MemoryError(f"effective available memory below {MEMORY_GUARD_MIB} MiB")


def make(base: dict[str, Any], arch: str, seed: int, loss: str = "deep") -> dict[str, Any]:
    count = 1 if arch in {"plain-v1", "stack-v1"} else 4
    model = dict(base["model"])
    model["steps"] = count
    if arch == "stack-v1":
        model["depth"] = int(base["depth"])
    return {
        "architecture": arch,
        "mode": "task",
        "family": base["family"],
        "seed": seed,
        "steps": count,
        "batch": int(base["batch"]),
        "rate": float(base["rate"]),
        "decay": float(base["decay"]),
        "clip": float(base["clip"]),
        "loss": loss,
        "weights": list(base["weights"]) if count > 1 else [1.0],
        "wall": float(base["wall"]),
        "model": model,
    }


def load(config: dict[str, Any], payload: dict[str, Any]) -> torch.nn.Module:
    arch = config["architecture"]
    if arch == "plain-v1":
        model = TinyGPT(config["model"])
    elif arch == "stack-v1":
        model = Stack(config["model"])
    elif arch == "dual-v1":
        model = Refiner(config["model"])
    else:
        model = Decoder(config["model"])
    model.load_state_dict(payload["model"])
    return model.eval()


def forward(model: torch.nn.Module, arch: str, ids: torch.Tensor, steps: int) -> torch.Tensor:
    if arch in {"plain-v1", "stack-v1"}:
        return model(ids)
    return model(ids, steps=steps)


def stop(family: str, text: str) -> bool:
    value = text.strip()
    if family == "graph":
        return value.endswith(">T")
    if family == "bracket":
        return value in {"yes", "no"}
    return bool(value) and value[-1] in "\n. "


def generate(model: torch.nn.Module, arch: str, prompt: str, family: str,
             steps: int, budget: int) -> tuple[str, float]:
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode(prompt)
    started = time.perf_counter()
    text = ""
    context = model.context if isinstance(model, Refiner) else model.context_length
    for _ in range(budget):
        inp = torch.tensor([ids[-context:]], dtype=torch.long)
        with torch.inference_mode():
            token = int(forward(model, arch, inp, steps)[:, -1].argmax(-1))
        ids.append(token)
        text += tokenizer.decode([token])
        if stop(family, text):
            break
    return text, time.perf_counter() - started


def metric(model: torch.nn.Module, arch: str, rows: list[dict[str, Any]],
           steps: int, family: str) -> dict[str, Any]:
    errors: dict[str, int] = {}
    correct = 0
    times = []
    for row in rows:
        cap = 64 if family == "graph" else 8 if family == "bracket" else 20
        answer, elapsed = generate(model, arch, row["prompt"], family, steps, cap)
        result = score(row, answer)
        correct += int(result["correct"])
        times.append(elapsed)
        if result["error"]:
            name = str(result["error"])
            errors[name] = errors.get(name, 0) + 1
    count = len(rows)
    rate = correct / count if count else None
    if count:
        factor = 1 + 1.96**2 / count
        center = (rate + 1.96**2 / (2 * count)) / factor
        radius = 1.96 * math.sqrt(rate * (1 - rate) / count + 1.96**2 / (4 * count**2)) / factor
        ci = [max(0.0, center - radius), min(1.0, center + radius)]
    else:
        ci = None
    return {"samples": count, "correct": correct, "accuracy": rate, "ci95": ci,
            "errors": errors, "mean": statistics.mean(times) if times else None,
            "p50": statistics.median(times) if times else None}


def language(model: torch.nn.Module, arch: str, docs: list[list[int]], steps: int,
             context: int) -> dict[str, float | int]:
    count = 0
    nll = 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for doc in docs:
            for start in range(0, len(doc) - 1, context):
                vals = doc[start:start + context + 1]
                if len(vals) < 2:
                    continue
                ids = torch.tensor([vals[:-1]], dtype=torch.long)
                target = torch.tensor(vals[1:], dtype=torch.long)
                logits = forward(model, arch, ids, steps)[0]
                nll += float(torch.nn.functional.cross_entropy(
                    logits, target, reduction="sum"))
                count += target.numel()
    return {"bytes": count, "nll": nll / count,
            "bpb": nll / count / math.log(2), "wall": time.perf_counter() - started}


def copy(model: torch.nn.Module, arch: str, rows: list[str], steps: int) -> dict[str, Any]:
    tokenizer = ByteTokenizer()
    passed = 0
    times = []
    for row in rows:
        prompt = f"Repeat exactly: {row}\nAnswer:"
        answer, elapsed = generate(model, arch, prompt, "copy", steps, len(tokenizer.encode(row)) + 1)
        passed += int(answer.strip() == row)
        times.append(elapsed)
    return {"samples": len(rows), "correct": passed,
            "accuracy": passed / len(rows), "ci": interval(passed, len(rows)),
            "mean": statistics.mean(times)}


def evaluate(model: torch.nn.Module, arch: str, data: dict[str, Any], steps: int,
             samples: int, text: list[list[int]], copies: list[str]) -> dict[str, Any]:
    report: dict[str, Any] = {"steps": steps, "tasks": {}, "cross": {},
                              "copy": copy(model, arch, copies, steps),
                              "language": language(model, arch, text, steps, 512)}
    for family, parts in data["families"].items():
        report["tasks"][family] = {}
        for split in ("interpolation", "extrapolation"):
            rows = parts[split][:samples]
            report["tasks"][family][split] = metric(model, arch, rows, steps, family)
    report["cross"]["chain"] = {
        "source": "graph", "target": "chain", "relation": "dependency",
        "metrics": report["tasks"]["chain"],
    }
    return report


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    base = read(ROOT / "ablation.json")
    if base["go"]["ref"] != "k1":
        raise ValueError("go/no-go reference must be the trained candidate at K=1")
    data = build()
    info = manifest(data)
    tokenizer = ByteTokenizer()
    source = HOME / "research/experiments/fixtures/tokenizer_corpus.jsonl"
    split = HOME / "research/prototype/splits.json"
    docs, audit = load_splits(source, split, tokenizer)
    text = docs["test"]
    results = []
    started = time.perf_counter()
    deadline = started + float(base["wall"])
    with tempfile.TemporaryDirectory(prefix="ablate-") as folder:
        paths = {}
        for arch in ARCHS:
            for seed in base["seeds"]:
                guard(deadline)
                config = make(base, arch, seed)
                path = Path(folder) / f"{arch}-{seed}.pt"
                result = run(config, int(base["updates"]), path)
                results.append({"arch": arch, "seed": seed, "train": result})
                paths[(arch, seed)] = (config, path)
        val = {}
        for arch in ("loop-v1", "dual-v1"):
            vals = [item["train"]["validation"]["nll"] for item in results
                    if item["arch"] == arch]
            val[arch] = statistics.mean(vals)
        best = min(val, key=val.get)
        for seed in base["seeds"]:
            config = make(base, best, seed, "final")
            path = Path(folder) / f"final-{seed}.pt"
            result = run(config, int(base["updates"]), path)
            results.append({"arch": best, "seed": seed, "loss": "final", "train": result})
            paths[(f"{best}-final", seed)] = (config, path)
        evals = {}
        for arch in ARCHS:
            counts = (1, 2, 4) if arch in {"loop-v1", "dual-v1"} else (1,)
            evals[arch] = {}
            for count in counts:
                evals[arch][str(count)] = []
                for seed in base["seeds"]:
                    guard(deadline)
                    config, path = paths[(arch, seed)]
                    payload = torch.load(path, map_location="cpu", weights_only=True)
                    model = load(config, payload)
                    item = evaluate(model, arch, data, count, int(base["samples"]),
                                    text, list(base["copy"]))
                    item["seed"] = seed
                    item["params"] = sum(value.numel() for value in model.parameters())
                    item["checkpoint"] = path.stat().st_size
                    item["apps"] = (
                        count * (2 if arch == "dual-v1" else 1)
                        if arch in {"loop-v1", "dual-v1"}
                        else int(config["model"].get("depth", 1)) if arch == "stack-v1" else 1
                    )
                    evals[arch][str(count)].append(item)
        evals[f"{best}-final"] = {"4": []}
        for seed in base["seeds"]:
            guard(deadline)
            config, path = paths[(f"{best}-final", seed)]
            payload = torch.load(path, map_location="cpu", weights_only=True)
            model = load(config, payload)
            item = evaluate(model, best, data, 4, int(base["samples"]), text,
                            list(base["copy"]))
            item["seed"] = seed
            item["params"] = sum(value.numel() for value in model.parameters())
            item["checkpoint"] = path.stat().st_size
            item["apps"] = 8 if best == "dual-v1" else 4
            evals[f"{best}-final"]["4"].append(item)
    plain = {item["seed"]: item for item in evals["plain-v1"]["1"]}
    same = {item["seed"]: item for item in evals[best]["1"]}
    cand = evals[best]["4"]
    final = evals[f"{best}-final"]["4"]
    gains = {"extra": [], "transfer": [], "text": []}
    for item in cand:
        prior = same[item["seed"]]
        gains["extra"].append(
            item["tasks"]["graph"]["extrapolation"]["accuracy"]
            - prior["tasks"]["graph"]["extrapolation"]["accuracy"])
        gains["transfer"].append(
            item["tasks"]["chain"]["extrapolation"]["accuracy"]
            - prior["tasks"]["chain"]["extrapolation"]["accuracy"])
        gains["text"].append(plain[item["seed"]]["language"]["bpb"]
                              - item["language"]["bpb"])
    need = int(base["go"]["seeds"])
    passed = (sum(value >= float(base["go"]["extra"]) for value in gains["extra"]) >= need
              and sum(value >= float(base["go"]["transfer"]) for value in gains["transfer"]) >= need
              and all(-value <= float(base["go"]["bpb"]) for value in gains["text"]))
    comps = {
        "steps": {"plain": [item["tasks"]["graph"]["extrapolation"]["accuracy"]
                             for item in plain.values()],
                  "one": [item["tasks"]["graph"]["extrapolation"]["accuracy"]
                          for item in same.values()],
                  "four": [item["tasks"]["graph"]["extrapolation"]["accuracy"]
                           for item in cand]},
        "states": {"single": [item["tasks"]["graph"]["extrapolation"]["accuracy"]
                               for item in evals["loop-v1"]["4"]],
                   "dual": [item["tasks"]["graph"]["extrapolation"]["accuracy"]
                            for item in evals["dual-v1"]["4"]]},
        "loss": {"deep": [item["tasks"]["graph"]["extrapolation"]["accuracy"]
                           for item in cand],
                 "final": [item["tasks"]["graph"]["extrapolation"]["accuracy"]
                           for item in final]},
    }
    report = {
        "version": 1,
        "protocol": {"seeds": base["seeds"], "updates": base["updates"],
                     "samples": base["samples"],
                     "family": base["family"], "selection": "mean graph validation byte NLL",
                     "selected": best, "validation": val,
                     "weights": base["weights"], "final": True,
                     "flops": "not measured; block applications are reported as compute proxy"},
        "thresholds": base["go"], "gains": gains,
        "decision": {"result": "go" if passed else "no-go", "passed": passed,
                     "rule": "registered per-seed graph extrapolation and graph-to-chain gains plus held-out text BPB regression"},
        "comparisons": comps, "results": results, "evals": evals,
        "hashes": {"config": hashlib.sha256((ROOT / "ablation.json").read_bytes()).hexdigest(),
                   "generator": info["source"], "splits": info["hashes"],
                   "text": audit["source_sha256"], "split": audit["split_sha256"],
                   "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "train": hashlib.sha256((ROOT / "train.py").read_bytes()).hexdigest(),
                   "stack": hashlib.sha256((ROOT / "stack.py").read_bytes()).hexdigest(),
                   "model": hashlib.sha256((ROOT / "model.py").read_bytes()).hexdigest(),
                   "dual": hashlib.sha256((ROOT / "dual.py").read_bytes()).hexdigest(),
                   "tiny": hashlib.sha256((HOME / "research/experiments/tiny_gpt.py").read_bytes()).hexdigest()},
        "runtime": {"wall": time.perf_counter() - started,
                    "rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    "threads": torch.get_num_threads(), "device": "cpu",
                    "dtype": "float32", "timeout": base["wall"],
                    "python": sys.version.split()[0], "torch": torch.__version__,
                    "platform": platform.platform()},
        "limits": ["Synthetic algorithmic task evidence only; no natural-language capability claim.",
                   "Two examples per generated evaluation slice produce very wide intervals.",
                   "Only graph-to-chain transfer was trained as a source-target pair.",
                   "K=8 was not trained and is not evaluated.",
                   "FLOPs were not instrumented; block applications proxy the known model work.",
                   "RSS is a process high-water mark across this run, not an isolated per-condition peak.",
                   "The copy strings are project-authored; the text test split is held out."],
    }
    (ROOT / "out.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
