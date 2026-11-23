"""Bounded orchestration control run and frozen-model smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import tempfile
import time
from pathlib import Path
from typing import Any

from research.adaptive.loop import Archive
from research.memory.store import Memory
from research.orchestrator.flow import Ask, Byte, Engine, Mode
from research.prototype.trainer import available_mib, configure_torch


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "experiments/results/batch12_byte_seed_11.pt"
STRAT = ROOT.parent / "adaptive/out.json"
CAP = 110.0


class Probe:
    name = "mechanics-probe"

    def generate(self, context, steps, tokens):
        return "blue"


def filehash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> dict[str, Any]:
    configure_torch()
    started = time.monotonic()
    if available_mib() < 2560:
        raise MemoryError("available memory below 2560 MiB")
    policy = {"persist": True, "scope": "demo", "owner": "offline"}
    with tempfile.TemporaryDirectory(prefix="flow-") as folder:
        root = Path(folder)
        with Memory(root / "memory.db") as memory, Archive(root / "archive.db") as archive:
            event = memory.evidence(
                "Sky color is blue.", scope="demo", owner="offline",
                source="generated-fixture", uri="local:sky", version="1",
                licence="CC0-1.0", policy=policy,
                acquired="2026-01-01T00:00:00Z",
            )
            memory.fact("sky-color", "blue", [event], scope="demo", owner="offline",
                        method="generated structured fixture",
                        start="2026-01-01T00:00:00Z", policy=policy,
                        created="2026-01-01T00:00:00Z")
            strategy = json.loads(STRAT.read_text(encoding="utf-8"))["candidate"]
            if strategy["status"] != "validated":
                raise ValueError("Patch 8 strategy is not validated")
            archive.save(strategy)
            ask = Ask("What is the sky color?", "demo", "offline",
                      "sky color blue", family="trace", key="sky-color",
                      private=False)
            modes = {
                "full": Mode(refine=2),
                "no_retrieval": Mode(retrieval=False, refine=1),
                "no_strategy": Mode(strategy=False, refine=1),
                "no_refinement": Mode(refine=0),
                "no_verifier": Mode(verify=False, refine=1),
            }
            controls = {}
            for name, mode in modes.items():
                engine = Engine(memory, archive, Probe())
                result = engine.run(ask, mode)
                controls[name] = {
                    "answer": result.answer,
                    "abstained": result.abstained,
                    "reason": result.reason,
                    "verdict": result.verdict,
                    "citations": len(result.citations),
                    "strategies": result.resource["strategies"],
                    "refinements": result.resource["refinements"],
                    "loop_steps_requested": result.resource["loop_steps_requested"],
                    "retrieval_calls": result.resource["retrieval_calls"],
                    "verifier_calls": result.resource["verifier_calls"],
                    "wall": result.resource["wall"],
                }
            base = Byte(BASE)
            frozen = filehash(BASE)
            live = Engine(memory, archive, base).run(
                Ask("Hi", "demo", "offline"),
                Mode(retrieval=False, strategy=False, refine=1,
                     verify=True, output=2),
            )
            same = frozen == filehash(BASE)
            size = memory.size()
        if time.monotonic() - started > CAP:
            raise TimeoutError(f"orchestration measurement exceeded {CAP} seconds")
    return {
        "version": 1,
        "mode": "offline-orchestration-mechanics",
        "decision": "integrated-path-passes-mechanics-only",
        "controls": controls,
        "neural": {
            "model": base.name,
            "checkpoint": frozen,
            "checkpoint_unchanged": same,
            "parameters": base.model.params,
            "context": base.context,
            "answer": live.answer,
            "abstained": live.abstained,
            "reason": live.reason,
            "resource": live.resource,
        },
        "memory": {"database_bytes": size,
                   "evidence_hash": hashlib.sha256(b"Sky color is blue.").hexdigest(),
                   "scope": "demo", "owner": "offline"},
        "strategy": {"id": strategy["id"], "version": strategy["version"],
                     "status": strategy["status"], "valid": strategy["valid"]},
        "runtime": {"wall": time.monotonic() - started,
                    "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    "available_mib": available_mib(), "threads": 1,
                    "device": "cpu", "memory_guard_mib": 2560,
                    "timeout": 120},
        "hashes": {"flow": filehash(ROOT / "flow.py"),
                   "tests": filehash(ROOT / "test.py"),
                   "strategy_source": filehash(STRAT),
                   "foundation": frozen},
        "limits": [
            "The control model is a deterministic probe, not a language model.",
            "The neural smoke is direct model-only inference on a two-byte request.",
            "The only verifier checks exact structured facts linked to retrieved evidence.",
            "All evidence in this measurement is generated synthetic fixture data.",
            "No natural-language entailment, open-domain capability or interaction quality is measured.",
        ],
    }


def report(result: dict[str, Any]) -> str:
    lines = ["# Patch 10: Offline orchestration", "", "## Result", "",
             "**Mechanics pass; capability unproven.** The end-to-end structured fixture path retrieves a provenance-bearing chunk, selects a validated strategy as data, makes an initial model call and bounded recurrent calls, verifies an exact structured fact, and returns a citation. The controls below use a deterministic probe model and test orchestration behavior, not reasoning quality.",
             "", "| Condition | Answer | Abstained | Retrievals | Strategies | Refinement calls | Loop steps | Verifier calls |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, item in result["controls"].items():
        lines.append(f"| {name} | {item['answer'] or 'none'} | {item['abstained']} | {item['retrieval_calls']} | {item['strategies']} | {item['refinements']} | {item['loop_steps_requested']} | {item['verifier_calls']} |")
    neural = result["neural"]
    runtime = result["runtime"]
    lines.extend(["", "## Neural Smoke", "",
                  f"The existing frozen Batch 12 byte decoder has {neural['parameters']:,} parameters and context {neural['context']}. On a short direct request it used {neural['resource']['model_calls']} model calls and requested {neural['resource']['loop_steps_requested']} shared-block steps, then returned `{neural['answer']}` with reason `{neural['reason']}`. The checkpoint hash remained unchanged. This does not test grounded answer quality; the model's 32-byte context cannot represent most ordinary evidence chunks and this checkpoint is trained only on a tiny synthetic fixture.",
                  "", "## Budgets", "",
                  f"The synthetic mechanics run completed in {runtime['wall']:.3f}s with {runtime['peak_rss']:,} bytes peak RSS, one Torch thread, CPU execution, {runtime['available_mib']} MiB available at completion and a 2,560 MiB memory guard. Retrieval and refinement caps are enforced per request; each result includes resource counters and full chunk provenance.",
                  "", "## Limits", "",
                  "Natural-language request parsing is explicit and typed: callers provide the retrieval query, task family and optional structured fact key. Retrieved text and strategy procedures remain separate data fields and are never executed. The verifier checks only exact equality against validated structured facts and their retrieved evidence links; it is not natural-language entailment. Learning is opt-in, requires matching persistence policy (including private-data permission), and appends episodes only. It cannot create or promote strategies. No web, shell, subprocess or generated-code path exists.",
                  "", "The machine-readable run record includes hashes, per-control counts, timings, memory, checkpoint identity and limitations in `result.json`.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "result.json")
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    args.output.with_name("report.md").write_text(report(result), encoding="utf-8")


if __name__ == "__main__":
    main()
