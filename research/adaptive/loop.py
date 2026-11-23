"""Offline symbolic discovery and transfer validation for reasoning strategies."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sqlite3
import time
from pathlib import Path
from typing import Any

from research.recursive.evaluate import score, solve
from research.recursive.tasks import VERSION, build, manifest


GATE = {"accuracy": 0.75, "gain": 0.5, "families": ["trace"],
        "probe": ["bracket"],
        "splits": ["interpolation", "extrapolation"], "protected": "addition"}
ROOT = Path(__file__).resolve().parent
BASE = Path(__file__).resolve().parents[1] / "experiments/results/baseline_tiny_gpt.pt"


def digest(value: Any) -> str:
    raw = value if isinstance(value, bytes) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def pack(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def inputhash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class Archive:
    def __init__(self, path: str | Path = ":memory:"):
        self.sql = sqlite3.connect(str(path), timeout=0.1)
        self.sql.row_factory = sqlite3.Row
        self.sql.execute("PRAGMA foreign_keys=ON")
        self.sql.execute("PRAGMA journal_mode=WAL")
        self.sql.execute("PRAGMA synchronous=FULL")
        self.sql.executescript("""
        CREATE TABLE IF NOT EXISTS episodes(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS strategies(id TEXT NOT NULL, version INTEGER NOT NULL,
            payload TEXT NOT NULL, PRIMARY KEY(id,version));
        CREATE TRIGGER IF NOT EXISTS ep1 BEFORE UPDATE ON episodes BEGIN SELECT RAISE(ABORT,'immutable'); END;
        CREATE TRIGGER IF NOT EXISTS ep2 BEFORE DELETE ON episodes BEGIN SELECT RAISE(ABORT,'immutable'); END;
        CREATE TRIGGER IF NOT EXISTS st1 BEFORE UPDATE ON strategies BEGIN SELECT RAISE(ABORT,'immutable'); END;
        CREATE TRIGGER IF NOT EXISTS st2 BEFORE DELETE ON strategies BEGIN SELECT RAISE(ABORT,'immutable'); END;
        """)
        self.sql.commit()

    def add(self, task: dict[str, Any], answer: str, result: dict[str, Any],
            actions: list[dict[str, Any]], cost: dict[str, Any],
            strategies: list[str] | None = None,
            evidence: list[str] | None = None) -> dict[str, Any]:
        text = task["prompt"]
        item = {"family": task["family"], "instance": task["id"],
                "split": task["split"], "generator": f"tasks-v{VERSION}",
                "inputhash": inputhash(text),
                "evidence": sorted(set(evidence or [])),
                "strategies": list(strategies or []), "actions": actions,
                "decision": {"mode": "base", "strategies": list(strategies or [])},
                "tools": [],
                "answer": answer, "result": result,
                "cost": cost, "model": "frozen-byte-tiny-v1",
                "adapter": "none-v1", "strategy": "none-v1",
                "dataset": f"tasks-v{VERSION}"}
        item["id"] = digest([item["family"], item["instance"], item["inputhash"]])
        with self.sql:
            self.sql.execute("INSERT OR IGNORE INTO episodes VALUES(?,?)",
                             (item["id"], pack(item)))
        return item

    def episodes(self, family: str | None = None) -> list[dict[str, Any]]:
        rows = self.sql.execute("SELECT payload FROM episodes ORDER BY id").fetchall()
        items = [json.loads(row[0]) for row in rows]
        return [item for item in items if family is None or item["family"] == family]

    def save(self, item: dict[str, Any]) -> dict[str, Any]:
        item = dict(item)
        item.setdefault("version", 1)
        item.setdefault("previous", None)
        with self.sql:
            self.sql.execute("INSERT OR IGNORE INTO strategies VALUES(?,?,?)",
                             (item["id"], item["version"], pack(item)))
        return item

    def latest(self, key: str) -> dict[str, Any]:
        row = self.sql.execute(
            "SELECT payload FROM strategies WHERE id=? ORDER BY version DESC LIMIT 1",
            (key,),
        ).fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row[0])

    def versions(self, key: str) -> list[dict[str, Any]]:
        return [json.loads(row[0]) for row in self.sql.execute(
            "SELECT payload FROM strategies WHERE id=? ORDER BY version", (key,))]

    def change(self, key: str, status: str, report: dict[str, Any] | None = None) -> dict[str, Any]:
        if status not in {"candidate", "validated", "deprecated", "rejected"}:
            raise ValueError("invalid strategy status")
        old = self.latest(key)
        if status == "validated" and (not isinstance(report, dict)
                or report.get("gate") != "adaptive-transfer-v1"
                or report.get("passed") is not True
                or not report.get("heldout")
                or not set(report.get("validationfamilies", [])) - set(old["train"])
                or not report.get("tracks")):
            raise ValueError("promotion requires a passing independent transfer report")
        item = dict(old)
        item["version"] = old["version"] + 1
        item["previous"] = old["version"]
        item["status"] = status
        if report is not None:
            item["report"] = report
            if status == "rejected":
                item["counterexamples"] = sorted(set(item.get("counterexamples", []))
                                                  | set(report.get("counterexamples", [])))
            if status == "validated":
                item["valid"] = list(report["criterion"]["families"])
        with self.sql:
            self.sql.execute("INSERT INTO strategies VALUES(?,?,?)",
                             (key, item["version"], pack(item)))
        return item

    def deprecate(self, key: str) -> dict[str, Any]:
        return self.change(key, "deprecated")

    def rollback(self, key: str, version: int) -> dict[str, Any]:
        rows = {item["version"]: item for item in self.versions(key)}
        if version not in rows:
            raise KeyError(version)
        old = self.latest(key)
        item = dict(rows[version])
        item["version"] = old["version"] + 1
        item["previous"] = old["version"]
        with self.sql:
            self.sql.execute("INSERT INTO strategies VALUES(?,?,?)",
                             (key, item["version"], pack(item)))
        return item

    def select(self, family: str) -> list[dict[str, Any]]:
        keys = [row[0] for row in self.sql.execute(
            "SELECT id FROM strategies GROUP BY id ORDER BY id")]
        chosen = []
        for key in keys:
            item = self.latest(key)
            if item["status"] == "validated" and family in item["valid"]:
                chosen.append(item)
        return chosen

    def close(self) -> None:
        self.sql.close()

    def __enter__(self) -> "Archive":
        return self

    def __exit__(self, kind: Any, value: Any, trace: Any) -> None:
        self.close()


def actions(task: dict[str, Any]) -> list[dict[str, Any]]:
    family = task["family"]
    prompt = task["prompt"]
    if family == "chain":
        rows = prompt.split("equations: ", 1)[1].split(". What", 1)[0].split(";")
        return [{"op": "update", "step": index} for index, row in enumerate(rows[1:], 1)] + [
            {"op": "check", "step": len(rows)}]
    if family == "trace":
        program = prompt.split("DSL: ", 1)[1].split(". What", 1)[0]
        ops = [row for row in program.split(";") if row in {"ADD", "SUB", "MUL", "STORE out"}]
        return [{"op": "apply", "step": index, "kind": row.split()[0]}
                for index, row in enumerate(ops, 1)]
    if family == "bracket":
        seq = prompt.rsplit("Sequence: ", 1)[1].split(".", 1)[0]
        return [{"op": "scan", "step": index} for index, _ in enumerate(seq, 1)] + [
            {"op": "check", "step": len(seq)}]
    return [{"op": "solve", "step": 1}]


def propose(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Extract only from episode records; this function has no task-label input."""
    eligible = [item for item in rows if item.get("split") == "train"]
    good = [item for item in eligible if item["result"].get("correct")]
    unique = {item["inputhash"]: item for item in good}
    families = sorted({item["family"] for item in eligible})
    if len(families) != 1 or len(unique) < 3:
        return None
    family = families[0]
    if family == "chain" and all(any(action["op"] == "update" for action in item["actions"])
                                  for item in unique.values()):
        proc = "stepwise"
        principle = "Carry explicit state through ordered updates, then check the final invariant."
        desc = "Parse ordered assignments or stack instructions, update state one operation at a time, and check the final state."
        pre = ["ordered assignments or stack operations", "each transition is locally defined"]
        avoid = ["unordered dependencies", "ambiguous or unsupported operations"]
        benefit = "More exact answers than a constant responder on unseen ordered-state tasks."
    else:
        proc = "memorize"
        principle = "Reuse an observed answer only for an identical input."
        desc = "Hash the input and return an answer only when its hash and family match exactly."
        pre = [f"exact input hash in {family} training episodes"]
        avoid = ["any unseen input", "any other task family"]
        benefit = "Exact repeat recall only; no transfer is expected."
    refs = sorted(item["id"] for item in unique.values())
    key = digest([proc, principle, refs])
    return {"id": key, "version": 1, "previous": None, "principle": principle,
            "procedure": proc, "description": desc, "preconditions": pre, "avoid": avoid,
            "benefit": benefit, "evidence": sorted({ref for item in unique.values()
                                                       for ref in item["evidence"]}),
            "episodes": refs, "counterexamples": [], "train": families,
            "valid": [], "report": {}, "status": "candidate"}


def base(task: dict[str, Any]) -> str:
    return "no" if task["family"] == "bracket" else "S>T" if task["family"] == "graph" else "0"


def run(strategy: dict[str, Any] | None, task: dict[str, Any],
        episodes: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    if strategy is None:
        return base(task), []
    if strategy["procedure"] == "stepwise":
        return solve(task, "rules"), actions(task)
    if strategy["procedure"] == "memorize":
        key = inputhash(task["prompt"])
        found = next((item["answer"] for item in episodes
                      if item["inputhash"] == key and item["family"] == task["family"]), "")
        return found, [{"op": "lookup", "step": 1}] if found else []
    return base(task), []


def metrics(rows: list[dict[str, Any]], strategy: dict[str, Any] | None,
            episodes: list[dict[str, Any]], budget: dict[str, int]) -> dict[str, Any]:
    start = time.perf_counter()
    correct = 0
    errors: dict[str, int] = {}
    samples = []
    for task in rows:
        answer, used = run(strategy, task, episodes)
        outcome = score(task, answer)
        correct += int(outcome["correct"])
        if outcome["error"]:
            errors[outcome["error"]] = errors.get(outcome["error"], 0) + 1
        samples.append({"id": task["id"], "answer": answer,
                        "correct": outcome["correct"], "error": outcome["error"],
                        "actions": len(used), "budget": budget})
    elapsed = time.perf_counter() - start
    count = len(rows)
    return {"samples": count, "correct": correct,
            "accuracy": correct / count if count else 0.0,
            "errors": errors, "latency": elapsed / count if count else 0.0,
            "sampleset": samples}


def validate(archive: Archive, item: dict[str, Any], data: dict[str, Any],
             episodes: list[dict[str, Any]]) -> dict[str, Any]:
    budget = {"iterations": 1, "actions": 128, "retrievals": 0}
    tracks = {}
    ids = []
    passed = True
    tested = GATE["families"] + GATE["probe"]
    for family in tested:
        tracks[family] = {}
        for split in GATE["splits"]:
            rows = data["families"][family][split]
            ids.extend(row["id"] for row in rows)
            plain = metrics(rows, None, episodes, budget)
            cand = metrics(rows, item, episodes, budget)
            gain = cand["accuracy"] - plain["accuracy"]
            good = (cand["accuracy"] >= GATE["accuracy"] and gain >= GATE["gain"]
                    if family in GATE["families"]
                    else cand["accuracy"] >= plain["accuracy"])
            passed = passed and good
            tracks[family][split] = {"base": plain, "strategy": cand,
                                     "gain": gain, "passed": good,
                                     "criterion": "transfer-gain" if family in GATE["families"]
                                     else "no-regression probe"}
    family = GATE["protected"]
    rows = data["families"][family]["extrapolation"]
    plain = metrics(rows, None, episodes, budget)
    applicable = item if family in item["valid"] else None
    guarded = metrics(rows, applicable, episodes, budget)
    protected = {"base": plain, "strategy": guarded,
                 "regression": plain["accuracy"] - guarded["accuracy"],
                 "passed": guarded["accuracy"] >= plain["accuracy"]}
    passed = passed and protected["passed"]
    report = {"gate": "adaptive-transfer-v1", "passed": passed,
              "criterion": GATE, "tracks": tracks, "protected": protected,
              "heldout": sorted(ids), "heldouthash": digest(sorted(ids)),
              "trainfamilies": item["train"], "validationfamilies": tested,
              "strategyversion": archive.latest(item["id"])["version"] + 1,
              "budget": budget, "leakage": "candidate extractor received training episodes only"}
    report["counterexamples"] = sorted({sample["id"] for track in tracks.values()
                                         for split in track.values()
                                         for sample in split["strategy"]["sampleset"]
                                         if not sample["correct"]})
    archive.change(item["id"], "validated" if passed else "rejected", report)
    return report


def collect(archive: Archive, data: dict[str, Any], info: dict[str, Any]) -> list[dict[str, Any]]:
    rows = data["families"]["chain"]["train"]
    for task in rows:
        start = time.perf_counter()
        answer = solve(task, "rules")
        outcome = score(task, answer)
        cost = {"iterations": 1, "retrievals": 0,
                "seconds": time.perf_counter() - start, "actions": len(actions(task))}
        archive.add(task, answer, outcome, actions(task), cost)
    return archive.episodes("chain")


def generate() -> tuple[dict[str, Any], str]:
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = 1024**3
    if soft >= 0:
        cap = min(cap, soft)
    if hard >= 0:
        cap = min(cap, hard)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    started = time.monotonic()
    data = build()
    info = manifest(data)
    before = hashlib.sha256(BASE.read_bytes()).hexdigest()
    with Archive() as archive:
        train = collect(archive, data, info)
        candidate = propose(train)
        if candidate is None:
            raise AssertionError("training episodes did not yield a candidate")
        archive.save(candidate)
        report = validate(archive, candidate, data, train)
        if time.monotonic() - started > 110:
            raise TimeoutError("adaptive reasoning run exceeded its 110 second wall cap")
        latest = archive.latest(candidate["id"])
        chosen = {family: [item["id"] for item in archive.select(family)]
                  for family in GATE["families"] + GATE["probe"] + [GATE["protected"]]}
        after = hashlib.sha256(BASE.read_bytes()).hexdigest()
        if before != after:
            raise AssertionError("foundation checkpoint changed during symbolic validation")
        result = {"version": 1, "mode": "synthetic-symbolic", "foundation": before,
                  "after": after, "parameters": 0, "loops": 0,
                  "episodes": len(train), "failedepisodes": sum(not row["result"]["correct"]
                                                                   for row in train),
                  "candidate": latest, "promotion": report,
                  "selected": chosen,
                  "generator": info["generator"], "splits": info["hashes"],
                  "source": {"tasks": info["source"],
                             "loop": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                  "runtime": {"wall": time.monotonic() - started,
                              "rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                              "address": cap, "timeout": 110, "device": "cpu",
                              "threads": 1, "model": "not loaded"},
                  "limits": ["Project-generated synthetic tasks only.",
                             "The frozen checkpoint was hashed but not loaded or modified.",
                             "Candidate discovery used chain training episodes only.",
                             "The symbolic procedure calls the authored task reference solver; no model improvement is measured."]}
    return result, before


def markdown(result: dict[str, Any]) -> str:
    report = result["promotion"]
    rows = ["# Patch 8: Adaptive reasoning loop", "", "## Decision", "",
            "The symbolic candidate was " + ("validated" if report["passed"] else "rejected") +
            " by the preregistered transfer gate. This validates the episode/procedure/promotion mechanics only; it is not evidence that a language model learned a reasoning strategy.",
            "", "## Protocol", "",
            "The deterministic extractor read only the 12 `chain` training episodes. It grouped distinct input hashes and proposed an ordered-state-update procedure from their observable update/check actions. It did not receive interpolation or extrapolation labels. The validator compared a constant frozen baseline with the candidate on identical held-out `trace` and `bracket` inputs, and checked `addition` extrapolation as a protected family. Both conditions had one iteration, at most 128 symbolic actions, and zero retrieval calls. The foundation checkpoint was hashed before and after; it was not loaded.",
            "", "## Transfer"]
    for family in report["validationfamilies"]:
        rows.extend(["", f"### {family}", "", "| Split | Base | Strategy | Gain | Gate |", "|---|---:|---:|---:|---|"])
        for split, item in report["tracks"][family].items():
            rows.append(f"| {split} | {item['base']['correct']}/{item['base']['samples']} | {item['strategy']['correct']}/{item['strategy']['samples']} | {item['gain']:.2f} | {'pass' if item['passed'] else 'fail'} |")
    prot = report["protected"]
    rows.extend(["", "### Protected family", "",
                 f"Addition extrapolation: base {prot['base']['correct']}/{prot['base']['samples']}, selected system {prot['strategy']['correct']}/{prot['strategy']['samples']}; regression {prot['regression']:.2f}.",
                 "", "## Cost and limits", "",
                 f"The run recorded {result['episodes']} training episodes and took {result['runtime']['wall']:.3f} s with {result['runtime']['rss']} B peak RSS. There are no neural parameters or recurrent loops in this patch. The checkpoint SHA-256 was `{result['foundation']}` both before and after.",
                 "", "The `stepwise` candidate dispatches into the project-authored deterministic synthetic-task solvers. Thus transfer here means that an abstract procedure record selected a known symbolic solver on separately generated families; it does not show acquired model behavior, natural-language reasoning, or improvement to interactions. Test labels are consumed only by the validator after extraction and are not added to training episodes. Small synthetic slices have wide uncertainty.",
                 "", "Full episode/strategy schemas, split hashes, per-instance outcomes, promotion gate and resource data are in `out.json`.", ""])
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "out.json")
    args = parser.parse_args()
    result, _ = generate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_name("report.md").write_text(markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
