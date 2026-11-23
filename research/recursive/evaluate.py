"""Scoring and small deterministic baselines for synthetic reasoning tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import sys
import time
from collections import deque
from pathlib import Path

from research.recursive.tasks import CONTEXT, CROSS, build, crosses, manifest


def guard() -> int:
    limit = 2 * 1024**3
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = min(limit, hard) if hard >= 0 else limit
    if soft >= 0:
        cap = min(cap, soft)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    return cap


def normalize(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def edges(prompt: str) -> list[tuple[str, str]]:
    value = prompt.split("Edges: ", 1)[1].split(". Return nodes joined by >.", 1)[0]
    rows = []
    for item in value.split(","):
        left, right = item.split(">", 1)
        rows.append((left, right))
    return rows


def path(prompt: str) -> list[str]:
    graph = {}
    for left, right in edges(prompt):
        graph.setdefault(left, []).append(right)
    queue = deque([("S", ["S"])])
    seen = {"S"}
    while queue:
        node, route = queue.popleft()
        if node == "T":
            return route
        for next in sorted(graph.get(node, [])):
            if next not in seen:
                seen.add(next)
                queue.append((next, [*route, next]))
    return []


def trace(program: str) -> str:
    stack = []
    store = {}
    for row in program.split(";"):
        bits = row.strip().split()
        if len(bits) == 2 and bits[0] == "PUSH":
            stack.append(int(bits[1]))
        elif len(bits) == 2 and bits[0] == "STORE":
            if not stack:
                raise ValueError("empty stack")
            store[bits[1]] = stack.pop()
        elif len(bits) == 2 and bits[0] == "LOAD":
            stack.append(store[bits[1]])
        elif len(bits) == 1 and bits[0] in {"ADD", "SUB", "MUL"}:
            right = stack.pop()
            left = stack.pop()
            stack.append(left + right if bits[0] == "ADD" else left - right if bits[0] == "SUB" else left * right)
        else:
            raise ValueError("invalid DSL instruction")
    if len(stack) != 0 or "out" not in store:
        raise ValueError("invalid DSL final state")
    return str(store["out"])


def solve(row: dict[str, object], kind: str = "rules") -> str:
    if kind == "constant":
        return "0"
    family = row["family"]
    prompt = row["prompt"]
    if family == "addition":
        value = prompt.split("Add ", 1)[1].split(".", 1)[0]
        left, right = (int(item.strip()) for item in value.split("+", 1))
        return str(left + right)
    if family == "bracket":
        value = prompt.rsplit("Sequence: ", 1)[1].split(".", 1)[0]
        depth = 0
        good = True
        for char in value:
            depth += 1 if char == "(" else -1
            good = good and depth >= 0
        return "yes" if good and depth == 0 else "no"
    if family == "graph":
        return ">".join(path(prompt))
    if family == "chain":
        value = prompt.split("equations: ", 1)[1].split(". What", 1)[0]
        values = {}
        for item in value.split(";"):
            name, expr = item.split("=", 1)
            if expr.startswith("v"):
                point = next((at for at, char in enumerate(expr[1:], 1) if char in "+-*"), None)
                base = values[expr[:point]]
                amount = int(expr[point + 1:])
                op = expr[point]
                values[name] = base + amount if op == "+" else base - amount if op == "-" else base * amount
            else:
                values[name] = int(expr)
        return str(values["v" + str(max(int(name[1:]) for name in values))])
    if family == "trace":
        program = prompt.split("DSL: ", 1)[1].split(". What", 1)[0]
        return trace(program)
    raise ValueError(f"unknown family: {family}")


def score(row: dict[str, object], output: str) -> dict[str, object]:
    family = row["family"]
    value = normalize(output)
    target = normalize(row["target"])
    error = ""
    good = False
    if family == "graph":
        try:
            route = [part.strip() for part in output.split(">")]
            links = set(edges(row["prompt"]))
            shortest = path(row["prompt"])
            valid = bool(route) and route[0] == "S" and route[-1] == "T"
            valid = valid and all((left, right) in links for left, right in zip(route, route[1:]))
            if not valid:
                error = "invalid"
            elif len(route) != len(shortest):
                error = "suboptimal"
            else:
                good = True
        except (IndexError, ValueError):
            error = "format"
    elif family == "bracket":
        if value not in {"yes", "no"}:
            error = "format"
        else:
            good = value == target
            error = "balance" if not good else ""
    elif family in {"addition", "chain", "trace"}:
        try:
            int(value)
            good = value == target
            error = "arithmetic" if not good else ""
        except ValueError:
            error = "format"
    return {"correct": good, "error": error}


def interval(correct: int, count: int) -> list[float] | None:
    if count == 0:
        return None
    rate = correct / count
    scale = 1 + 1.96**2 / count
    center = (rate + 1.96**2 / (2 * count)) / scale
    radius = 1.96 * math.sqrt(rate * (1 - rate) / count + 1.96**2 / (4 * count**2)) / scale
    return [max(0.0, center - radius), min(1.0, center + radius)]


def metric(rows: list[dict[str, object]], kind: str) -> dict[str, object]:
    latencies = []
    errors = {}
    correct = 0
    for row in rows:
        started = time.perf_counter()
        output = solve(row, kind)
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)
        result = score(row, output)
        correct += int(result["correct"])
        if result["error"]:
            errors[result["error"]] = errors.get(result["error"], 0) + 1
    times = sorted(latencies)
    count = len(rows)
    return {"samples": count, "correct": correct, "accuracy": correct / count if count else None,
            "ci95": interval(correct, count), "errors": errors,
            "latency": {"mean": statistics.mean(latencies) if latencies else None,
                        "p50": statistics.median(latencies) if latencies else None,
                        "p95": times[max(0, math.ceil(0.95 * count) - 1)] if count else None}}


def check(data: dict[str, object]) -> dict[str, object]:
    count = 0
    for parts in data["families"].values():
        for rows in parts.values():
            for row in rows:
                result = score(row, row["target"])
                if not result["correct"]:
                    raise AssertionError(f"oracle scorer self-check failed for {row['id']}")
                count += 1
    return {"passed": True, "samples": count, "source": "gold targets used for scorer self-check only"}


def evaluate() -> tuple[dict[str, object], dict[str, object]]:
    started = time.perf_counter()
    cpu = time.process_time()
    data = build()
    info = manifest(data)
    tracks = {}
    for family, parts in data["families"].items():
        tracks[family] = {}
        for split, rows in parts.items():
            tracks[family][split] = {"constant": metric(rows, "constant"), "rules": metric(rows, "rules")}
    transfer = []
    for row in crosses(data):
        transfer.append({"source": row["source"], "target": row["target"],
                         "relation": row["relation"], "samples": len(row["evaluation"]),
                         "baseline": metric(row["evaluation"], "rules"),
                         "interpretation": "scorer and rule-baseline reference; no strategy transfer was trained"})
    result = {"version": 1, "mode": "synthetic", "params": 0, "loops": 0,
              "context": CONTEXT, "selfcheck": check(data), "tracks": tracks, "cross": transfer,
              "runtime": {"wall": time.perf_counter() - started,
                          "cpu": time.process_time() - cpu,
                          "rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                          "memory": 2 * 1024**3, "timeout": 120, "threads": 1,
                          "hashes": {"tasks": info["source"],
                                     "evaluate": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                     "test": hashlib.sha256(Path(__file__).with_name("test.py").read_bytes()).hexdigest()},
                          "python": sys.version.split()[0], "platform": platform.platform()},
              "limits": ["synthetic algorithmic tasks only", "no natural-language reasoning evidence",
                         "no model or strategy training was run", "rule baseline is a reference, not a learned system"]}
    return result, info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("research/recursive/result.json"))
    args = parser.parse_args()
    guard()
    result, info = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path("research/recursive/manifest.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
