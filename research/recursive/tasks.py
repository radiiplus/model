"""Deterministic synthetic procedures for later recurrent-model evaluation."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


VERSION = 1
CONTEXT = 2048
SPLITS = ("train", "validation", "interpolation", "extrapolation")
COUNTS = {"train": 12, "validation": 4, "interpolation": 4, "extrapolation": 4}
BANDS = {
    "addition": {"train": (1, 2), "validation": (1, 2), "interpolation": (1, 2), "extrapolation": (3, 4)},
    "bracket": {"train": (1, 3), "validation": (1, 3), "interpolation": (1, 3), "extrapolation": (4, 6)},
    "graph": {"train": (2, 3), "validation": (2, 3), "interpolation": (2, 3), "extrapolation": (4, 5)},
    "chain": {"train": (2, 3), "validation": (2, 3), "interpolation": (2, 3), "extrapolation": (4, 5)},
    "trace": {"train": (2, 4), "validation": (2, 4), "interpolation": (2, 4), "extrapolation": (6, 8)},
}
SEEDS = {
    "addition": {"train": 1101, "validation": 1102, "interpolation": 1103, "extrapolation": 1104},
    "bracket": {"train": 2201, "validation": 2202, "interpolation": 2203, "extrapolation": 2204},
    "graph": {"train": 3301, "validation": 3302, "interpolation": 3303, "extrapolation": 3304},
    "chain": {"train": 4401, "validation": 4402, "interpolation": 4403, "extrapolation": 4404},
    "trace": {"train": 5501, "validation": 5502, "interpolation": 5503, "extrapolation": 5504},
}
CROSS = (
    {"source": "graph", "target": "chain", "relation": "dependency"},
    {"source": "chain", "target": "graph", "relation": "dependency"},
    {"source": "bracket", "target": "trace", "relation": "stack"},
    {"source": "trace", "target": "bracket", "relation": "stack"},
)


def example(family: str, split: str, index: int, level: int, prompt: str, target: str) -> dict[str, str | int]:
    return {"id": f"{family}-{split}-{index:03d}", "family": family, "split": split,
            "level": level, "prompt": prompt, "target": target}


def draw(family: str, rng: random.Random, level: int) -> tuple[str, str]:
    if family == "addition":
        size = level
        low = 0 if size == 1 else 10 ** (size - 1)
        high = 10 ** size
        left = rng.randrange(low, high)
        right = rng.randrange(low, high)
        left, right = sorted((left, right))
        return f"Add {left} + {right}. Return only the integer.", str(left + right)
    if family == "bracket":
        depth = level
        count = rng.randint(depth, depth * 3)
        chars = []
        opened = 0
        closed = 0
        for _ in range(count * 2):
            if opened < count and (opened - closed < depth) and (opened == closed or rng.random() < 0.58):
                chars.append("(")
                opened += 1
            else:
                chars.append(")")
                closed += 1
        if rng.random() < 0.5:
            at = rng.randrange(len(chars))
            chars[at] = ")" if chars[at] == "(" else "("
        text = "".join(chars)
        valid = 0
        live = 0
        for char in text:
            live += 1 if char == "(" else -1
            valid = valid + 1 if live >= 0 else -100000
        balanced = live == 0 and valid >= 0
        return f"Is this bracket sequence balanced? Sequence: {text}. Return yes or no.", "yes" if balanced else "no"
    if family == "graph":
        hops = level
        nodes = [[f"N{layer}{code:04d}" for code in rng.sample(range(10000), 3)]
                 for layer in range(hops - 1)]
        chosen = [rng.choice(row) for row in nodes]
        edges = [("S", node) for node in nodes[0]
                 if node == chosen[0] or rng.random() < 0.6]
        for index, (before, after) in enumerate(zip(nodes, nodes[1:])):
            edges.extend((left, right) for left in before for right in after
                         if (left == chosen[index] and right == chosen[index + 1]) or rng.random() < 0.35)
        edges.extend((node, "T") for node in nodes[-1]
                     if node == chosen[-1] or rng.random() < 0.6)
        edges.sort()
        route = ["S", *chosen, "T"]
        edgestr = ",".join(f"{left}>{right}" for left, right in edges)
        return f"Find a shortest directed path from S to T. Edges: {edgestr}. Return nodes joined by >.", ">".join(route)
    if family == "chain":
        value = rng.randint(1, 9)
        rows = [f"v0={value}"]
        for step in range(1, level + 1):
            op = rng.choice(("+", "-", "*"))
            amount = rng.randint(1, 3)
            value = value + amount if op == "+" else value - amount if op == "-" else value * amount
            rows.append(f"v{step}=v{step - 1}{op}{amount}")
        prompt = f"Resolve these variable equations: {';'.join(rows)}. What is v{level}? Return only the integer."
        return prompt, str(value)
    if family == "trace":
        value = rng.randint(1, 9)
        rows = [f"PUSH {value}"]
        for _ in range(level):
            amount = rng.randint(1, 3)
            op = rng.choice(("ADD", "SUB", "MUL"))
            rows.extend((f"PUSH {amount}", op))
            value = value + amount if op == "ADD" else value - amount if op == "SUB" else value * amount
        rows.append("STORE out")
        return f"Trace the safe stack DSL: {';'.join(rows)}. What value is stored in out? Return only the integer.", str(value)
    raise ValueError(f"unknown family: {family}")


def build() -> dict[str, object]:
    families = {}
    for family in BANDS:
        parts = {}
        seen = set()
        for split in SPLITS:
            rng = random.Random(SEEDS[family][split])
            low, high = BANDS[family][split]
            rows = []
            for index in range(COUNTS[split]):
                for attempt in range(10000):
                    level = rng.randint(low, high)
                    prompt, target = draw(family, rng, level)
                    if prompt not in seen:
                        seen.add(prompt)
                        rows.append(example(family, split, index, level, prompt, target))
                        break
                else:
                    raise RuntimeError("could not draw a unique task instance")
            parts[split] = rows
        families[family] = parts
    return {"version": VERSION, "context": CONTEXT, "families": families}


def manifest(data: dict[str, object]) -> dict[str, object]:
    groups = data["families"]
    hashes = {}
    sizes = {}
    longest = 0
    for family, parts in groups.items():
        hashes[family] = {}
        sizes[family] = {}
        for split, rows in parts.items():
            packed = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            hashes[family][split] = hashlib.sha256(packed).hexdigest()
            sizes[family][split] = len(rows)
            longest = max(longest, *(len((row["prompt"] + row["target"]).encode("utf-8")) for row in rows))
    if longest > data["context"]:
        raise ValueError("generated prompt plus answer exceeds the configured byte context")
    source = Path(__file__).read_bytes()
    return {"version": VERSION, "generator": VERSION, "source": hashlib.sha256(source).hexdigest(),
            "seeds": SEEDS, "bands": BANDS, "counts": COUNTS,
            "sizes": sizes, "hashes": hashes, "context": data["context"],
            "longest": longest, "cross": CROSS}


def crosses(data: dict[str, object]) -> list[dict[str, object]]:
    families = data["families"]
    rows = []
    for pair in CROSS:
        source = pair["source"]
        target = pair["target"]
        rows.append({"source": source, "target": target, "relation": pair["relation"],
                     "train": families[source]["train"],
                     "evaluation": families[target]["interpolation"] + families[target]["extrapolation"]})
    return rows
