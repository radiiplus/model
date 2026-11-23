#!/usr/bin/env python3
"""Tiny deterministic curation ablation; standard library only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "experiments" / "fixtures" / "tiny_corpus.jsonl"
RESULTS = ROOT / "experiments" / "results"
WORD_RE = re.compile(r"[\w]+", re.UNICODE)
BOILERPLATE = {"this document was processed by example archive. distribution copy 2035."}


def available_memory_mib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable not exposed; refusing to run without memory guard")


def load_records(path: Path = INPUT) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def strip_boilerplate(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if normalize(line) not in BOILERPLATE).strip()


def words(text: str) -> list[str]:
    return WORD_RE.findall(normalize(text))


def shingles(text: str, width: int = 3) -> set[str]:
    tokens = words(text)
    if len(tokens) <= width:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + width]) for i in range(len(tokens) - width + 1)}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def exact_groups(records: list[dict[str, Any]]) -> list[set[int]]:
    groups: dict[str, set[int]] = {}
    for i, row in enumerate(records):
        groups.setdefault(digest(normalize(row["text"])), set()).add(i)
    return [group for group in groups.values() if len(group) > 1]


def minhash_signature(shingle_set: set[str], permutations: int = 64) -> tuple[int, ...]:
    if not shingle_set:
        return tuple([2**64 - 1] * permutations)
    values = [int.from_bytes(hashlib.blake2b(shingle.encode(), digest_size=8).digest(), "big") for shingle in shingle_set]
    # Fixed affine permutations over 64-bit integers; deterministic across Python versions.
    mask = 2**64 - 1
    sig = []
    for index in range(permutations):
        a = (0x9E3779B185EBCA87 + 2 * index) | 1
        b = 0xC2B2AE3D27D4EB4F * (index + 1) & mask
        sig.append(min((a * value + b) & mask for value in values))
    return tuple(sig)


def minhash_groups(records: list[dict[str, Any]], threshold: float = 0.70) -> tuple[list[set[int]], list[tuple[int, int, float]]]:
    signatures = [minhash_signature(shingles(row["text"])) for row in records]
    pairs = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            score = sum(a == b for a, b in zip(signatures[i], signatures[j])) / len(signatures[i])
            if score >= threshold:
                pairs.append((i, j, score))
    return connected_components(len(records), [(i, j) for i, j, _ in pairs]), pairs


def simhash_signature(text: str, bits: int = 64) -> int:
    shingle_set = shingles(text)
    if not shingle_set:
        return 0
    accum = [0] * bits
    for shingle in shingle_set:
        value = int.from_bytes(hashlib.blake2b(shingle.encode(), digest_size=8).digest(), "big")
        for bit in range(bits):
            accum[bit] += 1 if value & (1 << bit) else -1
    return sum((1 << bit) for bit, score in enumerate(accum) if score >= 0)


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def simhash_groups(records: list[dict[str, Any]], max_distance: int = 5) -> tuple[list[set[int]], list[tuple[int, int, int]]]:
    signatures = [simhash_signature(row["text"]) for row in records]
    pairs = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            distance = hamming(signatures[i], signatures[j])
            if distance <= max_distance:
                pairs.append((i, j, distance))
    return connected_components(len(records), [(i, j) for i, j, _ in pairs]), pairs


def connected_components(size: int, pairs: list[tuple[int, int]]) -> list[set[int]]:
    parent = list(range(size))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a
    groups: dict[int, set[int]] = {}
    for i in range(size):
        groups.setdefault(find(i), set()).add(i)
    return [g for g in groups.values() if len(g) > 1]


def quality_signals(row: dict[str, Any]) -> dict[str, float | bool | str]:
    text = row["text"]
    tokens = words(text)
    counts = Counter(tokens)
    repeated_fraction = max(counts.values(), default=0) / max(len(tokens), 1)
    unique_ratio = len(counts) / max(len(tokens), 1)
    domain = row["domain"]
    # A transparent diagnostic score; domain-specific signals avoid punishing short code/math.
    if domain == "code":
        score = min(1.0, 0.45 + 0.1 * bool(re.search(r"[{}():=]", text)) + 0.08 * len(set(text.splitlines())) / max(len(text.splitlines()), 1))
    elif domain == "math":
        score = min(1.0, 0.45 + 0.12 * bool(re.search(r"\d|[=+*/]", text)) + 0.1 * (len(tokens) >= 8))
    else:
        score = min(1.0, 0.35 + 0.4 * unique_ratio + 0.15 * (len(tokens) >= 12) + 0.1 * text.rstrip().endswith((".", "!", "?")))
    low_quality_flag = repeated_fraction >= 0.7 and len(tokens) >= 10
    return {"quality_score": round(score, 4), "repeated_token_fraction": round(repeated_fraction, 4),
            "unique_token_ratio": round(unique_ratio, 4), "low_quality_flag": low_quality_flag,
            "domain": domain}


def quality_filter(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept, removed = [], []
    for row in records:
        signals = quality_signals(row)
        annotated = dict(row, quality_signals=signals)
        if signals["low_quality_flag"]:
            removed.append(annotated)
        else:
            kept.append(annotated)
    return kept, removed


def group_labels(groups: list[set[int]], records: list[dict[str, Any]]) -> tuple[int, list[str]]:
    assigned = set()
    false_positive_ids = []
    for group in groups:
        ids = sorted(records[i]["id"] for i in group)
        for i in group:
            if i in assigned:
                continue
            assigned.add(i)
        # A group is false positive if the fixture labels its members as distinct.
        label_groups = {records[i].get("duplicate_group") for i in group}
        if None in label_groups or len(label_groups) > 1:
            false_positive_ids.extend(ids)
    removed = max(len(assigned) - len(groups), 0)
    return removed, sorted(set(false_positive_ids))


def split_by_duplicate_component(records: list[dict[str, Any]], groups: list[set[int]]) -> dict[str, list[str]]:
    parent = list(range(len(records)))
    for group in groups:
        root = min(group)
        for item in group:
            parent[item] = root
    buckets: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    for i, row in enumerate(records):
        component_ids = sorted(records[j]["id"] for j in range(len(records)) if parent[j] == parent[i])
        key = "|".join(component_ids)
        bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
        split = "train" if bucket < 80 else "dev" if bucket < 90 else "test"
        buckets[split].append(row["id"])
    return buckets


def choose_representatives(records: list[dict[str, Any]], groups: list[set[int]]) -> tuple[list[dict[str, Any]], list[list[str]]]:
    removed: set[int] = set()
    components = []
    component_by_id: dict[str, list[str]] = {}
    for group in groups:
        ordered = sorted(group, key=lambda i: records[i]["id"])
        removed.update(ordered[1:])
        ids = [records[i]["id"] for i in ordered]
        components.append(ids)
        component_by_id.update({doc_id: ids for doc_id in ids})
    selected = []
    for i, row in enumerate(records):
        if i not in removed:
            selected.append(dict(row, duplicate_component=component_by_id.get(row["id"], [row["id"]]), representative=True))
    return selected, components


def threshold_sweep(records: list[dict[str, Any]]) -> dict[str, Any]:
    signatures = [minhash_signature(shingles(row["text"])) for row in records]
    minhash_pairs = [(i, j, sum(a == b for a, b in zip(signatures[i], signatures[j])) / 64)
                     for i in range(len(records)) for j in range(i + 1, len(records))]
    simhashes = [simhash_signature(row["text"]) for row in records]
    simhash_pairs = [(i, j, hamming(simhashes[i], simhashes[j]))
                     for i in range(len(records)) for j in range(i + 1, len(records))]

    def summarize_pairs(pairs: list[tuple[int, int, float]], accept, name: str) -> dict[str, Any]:
        accepted = [(i, j) for i, j, score in pairs if accept(score)]
        groups = connected_components(len(records), accepted)
        removed, false_ids = group_labels(groups, records)
        return {"candidate_pairs": len(accepted), "duplicate_records_removed": removed,
                "false_positive_unique_ids": false_ids,
                "false_positive_pairs": [[records[i]["id"], records[j]["id"]] for i, j in accepted
                                          if records[i].get("duplicate_group") != records[j].get("duplicate_group")
                                          or records[i].get("duplicate_group") is None
                                          or records[j].get("duplicate_group") is None]}

    return {
        "minhash_jaccard_estimate": {str(t): summarize_pairs(minhash_pairs, lambda score, t=t: score >= t, "minhash") for t in [0.70, 0.20, 0.10]},
        "simhash_max_hamming_distance": {str(t): summarize_pairs(simhash_pairs, lambda score, t=t: score <= t, "simhash") for t in [5, 15, 25]},
    }


def run(path: Path = INPUT, min_available_mib: int = 2560) -> dict[str, Any]:
    if available_memory_mib() < min_available_mib:
        raise MemoryError(f"MemAvailable below guard ({min_available_mib} MiB)")
    records = load_records(path)
    tracemalloc.start()
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    normalized_exact = exact_groups(records)
    min_groups, min_pairs = minhash_groups(records)
    sim_groups, sim_pairs = simhash_groups(records)
    boilerplate_removed = [dict(row, text=strip_boilerplate(row["text"])) for row in records]
    quality_kept, quality_removed = quality_filter(boilerplate_removed)
    elapsed = time.perf_counter() - wall_start
    cpu_s = time.process_time() - cpu_start
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    own_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    child_peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024

    methods = {}
    for name, groups, pairs in [("exact_hash", normalized_exact, []), ("minhash_64x3gram", min_groups, min_pairs), ("simhash_64x3gram", sim_groups, sim_pairs)]:
        removed, false_ids = group_labels(groups, records)
        methods[name] = {"duplicate_components": [[records[i]["id"] for i in sorted(g)] for g in groups],
                         "candidate_pairs": len(pairs), "duplicate_records_removed": removed,
                         "false_positive_unique_ids": false_ids}

    trainable = [row for row in quality_kept if row.get("quality_label") != "low_quality"]
    exact_deduped, exact_components = choose_representatives(trainable, exact_groups(trainable))
    minhash_deduped, minhash_components = choose_representatives(trainable, minhash_groups(trainable)[0])
    simhash_deduped, simhash_components = choose_representatives(trainable, simhash_groups(trainable)[0])
    return {
        "schema_version": 1, "input": str(path), "input_sha256": digest(path.read_text(encoding="utf-8")),
        "seed": "fixed_hash_functions_v1", "n_input": len(records), "input_bytes": path.stat().st_size,
        "methods": methods,
        "threshold_sweep": threshold_sweep(records),
        "boilerplate": {"exact_footer_lines_removed": sum(row["text"].count("\n") + (1 if row["text"] else 0) - (strip_boilerplate(row["text"]).count("\n") + (1 if strip_boilerplate(row["text"]) else 0)) for row in records),
                        "records_with_removed_boilerplate": [row["id"] for row in records if strip_boilerplate(row["text"]) != row["text"].strip()]},
        "quality": {"rule": "remove only repeated-token fraction >= 0.70 for >=10 words; code/math receive domain-aware score but same repeated-token hard flag",
                    "kept_ids": [row["id"] for row in quality_kept], "removed_ids": [row["id"] for row in quality_removed],
                    "scores": {row["id"]: row["quality_signals"] for row in quality_kept + quality_removed},
                    "domain_counts_before": dict(Counter(row["domain"] for row in records)),
                    "domain_counts_after": dict(Counter(row["domain"] for row in trainable)),
                    "unique_examples_removed": [row["id"] for row in quality_removed if not row.get("duplicate_group")],
                    "post_filter_records": len(trainable),
                    "post_exact_dedup_records": len(exact_deduped),
                    "post_minhash_dedup_records": len(minhash_deduped),
                    "post_simhash_dedup_records": len(simhash_deduped),
                    "post_exact_dedup_text_bytes": sum(len(row["text"].encode("utf-8")) for row in exact_deduped),
                    "post_minhash_dedup_text_bytes": sum(len(row["text"].encode("utf-8")) for row in minhash_deduped),
                    "post_simhash_dedup_text_bytes": sum(len(row["text"].encode("utf-8")) for row in simhash_deduped),
                    "exact_components": exact_components, "minhash_components": minhash_components,
                    "simhash_components": simhash_components},
        "curated_manifests": {"exact": exact_deduped, "minhash": minhash_deduped, "simhash": simhash_deduped},
        "split_preview_using_minhash_components": split_by_duplicate_component(records, min_groups),
        "performance": {"wall_seconds": elapsed, "cpu_seconds": cpu_s,
                        "records_per_second": len(records) / elapsed if elapsed else None,
                        "peak_rss_bytes_resource_self_plus_children": own_peak + child_peak,
                        "python_tracemalloc_peak_bytes": traced_peak},
        "environment": {"python": sys.version.split()[0], "platform": sys.platform,
                        "mem_available_mib_at_start": available_memory_mib(),
                        "cgroup_memory_limit": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
                        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                        "thread_caps": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]}},
        "limitations": ["tiny synthetic fixture only", "MinHash and SimHash thresholds are exploratory and not tuned", "semantic paraphrase deduplication is not implemented", "quality score is a transparent heuristic, not validated quality prediction", "RSS is process high-water mark; tracemalloc misses native allocations"]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=RESULTS / "curation_tiny.json")
    parser.add_argument("--min-available-mib", type=int, default=2560)
    args = parser.parse_args()
    result = run(args.input, args.min_available_mib)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"n_input": result["n_input"], "methods": result["methods"], "quality": result["quality"], "performance": result["performance"], "result": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
