"""Deterministic byte windows and source/document split leakage checks."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from research.prototype.interfaces import ByteTokenizer


def load_documents(path: Path) -> dict[str, str]:
    docs: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["id"] in docs:
            raise ValueError(f"duplicate document ID: {row['id']}")
        docs[row["id"]] = row["text"]
    return docs


def load_splits(text_path: Path, split_path: Path, tokenizer: ByteTokenizer) -> tuple[dict[str, list[list[int]]], dict[str, Any]]:
    docs = load_documents(text_path)
    spec = json.loads(split_path.read_text(encoding="utf-8"))
    split_ids = {name: spec[f"{name}_ids"] for name in ("train", "validation", "test")}
    flattened = [doc_id for ids in split_ids.values() for doc_id in ids]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(docs):
        raise ValueError("split IDs must partition the corpus exactly once")
    digest_owners: dict[str, str] = {}
    normalized_owners: dict[str, str] = {}
    encoded: dict[str, list[list[int]]] = {}
    for split, ids in split_ids.items():
        encoded[split] = []
        for doc_id in ids:
            text = docs[doc_id]
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            normalized = " ".join(text.casefold().split())
            normalized_digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            for key, owners in ((digest, digest_owners), (normalized_digest, normalized_owners)):
                owner = owners.get(key)
                if owner is not None and owner != split:
                    raise ValueError(f"cross-split duplicate content: {doc_id} and {owner}")
                owners[key] = split
            token_ids = tokenizer.encode(text)
            if len(token_ids) < 2:
                raise ValueError(f"document too short for next-byte training: {doc_id}")
            encoded[split].append(token_ids)
    audit = {
        "source": str(text_path),
        "source_sha256": hashlib.sha256(text_path.read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "license": spec["license"],
        "document_counts": {name: len(ids) for name, ids in split_ids.items()},
        "byte_counts": {name: sum(map(len, rows)) for name, rows in encoded.items()},
        "cross_split_exact_or_normalized_duplicate_count": 0,
        "split_ids": split_ids,
    }
    return encoded, audit


class ByteWindowLoader:
    def __init__(self, documents: list[list[int]], context_length: int):
        self.context_length = context_length
        self.windows: list[torch.Tensor] = []
        for row in documents:
            for start in range(max(1, len(row) - context_length)):
                window = row[start : start + context_length + 1]
                if len(window) == context_length + 1:
                    self.windows.append(torch.tensor(window, dtype=torch.long))
        if not self.windows:
            raise ValueError("no complete context windows in training split")

    def sample(self, batch_size: int, generator: torch.Generator) -> torch.Tensor:
        indices = torch.randint(len(self.windows), (batch_size,), generator=generator)
        return torch.stack([self.windows[int(i)] for i in indices])


def evaluate_bytes(model: torch.nn.Module, documents: list[list[int]], context_length: int) -> dict[str, float | int]:
    model.eval()
    total_nll = 0.0
    correct = total = 0
    with torch.inference_mode():
        for doc in documents:
            for start in range(0, len(doc) - 1, context_length):
                window = torch.tensor(doc[start : start + context_length + 1], dtype=torch.long)[None, :]
                if window.shape[1] < 2:
                    continue
                logits = model(window[:, :-1])
                targets = window[:, 1:]
                total_nll += float(torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum"))
                correct += int((logits.argmax(-1) == targets).sum())
                total += targets.numel()
    if total == 0:
        raise ValueError("no evaluation target bytes")
    mean_nll = total_nll / total
    return {"target_bytes": total, "nll_nats_per_byte": mean_nll,
            "bits_per_byte": mean_nll / 0.6931471805599453, "next_byte_accuracy": correct / total}
