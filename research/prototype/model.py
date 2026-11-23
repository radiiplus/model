"""Adapter over the preserved Batch 4 TinyGPT implementation."""

from __future__ import annotations

from typing import Any

import torch

from research.experiments.tiny_gpt import TinyGPT


class ByteLanguageModel:
    name = "tiny-byte-gpt"

    def __init__(self, config: dict[str, Any], seed: int):
        torch.manual_seed(seed)
        self.module = TinyGPT(config).cpu()
        self.config = dict(config)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.module.parameters())

    def generate(self, token_ids: list[int], max_new_tokens: int) -> list[int]:
        if not 1 <= max_new_tokens <= 16:
            raise ValueError("generation budget must be in 1..16 tokens")
        self.module.eval()
        prefix = torch.tensor([token_ids], dtype=torch.long)
        with torch.inference_mode():
            return self.module.generate(prefix, max_new_tokens)[0].tolist()
