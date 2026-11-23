"""Non-shared causal stack used as a compute-matched control."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from research.experiments.tiny_gpt import DecoderBlock, TinyGPT


class Stack(nn.Module):
    name = "stack"

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = dict(config)
        self.vocab_size = int(config["vocab_size"])
        self.context_length = int(config["context_length"])
        width = int(config["d_model"])
        depth = int(config["depth"])
        if depth < 1:
            raise ValueError("depth must be positive")
        self.token = nn.Embedding(self.vocab_size, width)
        self.position = nn.Embedding(self.context_length, width)
        self.layers = nn.ModuleList([
            DecoderBlock(width, int(config["n_heads"]), int(config["d_ff"]))
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(width, eps=1e-5)
        self.apply(TinyGPT._init_weights)
        scale = 0.02 / (2.0**0.5)
        for layer in self.layers:
            nn.init.normal_(layer.attn.proj.weight, mean=0.0, std=scale)
            nn.init.normal_(layer.mlp.fc_out.weight, mean=0.0, std=scale)
        self.to(device="cpu", dtype=torch.float32)

    @property
    def params(self) -> int:
        return sum(value.numel() for value in self.parameters())

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2:
            raise ValueError("ids must have shape [batch, sequence]")
        batch, length = ids.shape
        if batch < 1 or not 1 <= length <= self.context_length:
            raise ValueError("batch and sequence dimensions must be within context")
        if ids.dtype != torch.long or ids.device.type != "cpu":
            raise ValueError("ids must be CPU torch.long values")
        if bool(torch.any(ids < 0)) or bool(torch.any(ids >= self.vocab_size)):
            raise ValueError("token ID is outside the configured vocabulary")
        positions = torch.arange(length, device=ids.device)
        state = self.token(ids) + self.position(positions)[None, :, :]
        for layer in self.layers:
            state = layer(state)
        return F.linear(self.norm(state), self.token.weight)
