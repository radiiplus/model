"""Independent-layer causal language model for the interaction corpus."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from research.experiments.tiny_gpt import DecoderBlock, FoundationKVCache, TinyGPT


class Foundation(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = dict(config)
        self.vocab = int(config["vocab_size"])
        self.context = int(config["context_length"])
        self.layers = int(config["layers"])
        width = int(config["d_model"])
        heads = int(config["n_heads"])
        hidden = int(config["d_ff"])
        if (self.vocab < 1 or self.context < 1 or self.layers < 1 or width < 1 or heads < 1
                or hidden < 1 or width % heads):
            raise ValueError("model dimensions must be positive and width divisible by heads")
        if config.get("position_embedding") != "learned_absolute":
            raise ValueError("only learned absolute positions are supported")
        if config.get("normalization") != "pre_layernorm":
            raise ValueError("only pre-LayerNorm blocks are supported")
        if config.get("weight_tying") is not True or config.get("dtype") != "float32":
            raise ValueError("the reference model requires tied weights and float32")
        self.token = nn.Embedding(self.vocab, width)
        self.position = nn.Embedding(self.context, width)
        self.blocks = nn.ModuleList(
            DecoderBlock(width, heads, hidden) for _ in range(self.layers)
        )
        self.norm = nn.LayerNorm(width, eps=1e-5)
        self.apply(TinyGPT._init_weights)
        scale = 0.02 / math.sqrt(2.0 * self.layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=scale)
            nn.init.normal_(block.mlp.fc_out.weight, mean=0.0, std=scale)
        self.to(device="cpu", dtype=torch.float32)

    @property
    def params(self) -> int:
        return sum(value.numel() for value in self.parameters())

    @property
    def head(self) -> torch.Tensor:
        return self.token.weight

    def hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        batch, length = tokens.shape
        if batch < 1 or not 1 <= length <= self.context:
            raise ValueError("batch and sequence must be nonempty and within context")
        if tokens.dtype != torch.long or tokens.device.type != "cpu":
            raise ValueError("tokens must be CPU torch.long values")
        if bool(torch.any(tokens < 0)) or bool(torch.any(tokens >= self.vocab)):
            raise ValueError("token ID is outside the configured vocabulary")
        positions = torch.arange(length, device=tokens.device)
        state = self.token(tokens) + self.position(positions)[None, :, :]
        for block in self.blocks:
            state = block(state)
        return self.norm(state)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.linear(self.hidden(tokens), self.head)

    def new_kv_cache(self, batch: int = 1,
                     backing_directory: Path | None = None) -> FoundationKVCache:
        return FoundationKVCache(
            batch=batch, layers=self.layers, heads=int(self.config["n_heads"]),
            context=self.context, head_dim=int(self.config["d_model"]) //
            int(self.config["n_heads"]), backing_directory=backing_directory)

    def forward_cached(self, tokens: torch.Tensor,
                       cache: FoundationKVCache) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.dtype != torch.long or tokens.device.type != "cpu":
            raise ValueError("cached tokens must be CPU torch.long with shape [batch, sequence]")
        batch, length = tokens.shape
        if (batch != cache.batch or length < 1 or cache.length + length > self.context
                or cache.layers != self.layers or cache.context != self.context
                or cache.heads != int(self.config["n_heads"])
                or cache.head_dim != int(self.config["d_model"]) //
                int(self.config["n_heads"])):
            raise ValueError("cached input shape exceeds or differs from its cache")
        if bool(torch.any(tokens < 0)) or bool(torch.any(tokens >= self.vocab)):
            raise ValueError("token ID is outside the configured vocabulary")
        start = cache.length
        positions = torch.arange(start, start + length, device=tokens.device)
        state = self.token(tokens) + self.position(positions)[None, :, :]
        for index, block in enumerate(self.blocks):
            state = block.forward_cached(
                state, cache.keys[index], cache.values[index], start)
        cache.length += length
        return F.linear(self.norm(state), self.head)

    def losses(self, tokens: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if tokens.ndim != 2 or tokens.shape[1] < 2:
            raise ValueError("next-token losses need at least two tokens")
        if mask.shape != tokens.shape or mask.dtype != torch.bool or mask.device.type != "cpu":
            raise ValueError("target mask must be a matching CPU boolean tensor")
        logits = self(tokens[:, :-1])
        target = tokens[:, 1:]
        values = F.cross_entropy(logits.reshape(-1, self.vocab), target.reshape(-1),
                                 reduction="none").reshape_as(target)
        selected = mask[:, 1:]
        answer = values.masked_select(selected).mean() if bool(torch.any(selected)) else values.sum() * 0.0
        return {"language": values.mean(), "interaction": answer,
                "tokens": torch.tensor(target.numel()),
                "answers": selected.sum()}
