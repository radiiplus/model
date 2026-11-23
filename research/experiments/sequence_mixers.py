"""Standalone recurrent-retention toy mixer for comparison with TinyGPT."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from tiny_gpt import MLP


class NormalizedRetention(nn.Module):
    """Simple recurrent key-value retention; educational, not canonical RetNet."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.register_buffer("decay", torch.tensor([0.50, 0.80, 0.90, 0.99][:n_heads]))

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype):
        state = torch.zeros(batch, self.n_heads, self.head_dim, self.head_dim, device=device, dtype=dtype)
        scale = torch.zeros(batch, self.n_heads, 1, 1, device=device, dtype=dtype)
        return state, scale

    def step(self, x: torch.Tensor, state):
        batch, width = x.shape
        qkv = self.qkv(x).view(batch, 3, self.n_heads, self.head_dim).permute(1, 0, 2, 3)
        q, k, v = qkv.unbind(0)
        matrix, scale = state
        decay = self.decay.to(dtype=x.dtype, device=x.device).view(1, self.n_heads, 1, 1)
        matrix = decay * matrix + k.unsqueeze(-1) * v.unsqueeze(-2)
        scale = decay * scale + 1.0
        output = torch.einsum("bhd,bhdv->bhv", q, matrix) / scale.squeeze(-1)
        output = output.reshape(batch, width) / math.sqrt(self.head_dim)
        return self.proj(output), (matrix, scale)


class RetentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln_mix = nn.LayerNorm(d_model, eps=1e-5)
        self.mixer = NormalizedRetention(d_model, n_heads)
        self.ln_mlp = nn.LayerNorm(d_model, eps=1e-5)
        self.mlp = MLP(d_model, d_ff)


class RetentionGPT(nn.Module):
    """GPT-shaped model with recurrence replacing self-attention only."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = dict(config)
        self.vocab_size = int(config["vocab_size"])
        self.context_length = int(config["context_length"])
        d_model = int(config["d_model"])
        self.token_embedding = nn.Embedding(self.vocab_size, d_model)
        self.position_embedding = nn.Embedding(self.context_length, d_model)
        self.block = RetentionBlock(d_model, int(config["n_heads"]), int(config["d_ff"]))
        self.final_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.apply(self._init_weights)
        nn.init.normal_(self.block.mixer.proj.weight, mean=0.0, std=0.02 / math.sqrt(2.0))
        nn.init.normal_(self.block.mlp.fc_out.weight, mean=0.0, std=0.02 / math.sqrt(2.0))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype):
        return self.block.mixer.initial_state(batch, device, dtype)

    def step(self, token_ids: torch.Tensor, position: int, state):
        x = self.token_embedding(token_ids) + self.position_embedding.weight[position]
        mixed, state = self.block.mixer.step(self.block.ln_mix(x), state)
        x = x + mixed
        x = x + self.block.mlp(self.block.ln_mlp(x))
        logits = F.linear(self.final_norm(x), self.token_embedding.weight)
        return logits, state

    def forward_with_state(self, token_ids: torch.Tensor, state=None):
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        batch, length = token_ids.shape
        if not 1 <= length <= self.context_length:
            raise ValueError("sequence length must be between 1 and context_length")
        if state is None:
            state = self.initial_state(batch, token_ids.device, self.token_embedding.weight.dtype)
        logits = []
        for position in range(length):
            values, state = self.step(token_ids[:, position], position, state)
            logits.append(values)
        return torch.stack(logits, dim=1), state

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.forward_with_state(token_ids)[0]

    def loss(self, token_ids: torch.Tensor) -> torch.Tensor:
        logits = self(token_ids[:, :-1])
        targets = token_ids[:, 1:]
        return F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))


def build_models(config: dict[str, Any]):
    from tiny_gpt import TinyGPT

    return {"transformer": TinyGPT(config), "retention": RetentionGPT(config)}
