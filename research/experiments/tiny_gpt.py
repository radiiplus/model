"""Readable one-layer causal Transformer baseline implemented with PyTorch."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        future = torch.triu(torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(future, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        attended = (weights @ v).transpose(1, 2).contiguous().view(batch, length, width)
        return self.proj(attended)

    def forward_cached(self, x: torch.Tensor, key_cache: torch.Tensor,
                       value_cache: torch.Tensor, past_length: int) -> torch.Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.n_heads, self.head_dim)
        q, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        end = past_length + length
        key_cache[:, :, past_length:end].copy_(key)
        value_cache[:, :, past_length:end].copy_(value)
        keys = key_cache[:, :, :end]
        values = value_cache[:, :, :end]
        scores = (q @ keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if length > 1:
            query_positions = past_length + torch.arange(length, device=x.device)
            key_positions = torch.arange(end, device=x.device)
            future = key_positions[None, :] > query_positions[:, None]
            scores = scores.masked_fill(
                future[None, None, :, :], torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        attended = (weights @ values).transpose(1, 2).contiguous().view(batch, length, width)
        return self.proj(attended)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc_in = nn.Linear(d_model, d_ff, bias=True)
        self.fc_out = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(F.gelu(self.fc_in(x), approximate="none"))


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln_mlp = nn.LayerNorm(d_model, eps=1e-5)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_attn(x))
        return x + self.mlp(self.ln_mlp(x))

    def forward_cached(self, x: torch.Tensor, key_cache: torch.Tensor,
                       value_cache: torch.Tensor, past_length: int) -> torch.Tensor:
        x = x + self.attn.forward_cached(
            self.ln_attn(x), key_cache, value_cache, past_length)
        return x + self.mlp(self.ln_mlp(x))


class FoundationKVCache:
    """Preallocated CPU keys/values; optional mmap backing lets the OS page cold cache data."""

    def __init__(self, *, batch: int, layers: int, heads: int, context: int,
                 head_dim: int, backing_directory: Path | None = None):
        if min(batch, layers, heads, context, head_dim) < 1:
            raise ValueError("KV cache dimensions must be positive")
        self.batch = batch
        self.layers = layers
        self.heads = heads
        self.context = context
        self.head_dim = head_dim
        self.length = 0
        self.backing_directory = backing_directory
        if backing_directory is not None:
            backing_directory.mkdir(parents=True, exist_ok=True)
        shape = (batch, heads, context, head_dim)
        size = math.prod(shape)
        self.keys = []
        self.values = []
        self._files = []
        for layer in range(layers):
            self.keys.append(self._allocate(shape, size, f"layer-{layer:02d}-key"))
            self.values.append(self._allocate(shape, size, f"layer-{layer:02d}-value"))

    def _allocate(self, shape: tuple[int, ...], size: int, name: str) -> torch.Tensor:
        if self.backing_directory is None:
            return torch.empty(shape, dtype=torch.float32, device="cpu")
        path = self.backing_directory / f"{name}.bin"
        with path.open("xb") as stream:
            stream.truncate(size * torch.tensor([], dtype=torch.float32).element_size())
        self._files.append(path)
        return torch.from_file(str(path), shared=True, size=size, dtype=torch.float32).view(shape)

    @property
    def bytes(self) -> int:
        return sum(item.numel() * item.element_size() for item in self.keys + self.values)

    @property
    def backing(self) -> str:
        return "mmap" if self.backing_directory is not None else "ram"

    def reset(self) -> None:
        self.length = 0


class TinyGPT(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = dict(config)
        self.vocab_size = int(config["vocab_size"])
        self.context_length = int(config["context_length"])
        d_model = int(config["d_model"])
        self.token_embedding = nn.Embedding(self.vocab_size, d_model)
        self.position_embedding = nn.Embedding(self.context_length, d_model)
        self.block = DecoderBlock(d_model, int(config["n_heads"]), int(config["d_ff"]))
        self.final_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.apply(self._init_weights)
        residual_scale = 0.02 / math.sqrt(2.0)
        nn.init.normal_(self.block.attn.proj.weight, mean=0.0, std=residual_scale)
        nn.init.normal_(self.block.mlp.fc_out.weight, mean=0.0, std=residual_scale)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        batch, length = token_ids.shape
        if not 1 <= length <= self.context_length:
            raise ValueError("sequence length must be between 1 and context_length")
        positions = torch.arange(length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        x = self.final_norm(self.block(x))
        return F.linear(x, self.token_embedding.weight)

    def loss(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.shape[1] < 2:
            raise ValueError("next-token loss needs sequence length >= 2")
        logits = self(token_ids[:, :-1])
        targets = token_ids[:, 1:]
        return F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] == 0:
            raise ValueError("prefix must have shape [batch, nonempty_sequence]")
        result = token_ids
        for _ in range(max_new_tokens):
            context = result[:, -self.context_length:]
            next_id = self(context)[:, -1].argmax(dim=-1, keepdim=True)
            result = torch.cat((result, next_id), dim=1)
        return result


def checkpoint_payload(model: TinyGPT, optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
    result = {"config": model.config, "model": model.state_dict()}
    if optimizer is not None:
        result["optimizer"] = optimizer.state_dict()
    return result


def load_checkpoint(path: Path, *, map_location: str = "cpu") -> tuple[TinyGPT, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    model = TinyGPT(payload["config"])
    model.load_state_dict(payload["model"])
    return model, payload
