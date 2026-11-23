"""A byte decoder that reuses one causal block for several refinement steps."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from research.experiments.tiny_gpt import DecoderBlock, TinyGPT


class Decoder(nn.Module):
    name = "decoder"
    choices = (1, 2, 4, 8)

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = dict(config)
        self.vocab_size = int(config["vocab_size"])
        self.context_length = int(config["context_length"])
        width = int(config["d_model"])
        heads = int(config["n_heads"])
        hidden = int(config["d_ff"])
        self.limit = int(config.get("limit", 8))
        self.steps = int(config.get("steps", 1))
        if self.vocab_size < 1 or self.context_length < 1 or width < 1 or hidden < 1:
            raise ValueError("model dimensions must be positive")
        if self.limit not in self.choices:
            raise ValueError("limit must be one of 1, 2, 4, 8")
        self.check(self.steps)
        self.token_embedding = nn.Embedding(self.vocab_size, width)
        self.position_embedding = nn.Embedding(self.context_length, width)
        self.block = DecoderBlock(width, heads, hidden)
        self.final_norm = nn.LayerNorm(width, eps=1e-5)
        self.step = nn.Parameter(torch.zeros(max(0, self.limit - 1), width))
        self.gate = nn.Parameter(torch.zeros(max(0, self.limit - 1), width))
        self.apply(TinyGPT._init_weights)
        scale = 0.02 / (2.0**0.5)
        nn.init.normal_(self.block.attn.proj.weight, mean=0.0, std=scale)
        nn.init.normal_(self.block.mlp.fc_out.weight, mean=0.0, std=scale)
        self.to(device="cpu", dtype=torch.float32)

    @property
    def head(self) -> torch.Tensor:
        return self.token_embedding.weight

    @property
    def params(self) -> int:
        return sum(value.numel() for value in self.parameters())

    def check(self, steps: int) -> None:
        if not isinstance(steps, int) or isinstance(steps, bool):
            raise ValueError("steps must be an integer in the configured choices")
        if steps not in self.choices or steps > self.limit:
            raise ValueError("steps must be one of 1, 2, 4, 8 and no greater than limit")

    def forward(self, token_ids: torch.Tensor, steps: int = 1, trace: bool = False,
                targets: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor | tuple[
                    torch.Tensor, dict[str, torch.Tensor | None]
                ]:
        self.check(steps)
        if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        batch, length = token_ids.shape
        if batch < 1 or not 1 <= length <= self.context_length:
            raise ValueError("batch and sequence dimensions must be nonempty and within context_length")
        if token_ids.dtype != torch.long:
            raise ValueError("token_ids must use torch.long dtype")
        if token_ids.device.type != "cpu":
            raise ValueError("the reference decoder accepts CPU token IDs only")
        if bool(torch.any(token_ids < 0)) or bool(torch.any(token_ids >= self.vocab_size)):
            raise ValueError("token ID is outside the configured vocabulary")
        if not isinstance(trace, bool):
            raise ValueError("trace must be a boolean")
        if targets is not None:
            if (not trace or not isinstance(targets, torch.Tensor)
                    or targets.shape != token_ids.shape or targets.dtype != torch.long):
                raise ValueError("trace targets must be torch.long values with token_ids shape")
            if targets.device.type != "cpu" or bool(torch.any(targets < 0)) \
                    or bool(torch.any(targets >= self.vocab_size)):
                raise ValueError("target ID is outside the configured vocabulary")
            if mask is None:
                mask = torch.ones_like(targets, dtype=torch.bool)
            if not isinstance(mask, torch.Tensor) or mask.shape != targets.shape \
                    or mask.dtype != torch.bool or mask.device.type != "cpu" \
                    or not bool(torch.any(mask)):
                raise ValueError("mask must be a nonempty CPU boolean tensor matching targets")
        elif mask is not None:
            raise ValueError("mask requires targets")
        positions = torch.arange(length, device=token_ids.device)
        source = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        state = self.block(source)
        outputs = [F.linear(self.final_norm(state), self.head)]
        losses = []
        if targets is not None:
            values = F.cross_entropy(
                outputs[-1].reshape(-1, self.vocab_size), targets.reshape(-1),
                reduction="none",
            ).reshape_as(targets)
            losses.append(values.masked_select(mask).mean())
        for index in range(steps - 1):
            state = self.block(state + self.step[index])
            state = state + torch.tanh(self.gate[index]) * source
            outputs.append(F.linear(self.final_norm(state), self.head))
            if targets is not None:
                values = F.cross_entropy(
                    outputs[-1].reshape(-1, self.vocab_size), targets.reshape(-1),
                    reduction="none",
                ).reshape_as(targets)
                losses.append(values.masked_select(mask).mean())
        logits = outputs[-1]
        if not trace:
            return logits
        return logits, {
            "logits": torch.stack(outputs),
            "losses": torch.stack(losses) if targets is not None else None,
        }

    @torch.no_grad()
    def generate(self, token_ids: list[int], max_new_tokens: int,
                 steps: int | None = None) -> list[int]:
        if not 1 <= max_new_tokens <= 16:
            raise ValueError("generation budget must be in 1..16 tokens")
        if not token_ids:
            raise ValueError("prefix must be nonempty")
        count = self.steps if steps is None else steps
        self.check(count)
        if any(not isinstance(value, int) or value < 0 or value >= self.vocab_size for value in token_ids):
            raise ValueError("token ID is outside the configured vocabulary")
        self.eval()
        result = torch.tensor([token_ids], dtype=torch.long)
        for _ in range(max_new_tokens):
            context = result[:, -self.context_length:]
            nextid = self(context, steps=count)[:, -1].argmax(dim=-1, keepdim=True)
            result = torch.cat((result, nextid), dim=1)
        return result[0].tolist()


def transfer(model: Decoder, payload: dict[str, Any]) -> None:
    config = payload.get("config")
    weights = payload.get("model")
    if not isinstance(config, dict) or not isinstance(weights, dict):
        raise ValueError("checkpoint must contain config and model state")
    expected = {"vocab_size": model.vocab_size, "context_length": model.context_length,
                "d_model": model.token_embedding.embedding_dim,
                "n_heads": model.block.attn.n_heads,
                "d_ff": model.block.mlp.fc_in.out_features}
    try:
        mismatch = any(int(config[key]) != value for key, value in expected.items())
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint config is missing compatible dimensions") from exc
    if mismatch:
        raise ValueError("checkpoint dimensions do not match recurrent decoder")
    outcome = model.load_state_dict(weights, strict=False)
    if outcome.unexpected_keys or set(outcome.missing_keys) != {"step", "gate"}:
        raise ValueError("checkpoint is not a compatible one-block TinyGPT state")
    with torch.no_grad():
        model.step.zero_()
        model.gate.zero_()
    model.eval()
