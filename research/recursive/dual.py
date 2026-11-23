"""Dual-state causal byte refiner with a shared transformation block."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from research.experiments.tiny_gpt import DecoderBlock, TinyGPT
from research.recursive.model import Decoder


class Refiner(nn.Module):
    """Refine an answer state y while maintaining separate latent state z.

    Both states have shape [batch, sequence, width]. The source x is the
    original token-plus-position embedding and is reinjected into each update.
    """

    name = "refiner"
    choices = (1, 2, 4, 8)

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = dict(config)
        self.vocab = int(config["vocab_size"])
        self.context = int(config["context_length"])
        self.width = int(config["d_model"])
        self.limit = int(config.get("limit", 8))
        self.steps = int(config.get("steps", 1))
        self.budget = int(config.get("budget", 1_000_000))
        if self.vocab < 1 or self.context < 1 or self.width < 1:
            raise ValueError("model dimensions must be positive")
        if self.limit not in self.choices:
            raise ValueError("limit must be one of 1, 2, 4, 8")
        self.check(self.steps)
        self.token = nn.Embedding(self.vocab, self.width)
        self.position = nn.Embedding(self.context, self.width)
        self.block = DecoderBlock(self.width, int(config["n_heads"]), int(config["d_ff"]))
        self.norm = nn.LayerNorm(self.width, eps=1e-5)
        self.step = nn.Parameter(torch.zeros(self.limit, self.width))
        self.gate = nn.Parameter(torch.zeros(self.limit, 4, self.width))
        self.apply(TinyGPT._init_weights)
        scale = 0.02 / (2.0**0.5)
        nn.init.normal_(self.block.attn.proj.weight, mean=0.0, std=scale)
        nn.init.normal_(self.block.mlp.fc_out.weight, mean=0.0, std=scale)
        self.to(device="cpu", dtype=torch.float32)
        if self.params > self.budget:
            raise ValueError("model exceeds configured parameter budget")

    @property
    def head(self) -> torch.Tensor:
        return self.token.weight

    @property
    def params(self) -> int:
        return sum(value.numel() for value in self.parameters())

    def check(self, steps: int) -> None:
        if not isinstance(steps, int) or isinstance(steps, bool):
            raise ValueError("steps must be an integer in the configured choices")
        if steps not in self.choices or steps > self.limit:
            raise ValueError("steps must be one of 1, 2, 4, 8 and no greater than limit")

    def forward(
        self,
        ids: torch.Tensor,
        steps: int | None = None,
        trace: bool = False,
        targets: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        count = self.steps if steps is None else steps
        self.check(count)
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2:
            raise ValueError("ids must have shape [batch, sequence]")
        batch, length = ids.shape
        if batch < 1 or not 1 <= length <= self.context:
            raise ValueError("batch and sequence dimensions must be nonempty and within context")
        if ids.dtype != torch.long or ids.device.type != "cpu":
            raise ValueError("ids must be CPU torch.long values")
        if bool(torch.any(ids < 0)) or bool(torch.any(ids >= self.vocab)):
            raise ValueError("token ID is outside the configured vocabulary")
        if not isinstance(trace, bool):
            raise ValueError("trace must be a boolean")
        if targets is not None:
            if not trace:
                raise ValueError("targets require trace=True")
            if (not isinstance(targets, torch.Tensor) or targets.shape != ids.shape
                    or targets.dtype != torch.long or targets.device.type != "cpu"):
                raise ValueError("targets must be CPU torch.long values with ids shape")
            if bool(torch.any(targets < 0)) or bool(torch.any(targets >= self.vocab)):
                raise ValueError("target ID is outside the configured vocabulary")
            if mask is None:
                mask = torch.ones_like(targets, dtype=torch.bool)
            if not isinstance(mask, torch.Tensor) or mask.shape != targets.shape:
                raise ValueError("mask must match targets")
            if mask.device.type != "cpu" or mask.dtype != torch.bool:
                raise ValueError("mask must be a CPU boolean tensor")
            if not bool(torch.any(mask)):
                raise ValueError("mask must select at least one target")
        elif mask is not None:
            raise ValueError("mask requires targets")

        positions = torch.arange(length, device=ids.device)
        source = self.token(ids) + self.position(positions)[None, :, :]
        latent = torch.zeros_like(source)
        answer = source
        logs = []
        deltas = []
        losses = []
        for index in range(count):
            oldlatent = latent
            oldanswer = answer
            gate = torch.tanh(self.gate[index])
            step = self.step[index]
            latent = self.block(
                latent + gate[0] * source + gate[1] * answer + step
            )
            answer = self.block(
                answer + gate[2] * source + gate[3] * latent + step
            )
            logits = F.linear(self.norm(answer), self.head)
            if trace:
                logs.append(logits)
                deltas.append(torch.stack((
                    (latent - oldlatent).detach().abs().mean(),
                    (answer - oldanswer).detach().abs().mean(),
                )))
                if targets is not None:
                    values = F.cross_entropy(
                        logits.reshape(-1, self.vocab), targets.reshape(-1), reduction="none"
                    ).reshape_as(targets)
                    losses.append(values.masked_select(mask).mean())
        if not trace:
            return logits
        return logits, {
            "logits": torch.stack(logs),
            "losses": torch.stack(losses) if targets is not None else None,
            "deltas": torch.stack(deltas),
        }

    @torch.no_grad()
    def generate(self, ids: list[int], max_new_tokens: int,
                 steps: int | None = None) -> list[int]:
        if not 1 <= max_new_tokens <= 16:
            raise ValueError("generation budget must be in 1..16 tokens")
        if not ids:
            raise ValueError("prefix must be nonempty")
        count = self.steps if steps is None else steps
        self.check(count)
        if any(not isinstance(value, int) or value < 0 or value >= self.vocab
               for value in ids):
            raise ValueError("token ID is outside the configured vocabulary")
        self.eval()
        result = torch.tensor([ids], dtype=torch.long)
        for _ in range(max_new_tokens):
            context = result[:, -self.context:]
            nextid = self(context, steps=count)[:, -1].argmax(dim=-1, keepdim=True)
            result = torch.cat((result, nextid), dim=1)
        return result[0].tolist()


def transfer(model: Refiner, source: Decoder) -> None:
    if not isinstance(source, Decoder):
        raise ValueError("source must be a compatible shared-weight Decoder")
    pairs = ((model.token.weight, source.token_embedding.weight),
             (model.position.weight, source.position_embedding.weight),
             (model.norm.weight, source.final_norm.weight),
             (model.norm.bias, source.final_norm.bias))
    if any(left.shape != right.shape for left, right in pairs):
        raise ValueError("source dimensions do not match dual-state refiner")
    if model.block.state_dict().keys() != source.block.state_dict().keys():
        raise ValueError("source block does not match dual-state refiner")
    with torch.no_grad():
        for left, right in pairs:
            left.copy_(right)
        model.block.load_state_dict(source.block.state_dict())
        model.step.zero_()
        model.gate.zero_()
    model.eval()
