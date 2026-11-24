"""Four-block shared-stack dual-state language refiner."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from research.foundation.model import Foundation


class Refiner(Foundation):
    def __init__(self, config: dict[str, Any], limit: int = 4):
        super().__init__(config)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("refinement limit must be positive")
        self.limit = limit
        self.gate = nn.Parameter(torch.zeros(limit, 4, self.token.embedding_dim))

    @property
    def params(self) -> int:
        return sum(value.numel() for value in self.parameters())

    def check(self, steps: int) -> None:
        if not isinstance(steps, int) or isinstance(steps, bool):
            raise ValueError("refinement steps must be an integer")
        if not 1 <= steps <= self.limit:
            raise ValueError("refinement steps exceed the configured limit")

    def forward(self, tokens: torch.Tensor, steps: int = 1,
                trace: bool = False) -> torch.Tensor | dict[str, Any]:
        self.check(steps)
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
        source = self.token(tokens) + self.position(positions)[None, :, :]
        latent = torch.zeros_like(source)
        answer = source.clone()
        logits = []
        states = []
        for index in range(steps):
            gate = self.gate[index]
            current = (source + torch.tanh(gate[0]) * (answer - source)
                       + torch.tanh(gate[1]) * latent)
            latent = current
            for block in self.blocks:
                latent = block(latent)
            answer = (latent + torch.tanh(gate[2]) * (answer - latent)
                      + torch.tanh(gate[3]) * source)
            logits.append(F.linear(self.norm(answer), self.head))
            if trace:
                states.append((latent, answer))
        if not trace:
            return logits[-1]
        return {"logits": logits, "states": states}

    def losses(self, tokens: torch.Tensor, mask: torch.Tensor, steps: int,
               weights: list[float]) -> list[dict[str, torch.Tensor]]:
        self.check(steps)
        if len(weights) != steps or any(value < 0 for value in weights):
            raise ValueError("one nonnegative supervision weight is required per step")
        if abs(sum(weights) - 1.0) > 1e-8:
            raise ValueError("deep-supervision weights must sum to one")
        if tokens.ndim != 2 or tokens.shape[1] < 2:
            raise ValueError("next-token losses need at least two tokens")
        if mask.shape != tokens.shape or mask.dtype != torch.bool or mask.device.type != "cpu":
            raise ValueError("target mask must be a matching CPU boolean tensor")
        result = self(tokens[:, :-1], steps=steps, trace=True)
        target = tokens[:, 1:]
        selected = mask[:, 1:]
        rows = []
        for logits in result["logits"]:
            values = F.cross_entropy(logits.reshape(-1, self.vocab), target.reshape(-1),
                                     reduction="none").reshape_as(target)
            answer = (values.masked_select(selected).mean() if bool(torch.any(selected))
                      else values.sum() * 0.0)
            rows.append({"language": values.mean(), "interaction": answer,
                         "tokens": torch.tensor(target.numel()),
                         "answers": selected.sum()})
        return rows


def transfer(model: Refiner, source: Foundation) -> None:
    if model.config != source.config:
        raise ValueError("foundation and refiner configurations differ")
    outcome = model.load_state_dict(source.state_dict(), strict=False)
    if outcome.unexpected_keys or set(outcome.missing_keys) != {"gate"}:
        raise ValueError("source state is not a compatible plain foundation")
    with torch.no_grad():
        model.gate.zero_()
