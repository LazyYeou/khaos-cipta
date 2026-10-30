"""Exponential Moving Average bobot model untuk sampling."""

import copy
from typing import Any

import torch
from torch import nn


class EMA:
    """Menyimpan salinan model dengan bobot theta_ema ← d · theta_ema + (1 − d) · theta.

    Decay efektif = min(decay, (1 + n) / (10 + n)) dengan n = jumlah update, sehingga di awal
    training EMA mengikuti bobot asli dengan cepat (warm-up) sebelum mencapai `decay` penuh.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.num_updates = 0
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for ema_p, p in zip(self.model.parameters(), model.parameters(), strict=True):
            ema_p.lerp_(p, 1 - decay)
        for ema_b, b in zip(self.model.buffers(), model.buffers(), strict=True):
            ema_b.copy_(b)

    def state_dict(self) -> dict[str, Any]:
        return {"model": self.model.state_dict(), "num_updates": self.num_updates}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.num_updates = state["num_updates"]
