import pytest
import torch
from torch import nn

from src.ema import EMA


def test_ema_moves_towards_model_weights() -> None:
    model = nn.Linear(2, 2)
    ema = EMA(model, decay=0.9)
    start = ema.model.weight.clone()
    with torch.no_grad():
        model.weight.add_(1.0)

    ema.update(model)
    # Update pertama: decay efektif = min(0.9, 2/11) → EMA bergerak ~82% ke bobot model.
    expected = start + (1 - 2 / 11) * 1.0
    assert torch.allclose(ema.model.weight, expected)


def test_ema_reaches_full_decay_after_warmup() -> None:
    model = nn.Linear(2, 2)
    ema = EMA(model, decay=0.5)
    for _ in range(20):
        ema.update(model)
    start = ema.model.weight.clone()
    with torch.no_grad():
        model.weight.add_(1.0)

    ema.update(model)
    assert torch.allclose(ema.model.weight, start + 0.5)


def test_ema_state_dict_roundtrip() -> None:
    model = nn.Linear(2, 2)
    ema = EMA(model, decay=0.99)
    ema.update(model)
    other = EMA(nn.Linear(2, 2), decay=0.99)
    other.load_state_dict(ema.state_dict())

    assert other.num_updates == 1
    assert torch.equal(other.model.weight, ema.model.weight)
    assert not any(p.requires_grad for p in other.model.parameters())


def test_ema_model_does_not_require_grad() -> None:
    ema = EMA(nn.Linear(2, 2), decay=0.9)
    assert ema.model.training is False
    assert all(not p.requires_grad for p in ema.model.parameters())
    assert ema.decay == pytest.approx(0.9)
