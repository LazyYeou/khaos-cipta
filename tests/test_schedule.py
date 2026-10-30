import pytest
import torch

from src.diffusion import GaussianDiffusion, extract


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_alpha_bar_properties(schedule: str) -> None:
    diffusion = GaussianDiffusion(timesteps=1000, schedule=schedule)
    alpha_bar = diffusion.alpha_bar

    assert alpha_bar.shape == (1000,)
    assert alpha_bar.dtype == torch.float32
    assert torch.all(alpha_bar[1:] < alpha_bar[:-1]), "alpha_bar harus menurun monoton"
    assert alpha_bar[0] == pytest.approx(1.0, abs=1e-3)
    assert alpha_bar[-1] == pytest.approx(0.0, abs=1e-3)
    assert torch.all((alpha_bar > 0) & (alpha_bar < 1))


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_posterior_variance_is_valid(schedule: str) -> None:
    diffusion = GaussianDiffusion(timesteps=1000, schedule=schedule)
    # beta_tilde_0 = 0 karena alpha_bar_{-1} = 1, sisanya positif dan ≤ beta_t.
    assert diffusion.posterior_variance[0] == 0
    assert torch.all(diffusion.posterior_variance[1:] > 0)
    assert torch.all(diffusion.posterior_variance <= diffusion.betas + 1e-8)


def test_unknown_schedule_raises() -> None:
    with pytest.raises(ValueError):
        GaussianDiffusion(timesteps=10, schedule="quadratic")


def test_extract_shape_and_values() -> None:
    buf = torch.arange(10, dtype=torch.float32)
    t = torch.tensor([0, 3, 9])
    out = extract(buf, t, torch.Size([3, 3, 8, 8]))

    assert out.shape == (3, 1, 1, 1)
    assert out.flatten().tolist() == [0.0, 3.0, 9.0]
