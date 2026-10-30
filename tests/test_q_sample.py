import pytest
import torch

from src.diffusion import GaussianDiffusion


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_large_t_is_standard_normal(schedule: str) -> None:
    torch.manual_seed(0)
    diffusion = GaussianDiffusion(timesteps=1000, schedule=schedule)
    # x0 jauh dari N(0, 1) agar test benar-benar menguji forward process.
    x0 = torch.full((256, 3, 16, 16), 0.8)
    t = torch.full((256,), 999, dtype=torch.long)
    xt = diffusion.q_sample(x0, t, torch.randn_like(x0))

    assert xt.mean().item() == pytest.approx(0.0, abs=0.02)
    assert xt.std().item() == pytest.approx(1.0, abs=0.02)


def test_matches_closed_form_statistics() -> None:
    torch.manual_seed(0)
    diffusion = GaussianDiffusion(timesteps=1000, schedule="linear")
    x0 = torch.full((512, 1, 16, 16), 0.5)
    t = torch.full((512,), 300, dtype=torch.long)
    xt = diffusion.q_sample(x0, t, torch.randn_like(x0))

    # q(x_t | x_0) = N(sqrt(alpha_bar_t) · x0, (1 − alpha_bar_t) I)  (Ho et al. Eq. 4)
    alpha_bar = diffusion.alpha_bar[300].item()
    assert xt.mean().item() == pytest.approx(alpha_bar**0.5 * 0.5, abs=0.01)
    assert xt.std().item() == pytest.approx((1 - alpha_bar) ** 0.5, abs=0.01)


def test_small_t_stays_close_to_x0() -> None:
    torch.manual_seed(0)
    diffusion = GaussianDiffusion(timesteps=1000, schedule="cosine")
    x0 = torch.rand(8, 3, 16, 16) * 2 - 1
    t = torch.zeros(8, dtype=torch.long)
    xt = diffusion.q_sample(x0, t, torch.randn_like(x0))

    assert (xt - x0).abs().mean().item() < 0.02


def test_predict_x0_inverts_q_sample() -> None:
    torch.manual_seed(0)
    diffusion = GaussianDiffusion(timesteps=1000, schedule="linear")
    x0 = torch.rand(4, 3, 8, 8) * 2 - 1
    t = torch.tensor([0, 10, 500, 999])
    noise = torch.randn_like(x0)
    xt = diffusion.q_sample(x0, t, noise)

    assert torch.allclose(diffusion.predict_x0_from_eps(xt, t, noise), x0, atol=1e-3)
