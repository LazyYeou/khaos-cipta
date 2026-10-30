import pytest
import torch

from src.diffusion import GaussianDiffusion, extract
from src.samplers import ddim_timesteps, ddpm_steps, sample
from src.unet import UNet


@pytest.fixture
def model() -> UNet:
    torch.manual_seed(0)
    model = UNet(
        image_size=16,
        in_channels=3,
        base_channels=32,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attn_resolutions=(8,),
        dropout=0.0,
        num_groups=8,
    )
    # Bobot acak (bukan zero-init) agar prediksi noise tidak nol.
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return model.eval()


@pytest.mark.parametrize(("sampler", "steps"), [("ddpm", 20), ("ddim", 5), ("ddim", 20)])
def test_sampler_shape_and_range(model: UNet, sampler: str, steps: int) -> None:
    diffusion = GaussianDiffusion(timesteps=20, schedule="cosine")
    x_T = torch.randn(2, 3, 16, 16)
    out = sample(model, diffusion, x_T, sampler, steps)

    assert out.shape == x_T.shape
    assert torch.isfinite(out).all()
    assert out.min() >= -1 and out.max() <= 1


def test_ddim_eta0_is_deterministic(model: UNet) -> None:
    diffusion = GaussianDiffusion(timesteps=20, schedule="linear")
    x_T = torch.randn(2, 3, 16, 16)
    a = sample(model, diffusion, x_T, "ddim", 10, eta=0.0)
    b = sample(model, diffusion, x_T, "ddim", 10, eta=0.0)
    assert torch.equal(a, b)


def test_ddim_eta1_is_finite(model: UNet) -> None:
    diffusion = GaussianDiffusion(timesteps=20, schedule="linear")
    out = sample(model, diffusion, torch.randn(2, 3, 16, 16), "ddim", 20, eta=1.0)
    assert torch.isfinite(out).all()


def test_ddpm_step_matches_eq11_without_clipping(model: UNet) -> None:
    """Mean via prediksi x0 + posterior (Eq. 7) harus sama dengan Eq. 11 jika tanpa clipping."""
    diffusion = GaussianDiffusion(timesteps=20, schedule="linear")
    x_T = torch.randn(2, 3, 16, 16)
    t = torch.full((2,), 19, dtype=torch.long)
    gen = torch.Generator().manual_seed(1)
    x_prev = next(ddpm_steps(model, diffusion, x_T, gen, clip_denoised=False))

    with torch.no_grad():
        eps = model(x_T, t)
    beta = extract(diffusion.betas, t, x_T.shape)
    alpha = extract(diffusion.alphas, t, x_T.shape)
    one_minus_ab = 1 - extract(diffusion.alpha_bar, t, x_T.shape)
    mean = (x_T - beta / one_minus_ab.sqrt() * eps) / alpha.sqrt()
    noise = torch.randn(x_T.shape, generator=torch.Generator().manual_seed(1))
    expected = mean + extract(diffusion.posterior_variance, t, x_T.shape).sqrt() * noise

    assert torch.allclose(x_prev, expected, atol=1e-4)


def test_ddpm_yields_one_step_per_timestep(model: UNet) -> None:
    diffusion = GaussianDiffusion(timesteps=7, schedule="linear")
    frames = list(ddpm_steps(model, diffusion, torch.randn(1, 3, 16, 16)))
    assert len(frames) == 7


def test_ddim_timesteps_cover_range() -> None:
    seq = ddim_timesteps(1000, 50)
    assert len(seq) == 50
    assert seq[0] == 0 and seq[-1] == 999
    assert all(b > a for a, b in zip(seq, seq[1:], strict=False))
    assert ddim_timesteps(50, 50) == list(range(50))
    with pytest.raises(ValueError):
        ddim_timesteps(50, 51)


def test_unknown_sampler_raises(model: UNet) -> None:
    diffusion = GaussianDiffusion(timesteps=5, schedule="linear")
    with pytest.raises(ValueError):
        sample(model, diffusion, torch.randn(1, 3, 16, 16), "euler", 5)
