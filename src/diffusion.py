"""Noise schedule, forward process, dan loss DDPM.

Referensi:
- Ho, Jain & Abbeel (2020), "Denoising Diffusion Probabilistic Models" (DDPM).
- Nichol & Dhariwal (2021), "Improved Denoising Diffusion Probabilistic Models" (cosine schedule).
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> Tensor:
    """Schedule linear dari Ho et al. 2020 (Sec. 4): beta_1 = 1e-4 … beta_T = 0.02.
    → betas: (T,) float64
    """
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Tensor:
    """Schedule cosine dari Nichol & Dhariwal 2021 (Eq. 17).
    alpha_bar(t) = f(t) / f(0),  f(t) = cos(((t/T + s) / (1 + s)) · π/2)²
    beta_t = 1 − alpha_bar(t) / alpha_bar(t−1), di-clip ke 0.999 agar tidak singular di t ≈ T.
    → betas: (T,) float64
    """
    steps = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
    f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(max=0.999)


def extract(buf: Tensor, t: Tensor, shape: torch.Size) -> Tensor:
    """Ambil buf[t] per sampel dan reshape agar bisa di-broadcast ke `shape`.
    buf: (T,), t: (B,) long, shape: (B, C, H, W) → (B, 1, 1, 1)
    """
    out = buf.gather(0, t)
    return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))


class GaussianDiffusion(nn.Module):
    """Menyimpan semua buffer schedule dan menyediakan forward process + loss.

    Semua buffer dihitung sekali dalam float64 lalu disimpan sebagai float32. Indeks t 0-based:
    t = 0 … T−1, sehingga buffer[t] di sini = besaran di paper pada langkah t+1.
    """

    def __init__(
        self,
        timesteps: int,
        schedule: str,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        cosine_s: float = 0.008,
    ) -> None:
        super().__init__()
        if schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(timesteps, cosine_s)
        else:
            raise ValueError(f"schedule tidak dikenal: {schedule!r} (pilih 'linear' atau 'cosine')")

        self.timesteps = timesteps
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bar[:-1]])

        # Variansi posterior q(x_{t-1} | x_t, x_0), Ho et al. Eq. 7:
        # beta_tilde_t = (1 − alpha_bar_{t−1}) / (1 − alpha_bar_t) · beta_t
        posterior_variance = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        # Koefisien mean posterior (Eq. 7):
        # mu_tilde = coef1 · x_0 + coef2 · x_t
        posterior_mean_coef1 = betas * alpha_bar_prev.sqrt() / (1.0 - alpha_bar)
        posterior_mean_coef2 = (1.0 - alpha_bar_prev) * alphas.sqrt() / (1.0 - alpha_bar)

        buffers = {
            "betas": betas,
            "alphas": alphas,
            "alpha_bar": alpha_bar,
            "alpha_bar_prev": alpha_bar_prev,
            "sqrt_alpha_bar": alpha_bar.sqrt(),
            "sqrt_one_minus_alpha_bar": (1.0 - alpha_bar).sqrt(),
            "sqrt_recip_alpha_bar": (1.0 / alpha_bar).sqrt(),
            "sqrt_recip_alpha_bar_minus_one": (1.0 / alpha_bar - 1.0).sqrt(),
            "posterior_variance": posterior_variance,
            "posterior_mean_coef1": posterior_mean_coef1,
            "posterior_mean_coef2": posterior_mean_coef2,
        }
        for name, value in buffers.items():
            self.register_buffer(name, value.float(), persistent=False)

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Forward process (Ho et al. 2020, Eq. 4).
        x_t = sqrt(alpha_bar_t) · x0 + sqrt(1 − alpha_bar_t) · eps
        x0: (B, C, H, W) dalam [-1, 1], t: (B,) long → x_t: (B, C, H, W)
        """
        return (
            extract(self.sqrt_alpha_bar, t, x0.shape) * x0
            + extract(self.sqrt_one_minus_alpha_bar, t, x0.shape) * noise
        )

    def predict_x0_from_eps(self, xt: Tensor, t: Tensor, eps: Tensor) -> Tensor:
        """Inversi Eq. 4: x0 = (x_t − sqrt(1 − alpha_bar_t) · eps) / sqrt(alpha_bar_t).
        xt, eps: (B, C, H, W), t: (B,) long → x0: (B, C, H, W)
        """
        return (
            extract(self.sqrt_recip_alpha_bar, t, xt.shape) * xt
            - extract(self.sqrt_recip_alpha_bar_minus_one, t, xt.shape) * eps
        )

    def q_posterior_mean(self, x0: Tensor, xt: Tensor, t: Tensor) -> Tensor:
        """Mean posterior q(x_{t−1} | x_t, x_0) (Ho et al. 2020, Eq. 7).
        x0, xt: (B, C, H, W), t: (B,) long → mean: (B, C, H, W)
        """
        return (
            extract(self.posterior_mean_coef1, t, xt.shape) * x0
            + extract(self.posterior_mean_coef2, t, xt.shape) * xt
        )

    def loss(self, model: nn.Module, x0: Tensor) -> Tensor:
        """Loss sederhana L_simple (Ho et al. 2020, Eq. 14): ‖eps − eps_theta(x_t, t)‖².
        x0: (B, C, H, W) dalam [-1, 1] → skalar
        """
        t = torch.randint(0, self.timesteps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        return F.mse_loss(model(xt, t), noise)
