"""Sampler DDPM dan DDIM.

Setiap sampler tersedia sebagai generator `*_steps` yang meng-yield x setelah tiap langkah
(dipakai untuk GIF reverse process), dan fungsi `sample` yang hanya mengembalikan hasil akhir.
Pemanggil bertanggung jawab memanggil `model.eval()` sebelum sampling.

Referensi:
- Ho, Jain & Abbeel (2020), "Denoising Diffusion Probabilistic Models".
- Song, Meng & Ermon (2021), "Denoising Diffusion Implicit Models".
"""

from collections.abc import Iterator

import torch
from torch import Tensor, nn

from src.diffusion import GaussianDiffusion, extract


@torch.no_grad()
def ddpm_steps(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    x_T: Tensor,
    generator: torch.Generator | None = None,
    clip_denoised: bool = True,
) -> Iterator[Tensor]:
    """Reverse process DDPM (Ho et al. 2020, Algorithm 2) dengan T langkah.
    x_T: (B, C, H, W) ~ N(0, I) → yield x_{t−1}: (B, C, H, W) untuk t = T … 1

    mu_theta dihitung lewat prediksi x0 lalu mean posterior (Eq. 7 & 15). Tanpa clipping ini
    identik dengan Eq. 11: mu = 1/sqrt(alpha_t) · (x_t − beta_t / sqrt(1 − alpha_bar_t) · eps).
    Clipping x0 ke [-1, 1] mencegah nilai keluar rentang data. Variansi: sigma_t² = beta_tilde_t.
    """
    x = x_T
    for i in reversed(range(diffusion.timesteps)):
        t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
        eps = model(x, t)
        x0 = diffusion.predict_x0_from_eps(x, t, eps)
        if clip_denoised:
            x0 = x0.clamp(-1, 1)
        mean = diffusion.q_posterior_mean(x0, x, t)
        if i > 0:
            noise = torch.randn(x.shape, device=x.device, generator=generator)
            x = mean + extract(diffusion.posterior_variance, t, x.shape).sqrt() * noise
        else:
            x = mean  # langkah terakhir (t = 0) tidak menambahkan noise
        yield x


def ddim_timesteps(timesteps: int, steps: int) -> list[int]:
    """Subsequence timestep berjarak rata dari 0 sampai T−1 (Song et al. 2021, Sec. 4.2)."""
    if not 1 <= steps <= timesteps:
        raise ValueError(f"steps harus di antara 1 dan {timesteps}, didapat {steps}")
    return torch.linspace(0, timesteps - 1, steps).round().long().tolist()


@torch.no_grad()
def ddim_steps(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    x_T: Tensor,
    steps: int,
    eta: float = 0.0,
    generator: torch.Generator | None = None,
    clip_denoised: bool = True,
) -> Iterator[Tensor]:
    """Reverse process DDIM (Song et al. 2021, Eq. 12) dengan `steps` langkah.
    x_T: (B, C, H, W) ~ N(0, I) → yield x_{tau_{i−1}}: (B, C, H, W)

    x_prev = sqrt(alpha_bar_prev) · x0_pred
             + sqrt(1 − alpha_bar_prev − sigma²) · eps
             + sigma · z
    dengan sigma = eta · sqrt((1 − alpha_bar_prev) / (1 − alpha_bar_t))
                       · sqrt(1 − alpha_bar_t / alpha_bar_prev)     (Eq. 16).
    eta = 0 → deterministik (DDIM), eta = 1 → setara DDPM.
    """
    seq = ddim_timesteps(diffusion.timesteps, steps)
    x = x_T
    for i in reversed(range(len(seq))):
        t = torch.full((x.shape[0],), seq[i], device=x.device, dtype=torch.long)
        alpha_bar = extract(diffusion.alpha_bar, t, x.shape)
        if i > 0:
            t_prev = torch.full_like(t, seq[i - 1])
            alpha_bar_prev = extract(diffusion.alpha_bar, t_prev, x.shape)
        else:
            alpha_bar_prev = torch.ones_like(alpha_bar)  # alpha_bar sebelum t = 0 adalah 1

        eps = model(x, t)
        x0 = diffusion.predict_x0_from_eps(x, t, eps)
        if clip_denoised:
            x0 = x0.clamp(-1, 1)
            # Hitung ulang eps agar konsisten dengan x0 yang sudah di-clip.
            eps = (x - alpha_bar.sqrt() * x0) / (1 - alpha_bar).sqrt()

        sigma = (
            eta
            * ((1 - alpha_bar_prev) / (1 - alpha_bar)).sqrt()
            * (1 - alpha_bar / alpha_bar_prev).sqrt()
        )
        # clamp(min=0) hanya menahan error pembulatan float saat eta = 1.
        x = alpha_bar_prev.sqrt() * x0 + (1 - alpha_bar_prev - sigma**2).clamp(min=0).sqrt() * eps
        if i > 0 and eta > 0:
            x = x + sigma * torch.randn(x.shape, device=x.device, generator=generator)
        yield x


def sample_steps(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    x_T: Tensor,
    sampler: str,
    steps: int,
    eta: float = 0.0,
    generator: torch.Generator | None = None,
) -> Iterator[Tensor]:
    """Pilih sampler berdasarkan nama ('ddpm' memakai semua T langkah, `steps` diabaikan)."""
    if sampler == "ddpm":
        return ddpm_steps(model, diffusion, x_T, generator)
    if sampler == "ddim":
        return ddim_steps(model, diffusion, x_T, steps, eta, generator)
    raise ValueError(f"sampler tidak dikenal: {sampler!r} (pilih 'ddpm' atau 'ddim')")


def sample(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    x_T: Tensor,
    sampler: str,
    steps: int,
    eta: float = 0.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Jalankan sampler sampai selesai. x_T: (B, C, H, W) → x_0: (B, C, H, W) dalam [-1, 1]"""
    x = x_T
    for x_step in sample_steps(model, diffusion, x_T, sampler, steps, eta, generator):
        x = x_step  # hanya simpan langkah terakhir agar memori tidak menumpuk
    return x.clamp(-1, 1)
