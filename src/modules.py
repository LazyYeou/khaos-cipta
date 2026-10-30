"""Komponen U-Net: time embedding, ResBlock, self-attention, down/upsample."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def zero_init(module: nn.Module) -> nn.Module:
    """Nol-kan bobot & bias agar blok residual mulai sebagai identitas."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class SinusoidalTimeEmbedding(nn.Module):
    """Positional encoding sinusoidal (Vaswani et al. 2017) untuk timestep.
    t: (B,) long → (B, dim)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("dim time embedding harus genap")
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class TimeEmbedding(nn.Module):
    """Sinusoidal → MLP. t: (B,) long → (B, out_dim)"""

    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.sinusoidal = SinusoidalTimeEmbedding(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self.sinusoidal(t))


class ResBlock(nn.Module):
    """GroupNorm → SiLU → Conv, ditambah time embedding, lalu GroupNorm → SiLU → Dropout → Conv.
    x: (B, in_ch, H, W), temb: (B, time_dim) → (B, out_ch, H, W)
    """

    def __init__(
        self, in_ch: int, out_ch: int, time_dim: int, dropout: float, num_groups: int
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = zero_init(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.skip(x) + h


class AttentionBlock(nn.Module):
    """Self-attention satu head antar posisi spasial, dengan residual.
    x: (B, C, H, W) → (B, C, H, W)
    """

    def __init__(self, ch: int, num_groups: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = zero_init(nn.Conv2d(ch, ch, 1))

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(dim=1)
        # softmax(Qᵀ K / sqrt(C)) dihitung di float32 agar stabil saat mixed precision.
        weights = torch.einsum("bci,bcj->bij", q, k) / math.sqrt(c)
        weights = weights.float().softmax(dim=-1).to(v.dtype)
        out = torch.einsum("bij,bcj->bci", weights, v).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    """Conv stride 2. (B, C, H, W) → (B, C, H/2, W/2)"""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbor ×2 lalu conv. (B, C, H, W) → (B, C, 2H, 2W)"""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class TimestepSequential(nn.Sequential):
    """Sequential yang meneruskan time embedding hanya ke ResBlock."""

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        for layer in self:
            x = layer(x, temb) if isinstance(layer, ResBlock) else layer(x)
        return x
