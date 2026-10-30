"""Utilitas: seed, config YAML, simpan gambar, logging CSV, dan checkpoint."""

import copy
import csv
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch import Tensor
from torchvision.utils import make_grid

from src.diffusion import GaussianDiffusion
from src.unet import UNet

# ---------------------------------------------------------------- seed & device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- config


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Gabungkan `updates` ke salinan `base` secara rekursif."""
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Terapkan override CLI berbentuk 'a.b.c=nilai'. Nilai di-parse sebagai YAML."""
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override harus berbentuk key=value, didapat {item!r}")
        update: dict[str, Any] = {}
        node = update
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = _parse_value(raw)
        cfg = deep_merge(cfg, update)
    return cfg


def _parse_value(raw: str) -> Any:
    # PyYAML (YAML 1.1) membaca '1e-4' sebagai string, jadi coba float sebagai fallback.
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


# ---------------------------------------------------------------- gambar


def to_uint8(x: Tensor) -> Tensor:
    """(B, C, H, W) dalam [-1, 1] → (B, C, H, W) uint8 dalam [0, 255]"""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def to_pil(img: Tensor) -> Image.Image:
    """(C, H, W) dalam [-1, 1] → PIL Image"""
    arr = to_uint8(img[None])[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr.squeeze(-1) if arr.shape[-1] == 1 else arr)


def grid_image(x: Tensor, nrow: int) -> Image.Image:
    """(B, C, H, W) dalam [-1, 1] → satu PIL Image berisi grid."""
    return to_pil(make_grid(x.detach().float().cpu(), nrow=nrow, padding=2, pad_value=1.0))


def save_grid(x: Tensor, path: str | Path, nrow: int | None = None) -> None:
    """Simpan batch (B, C, H, W) dalam [-1, 1] sebagai grid PNG."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nrow = nrow or max(1, int(len(x) ** 0.5))
    grid_image(x, nrow).save(path)


def save_images(x: Tensor, out_dir: str | Path, start_index: int = 0) -> None:
    """Simpan setiap gambar (B, C, H, W) dalam [-1, 1] sebagai PNG terpisah."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(x):
        to_pil(img).save(out_dir / f"{start_index + i:06d}.png")


# ---------------------------------------------------------------- logging


class CSVLogger:
    """Tambahkan baris ke CSV; header ditulis hanya jika file baru."""

    def __init__(self, path: str | Path, fields: list[str]) -> None:
        self.path = Path(path)
        self.fields = fields
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    def log(self, row: dict[str, Any]) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(row)


# ---------------------------------------------------------------- model & checkpoint


def build_model(cfg: dict[str, Any]) -> UNet:
    m, d = cfg["model"], cfg["data"]
    return UNet(
        image_size=d["image_size"],
        in_channels=d["channels"],
        base_channels=m["base_channels"],
        channel_mult=m["channel_mult"],
        num_res_blocks=m["num_res_blocks"],
        attn_resolutions=m["attn_resolutions"],
        dropout=m["dropout"],
        num_groups=m["num_groups"],
    )


def build_diffusion(cfg: dict[str, Any]) -> GaussianDiffusion:
    d = cfg["diffusion"]
    return GaussianDiffusion(
        timesteps=d["timesteps"],
        schedule=d["schedule"],
        beta_start=d["beta_start"],
        beta_end=d["beta_end"],
        cosine_s=d["cosine_s"],
    )


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    """Simpan secara atomik (tulis ke file sementara lalu rename) agar checkpoint tidak korup
    jika sesi terputus di tengah penyimpanan."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_model_for_sampling(
    ckpt_path: str | Path, device: torch.device
) -> tuple[UNet, GaussianDiffusion, dict[str, Any]]:
    """Muat model (bobot EMA jika ada) + diffusion dari checkpoint, siap untuk sampling."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg).to(device)
    weights = ckpt["ema"]["model"] if "ema" in ckpt else ckpt["model"]
    model.load_state_dict(weights)
    model.eval()
    return model, build_diffusion(cfg).to(device), cfg
