"""Dataset gambar dalam rentang [-1, 1] beserta augmentasi.

Semua dataset mengembalikan satu tensor (C, H, W) float32 dalam [-1, 1] per item (tanpa label).
"""

import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets, transforms

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class RandomRot90:
    """Rotasi acak kelipatan 90°. (C, H, W) → (C, H, W) untuk gambar persegi."""

    def __call__(self, x: Tensor) -> Tensor:
        return torch.rot90(x, random.randint(0, 3), dims=(1, 2))


def build_transform(image_size: int, hflip: bool, vflip: bool, rot90: bool) -> transforms.Compose:
    """Resize/crop ke `image_size`, augmentasi geometris saja (tanpa augmentasi warna agar palet
    batik terjaga), lalu skala piksel [0, 1] → [-1, 1]."""
    steps: list[Any] = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
    if hflip:
        steps.append(transforms.RandomHorizontalFlip())
    if vflip:
        steps.append(transforms.RandomVerticalFlip())
    if rot90:
        steps.append(RandomRot90())
    steps.append(transforms.Normalize(0.5, 0.5))  # (x − 0.5) / 0.5 → [-1, 1]
    return transforms.Compose(steps)


def list_images(root: str | Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


class ImageFolderDataset(Dataset):
    """Semua gambar di dalam `root` (rekursif), misalnya patch batik dari prepare_patches.py."""

    def __init__(self, root: str | Path, transform: transforms.Compose, channels: int) -> None:
        self.paths = list_images(root)
        if not self.paths:
            raise FileNotFoundError(f"tidak ada gambar di {root}")
        self.transform = transform
        self.mode = "RGB" if channels == 3 else "L"

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        with Image.open(self.paths[index]) as img:
            return self.transform(img.convert(self.mode))


class CIFAR10Images(Dataset):
    """CIFAR-10 train split tanpa label (diunduh otomatis ke `root`)."""

    def __init__(self, root: str | Path, transform: transforms.Compose) -> None:
        self.data = datasets.CIFAR10(root, train=True, download=True, transform=transform)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tensor:
        return self.data[index][0]


class SyntheticDataset(Dataset):
    """Gambar acak deterministik, hanya untuk smoke test (tanpa unduhan atau file data)."""

    def __init__(self, num_samples: int, channels: int, image_size: int, seed: int = 0) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.images = torch.rand(num_samples, channels, image_size, image_size, generator=gen)
        self.images = self.images * 2 - 1

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tensor:
        return self.images[index]


def build_dataset(data_cfg: dict[str, Any]) -> Dataset:
    """Buat dataset dari bagian `data` di config."""
    kind = data_cfg["dataset"]
    size = data_cfg["image_size"]
    transform = build_transform(size, data_cfg["hflip"], data_cfg["vflip"], data_cfg["rot90"])
    if kind == "folder":
        return ImageFolderDataset(data_cfg["root"], transform, data_cfg["channels"])
    if kind == "cifar10":
        return CIFAR10Images(data_cfg["root"], transform)
    if kind == "synthetic":
        return SyntheticDataset(data_cfg["num_samples"], data_cfg["channels"], size)
    raise ValueError(f"dataset tidak dikenal: {kind!r} (pilih 'folder', 'cifar10', 'synthetic')")
