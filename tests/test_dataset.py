from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.dataset import build_dataset


def data_cfg(**overrides: object) -> dict:
    cfg = {
        "dataset": "folder",
        "root": "",
        "image_size": 16,
        "channels": 3,
        "hflip": True,
        "vflip": True,
        "rot90": True,
        "num_samples": 4,
    }
    cfg.update(overrides)
    return cfg


def test_folder_dataset_normalizes_to_minus_one_one(tmp_path: Path) -> None:
    black = np.zeros((16, 16, 3), dtype=np.uint8)
    white = np.full((20, 24, 3), 255, dtype=np.uint8)  # ukuran lain → di-resize & crop
    Image.fromarray(black).save(tmp_path / "a.png")
    (tmp_path / "sub").mkdir()
    Image.fromarray(white).save(tmp_path / "sub" / "b.jpg")

    dataset = build_dataset(data_cfg(root=str(tmp_path)))
    assert len(dataset) == 2
    a, b = dataset[0], dataset[1]
    assert a.shape == b.shape == (3, 16, 16)
    assert a.dtype == torch.float32
    assert torch.allclose(a, torch.full_like(a, -1.0))
    assert torch.allclose(b, torch.full_like(b, 1.0), atol=0.02)


def test_synthetic_dataset_shape_and_range() -> None:
    dataset = build_dataset(data_cfg(dataset="synthetic"))
    assert len(dataset) == 4
    x = dataset[0]
    assert x.shape == (3, 16, 16)
    assert x.min() >= -1 and x.max() <= 1
