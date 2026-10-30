import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "prepare_patches", Path(__file__).resolve().parents[1] / "scripts" / "prepare_patches.py"
)
prepare_patches = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_patches)


def test_sliding_window_positions() -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(128, 96, 3), dtype=np.uint8)
    patches = prepare_patches.extract_patches(image, size=64, stride=32, min_std=0.0)

    positions = [(y, x) for y, x, _ in patches]
    assert positions == [(0, 0), (0, 32), (32, 0), (32, 32), (64, 0), (64, 32)]
    assert all(p.shape == (64, 64, 3) for _, _, p in patches)


def test_plain_patches_are_dropped() -> None:
    image = np.full((64, 128, 3), 200, dtype=np.uint8)  # kiri polos
    image[:, 64:] = np.random.default_rng(0).integers(0, 256, size=(64, 64, 3))  # kanan bermotif
    patches = prepare_patches.extract_patches(image, size=64, stride=64, min_std=0.05)

    assert [(y, x) for y, x, _ in patches] == [(0, 64)]
