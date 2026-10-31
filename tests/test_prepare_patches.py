import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

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


def random_image(w: int, h: int, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8))


@pytest.mark.parametrize(("w", "h"), [(64, 64), (100, 80), (37, 50), (300, 120)])
def test_resize_mode_always_gives_one_64x64(w: int, h: int) -> None:
    out = prepare_patches.convert_image(random_image(w, h), "resize", 64, 32, 0.0, None)
    assert len(out) == 1
    assert out[0][1].shape == (64, 64, 3)


def test_resize_keeps_64x64_input_unchanged() -> None:
    img = random_image(64, 64)
    out = prepare_patches.convert_image(img, "resize", 64, 32, 0.0, None)
    assert np.array_equal(out[0][1], np.asarray(img))


def test_patch_mode_scales_short_side_first() -> None:
    # 400×200 → sisi pendek 128 → 256×128 → (128−64)/32+1 = 3 baris × (256−64)/32+1 = 7 kolom
    out = prepare_patches.convert_image(random_image(400, 200), "patch", 64, 32, 0.0, 128)
    assert len(out) == 21
    assert all(arr.shape == (64, 64, 3) for _, arr in out)


def test_patch_mode_upscales_images_smaller_than_patch() -> None:
    out = prepare_patches.convert_image(random_image(40, 30), "patch", 64, 32, 0.0, None)
    assert len(out) >= 1
    assert out[0][1].shape == (64, 64, 3)


def test_auto_mode_picks_resize_for_small_and_patch_for_large() -> None:
    small = prepare_patches.convert_image(random_image(200, 150), "auto", 64, 32, 0.0, 256)
    large = prepare_patches.convert_image(random_image(1024, 512), "auto", 64, 32, 0.0, 256)
    assert [name for name, _ in small] == ["r"]
    assert len(large) == 7 * 15  # 512×256 → 7 baris × 15 kolom
