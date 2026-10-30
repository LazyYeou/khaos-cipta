from pathlib import Path

import pytest
import torch
from PIL import Image

from src.utils import CSVLogger, apply_overrides, deep_merge, save_grid, to_uint8


def test_deep_merge_does_not_mutate_base() -> None:
    base = {"train": {"lr": 1e-4, "batch_size": 8}, "seed": 0}
    merged = deep_merge(base, {"train": {"lr": 2e-4}})

    assert merged == {"train": {"lr": 2e-4, "batch_size": 8}, "seed": 0}
    assert base["train"]["lr"] == 1e-4


def test_apply_overrides_parses_yaml_values() -> None:
    cfg = {"train": {"lr": 1e-4, "amp": True}, "model": {"channel_mult": [1, 2]}}
    out = apply_overrides(cfg, ["train.lr=3e-4", "train.amp=false", "model.channel_mult=[1,2,4]"])

    assert out["train"] == {"lr": 3e-4, "amp": False}
    assert out["model"]["channel_mult"] == [1, 2, 4]


def test_apply_overrides_rejects_missing_equals() -> None:
    with pytest.raises(ValueError):
        apply_overrides({}, ["train.lr"])


def test_to_uint8_maps_range() -> None:
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0]).reshape(1, 1, 2, 2)
    assert to_uint8(x).flatten().tolist() == [0, 128, 255, 255]


def test_save_grid_writes_png(tmp_path: Path) -> None:
    path = tmp_path / "grid" / "samples.png"
    save_grid(torch.rand(4, 3, 8, 8) * 2 - 1, path)
    with Image.open(path) as img:
        assert img.size == (2 * 8 + 3 * 2, 2 * 8 + 3 * 2)


def test_csv_logger_appends_without_duplicate_header(tmp_path: Path) -> None:
    path = tmp_path / "log.csv"
    CSVLogger(path, ["step", "loss"]).log({"step": 1, "loss": 0.5})
    CSVLogger(path, ["step", "loss"]).log({"step": 2, "loss": 0.4})

    assert path.read_text().splitlines() == ["step,loss", "1,0.5", "2,0.4"]
