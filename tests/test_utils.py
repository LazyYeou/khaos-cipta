from pathlib import Path

import pytest
import torch
from PIL import Image

from src.utils import (
    AsyncCheckpointWriter,
    CSVLogger,
    apply_overrides,
    deep_merge,
    load_config,
    save_grid,
    to_uint8,
)


def test_load_config_extends_base(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("run_name: base\ntrain:\n  batch_size: 128\n  lr: 2.0e-4\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\ntrain:\n  batch_size: 64\n")

    assert load_config(tmp_path / "child.yaml") == {
        "run_name": "base",
        "train": {"batch_size": 64, "lr": 2e-4},
    }


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


def test_async_writer_snapshots_state_before_returning(tmp_path: Path) -> None:
    weights = torch.zeros(3)
    writer = AsyncCheckpointWriter()
    writer.save({tmp_path / "last.pt": {"w": weights, "step": 1}})
    weights.add_(5.0)  # training melanjutkan update in-place sebelum file selesai ditulis
    writer.wait()

    saved = torch.load(tmp_path / "last.pt")
    assert torch.equal(saved["w"], torch.zeros(3))
    assert saved["step"] == 1
    assert not (tmp_path / "last.pt.tmp").exists()


def test_async_writer_reraises_write_errors(tmp_path: Path) -> None:
    writer = AsyncCheckpointWriter()
    writer.save({tmp_path / "tidak_ada" / "last.pt": {"step": 1}})
    with pytest.raises(RuntimeError, match="gagal menyimpan checkpoint"):
        writer.wait()
    writer.wait()  # error hanya dilempar sekali
