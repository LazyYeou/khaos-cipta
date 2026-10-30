import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location(
    "interpolate", Path(__file__).resolve().parents[1] / "scripts" / "interpolate.py"
)
interpolate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(interpolate)


def test_slerp_endpoints_and_norm() -> None:
    torch.manual_seed(0)
    a, b = torch.randn(3, 8, 8), torch.randn(3, 8, 8)
    out = interpolate.slerp(a, b, torch.linspace(0, 1, 5))

    assert out.shape == (5, 3, 8, 8)
    assert torch.allclose(out[0], a, atol=1e-5)
    assert torch.allclose(out[-1], b, atol=1e-5)
    # Titik tengah slerp tetap bernorma mirip noise asli (lerp akan menyusut ~1/sqrt(2)).
    assert abs(out[2].norm() / a.norm() - 1) < 0.1
