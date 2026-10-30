"""Interpolasi antar dua motif hasil generate.

Dua noise x_T diinterpolasi secara spherical (slerp), lalu masing-masing di-denoise dengan DDIM
eta = 0 (deterministik), sehingga transisi antar motif terlihat mulus. Setiap baris = satu pasang.

Contoh:
    python scripts/interpolate.py --ckpt runs/batik64/ema_last.pt --steps 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch import Tensor  # noqa: E402

from src.samplers import sample  # noqa: E402
from src.utils import get_device, load_model_for_sampling, save_grid, set_seed  # noqa: E402


def slerp(a: Tensor, b: Tensor, w: Tensor) -> Tensor:
    """Spherical linear interpolation antar noise Gaussian.
    a, b: (C, H, W), w: (N,) dalam [0, 1] → (N, C, H, W)
    Slerp menjaga norma noise tetap sesuai N(0, I); interpolasi linear akan mengecilkannya.
    """
    a_flat, b_flat = a.flatten(), b.flatten()
    cos = torch.dot(a_flat, b_flat) / (a_flat.norm() * b_flat.norm())
    theta = torch.arccos(cos.clamp(-1, 1))
    w = w[:, None]
    out = (torch.sin((1 - w) * theta) * a_flat + torch.sin(w * theta) * b_flat) / torch.sin(theta)
    return out.reshape(len(w), *a.shape)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--steps", type=int, default=10, help="jumlah titik interpolasi per baris")
    parser.add_argument("--pairs", type=int, default=4, help="jumlah pasangan motif (baris)")
    parser.add_argument("--ddim-steps", type=int, help="langkah DDIM (default: sample.steps)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="assets/interpolation.png")
    parser.add_argument("--device", help="cuda / cpu (default: otomatis)")
    args = parser.parse_args()
    set_seed(args.seed)

    device = get_device(args.device)
    model, diffusion, cfg = load_model_for_sampling(args.ckpt, device)
    ddim_steps = args.ddim_steps or cfg["sample"]["steps"]
    shape = (cfg["data"]["channels"], cfg["data"]["image_size"], cfg["data"]["image_size"])
    gen = torch.Generator(device).manual_seed(args.seed)
    weights = torch.linspace(0, 1, args.steps, device=device)

    rows = []
    for _ in range(args.pairs):
        a = torch.randn(shape, device=device, generator=gen)
        b = torch.randn(shape, device=device, generator=gen)
        rows.append(sample(model, diffusion, slerp(a, b, weights), "ddim", ddim_steps, eta=0.0))

    save_grid(torch.cat(rows), args.out, nrow=args.steps)
    print(f"tersimpan: {args.out}")


if __name__ == "__main__":
    main()
