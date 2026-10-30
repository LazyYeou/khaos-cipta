"""Visualisasi proses diffusion.

- Reverse process: GIF grid sampel dari noise murni sampai motif jadi.
- Forward process (opsional, --forward-image): satu gambar asli yang perlahan menjadi noise.

Contoh:
    python scripts/make_gif.py --ckpt runs/batik64/ema_last.pt --out assets/reverse_process.gif
    python scripts/make_gif.py --ckpt runs/batik64/ema_last.pt \
        --forward-image data/patches/contoh.png --forward-out assets/forward_process.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.dataset import build_transform  # noqa: E402
from src.diffusion import GaussianDiffusion  # noqa: E402
from src.samplers import sample_steps  # noqa: E402
from src.utils import get_device, grid_image, load_model_for_sampling, save_grid, set_seed  # noqa: E402


def save_forward_strip(
    image_path: str, diffusion: GaussianDiffusion, image_size: int, n: int, out: str
) -> None:
    """Simpan strip x_t untuk n timestep berjarak rata dari 0 sampai T−1 (noise yang sama)."""
    transform = build_transform(image_size, hflip=False, vflip=False, rot90=False)
    with Image.open(image_path) as img:
        x0 = transform(img.convert("RGB")).to(diffusion.betas.device)
    t = torch.linspace(0, diffusion.timesteps - 1, n, device=x0.device).round().long()
    noise = torch.randn_like(x0).expand(n, -1, -1, -1)
    save_grid(diffusion.q_sample(x0.expand(n, -1, -1, -1), t, noise), out, nrow=n)
    print(f"tersimpan: {out}")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="assets/reverse_process.gif")
    parser.add_argument(
        "--sampler", choices=["ddpm", "ddim"], help="default: sample.sampler di config"
    )
    parser.add_argument("--steps", type=int, help="langkah DDIM (default: sample.steps di config)")
    parser.add_argument("--n", type=int, default=16, help="jumlah sampel di grid")
    parser.add_argument("--frames", type=int, default=50, help="jumlah frame GIF")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--forward-image", help="gambar asli untuk strip forward process")
    parser.add_argument("--forward-out", default="assets/forward_process.png")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", help="cuda / cpu (default: otomatis)")
    args = parser.parse_args()
    set_seed(args.seed)

    device = get_device(args.device)
    model, diffusion, cfg = load_model_for_sampling(args.ckpt, device)
    size, channels = cfg["data"]["image_size"], cfg["data"]["channels"]

    if args.forward_image:
        save_forward_strip(args.forward_image, diffusion, size, 8, args.forward_out)

    sampler = args.sampler or cfg["sample"]["sampler"]
    steps = args.steps or cfg["sample"]["steps"]
    total = diffusion.timesteps if sampler == "ddpm" else steps
    keep = set(np.linspace(0, total - 1, min(args.frames, total)).round().astype(int).tolist())

    gen = torch.Generator(device).manual_seed(args.seed)
    x_T = torch.randn(args.n, channels, size, size, device=device, generator=gen)
    nrow = max(1, int(args.n**0.5))
    frames = [np.asarray(grid_image(x_T, nrow))]
    steps_iter = sample_steps(model, diffusion, x_T, sampler, steps, cfg["sample"]["eta"], gen)
    for i, x in enumerate(steps_iter):
        if i in keep:
            frames.append(np.asarray(grid_image(x, nrow)))
    frames += [frames[-1]] * args.fps  # tahan frame terakhir ~1 detik

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, duration=1 / args.fps, loop=0)
    print(f"tersimpan: {args.out} ({len(frames)} frame)")


if __name__ == "__main__":
    main()
