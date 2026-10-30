"""Generate motif baru dari checkpoint.

Contoh:
    python scripts/sample.py --ckpt runs/batik64/ema_last.pt --n 64 --sampler ddpm
    python scripts/sample.py --ckpt runs/batik64/ema_last.pt --n 64 --sampler ddim --steps 50
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.samplers import sample  # noqa: E402
from src.utils import get_device, load_model_for_sampling, save_grid, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--ckpt", required=True, help="last.pt atau ema_last.pt")
    parser.add_argument("--n", type=int, default=16, help="jumlah gambar")
    parser.add_argument(
        "--sampler", choices=["ddpm", "ddim"], help="default: sample.sampler di config"
    )
    parser.add_argument("--steps", type=int, help="langkah DDIM (default: sample.steps di config)")
    parser.add_argument("--eta", type=float, help="eta DDIM (default: sample.eta di config)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="path PNG grid (default: <folder ckpt>/sample_<sampler>.png)")
    parser.add_argument("--device", help="cuda / cpu (default: otomatis)")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    model, diffusion, cfg = load_model_for_sampling(args.ckpt, device)
    sampler = args.sampler or cfg["sample"]["sampler"]
    steps = args.steps or cfg["sample"]["steps"]
    eta = cfg["sample"]["eta"] if args.eta is None else args.eta

    size, channels = cfg["data"]["image_size"], cfg["data"]["channels"]
    gen = torch.Generator(device).manual_seed(args.seed)
    images = []
    for start in range(0, args.n, args.batch_size):
        b = min(args.batch_size, args.n - start)
        x_T = torch.randn(b, channels, size, size, device=device, generator=gen)
        images.append(sample(model, diffusion, x_T, sampler, steps, eta, gen).cpu())
        print(f"{start + b}/{args.n} gambar")

    out = Path(args.out) if args.out else Path(args.ckpt).parent / f"sample_{sampler}.png"
    save_grid(torch.cat(images), out)
    print(f"tersimpan: {out}")


if __name__ == "__main__":
    main()
