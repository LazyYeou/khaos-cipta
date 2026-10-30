"""Hitung FID antara gambar hasil generate dan dataset asli (memakai torch-fidelity).

Gambar hasil generate disimpan sebagai PNG di --gen-dir lalu dibandingkan dengan folder data asli
(untuk config CIFAR-10 dipakai statistik bawaan torch-fidelity 'cifar10-train').

Contoh:
    python scripts/evaluate_fid.py --ckpt runs/batik64/ema_last.pt --n 10000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.samplers import sample  # noqa: E402
from src.utils import get_device, load_model_for_sampling, save_images, set_seed  # noqa: E402


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n", type=int, default=10000, help="jumlah gambar yang di-generate")
    parser.add_argument(
        "--sampler", choices=["ddpm", "ddim"], help="default: sample.sampler di config"
    )
    parser.add_argument("--steps", type=int, help="langkah DDIM (default: sample.steps di config)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--real-dir", help="folder gambar asli (default: data.root di config)")
    parser.add_argument("--gen-dir", help="default: <folder ckpt>/fid_<sampler>")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", help="cuda / cpu (default: otomatis)")
    args = parser.parse_args()
    set_seed(args.seed)

    try:
        import torch_fidelity
    except ImportError as e:
        raise SystemExit("torch-fidelity belum terpasang: pip install -r requirements.txt") from e

    device = get_device(args.device)
    model, diffusion, cfg = load_model_for_sampling(args.ckpt, device)
    sampler = args.sampler or cfg["sample"]["sampler"]
    steps = args.steps or cfg["sample"]["steps"]
    size, channels = cfg["data"]["image_size"], cfg["data"]["channels"]
    gen_dir = Path(args.gen_dir or Path(args.ckpt).parent / f"fid_{sampler}")
    if gen_dir.exists() and any(gen_dir.iterdir()):
        raise SystemExit(f"{gen_dir} tidak kosong; hapus dulu atau pakai --gen-dir lain")

    gen = torch.Generator(device).manual_seed(args.seed)
    for start in range(0, args.n, args.batch_size):
        b = min(args.batch_size, args.n - start)
        x_T = torch.randn(b, channels, size, size, device=device, generator=gen)
        x0 = sample(model, diffusion, x_T, sampler, steps, cfg["sample"]["eta"], gen)
        save_images(x0, gen_dir, start_index=start)
        print(f"{start + b}/{args.n} gambar")

    if args.real_dir:
        real = args.real_dir
    elif cfg["data"]["dataset"] == "cifar10":
        real = "cifar10-train"
    else:
        real = cfg["data"]["root"]

    metrics = torch_fidelity.calculate_metrics(
        input1=str(gen_dir), input2=str(real), cuda=device.type == "cuda", fid=True, verbose=False
    )
    fid = metrics["frechet_inception_distance"]
    print(f"FID ({sampler}, n={args.n}) = {fid:.2f}")


if __name__ == "__main__":
    main()
