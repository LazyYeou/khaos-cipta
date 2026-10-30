"""Training DDPM.

Contoh:
    python scripts/train.py --config configs/batik64.yaml
    python scripts/train.py --config configs/batik64.yaml --resume
    python scripts/train.py --config configs/cifar10.yaml --smoke-test
    python scripts/train.py --config configs/batik64.yaml --override train.lr=1e-4
"""

import argparse
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.dataset import build_dataset  # noqa: E402
from src.ema import EMA  # noqa: E402
from src.samplers import sample  # noqa: E402
from src.utils import (  # noqa: E402
    CSVLogger,
    apply_overrides,
    build_diffusion,
    build_model,
    deep_merge,
    get_device,
    load_config,
    save_checkpoint,
    save_config,
    save_grid,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", required=True, help="file YAML di configs/")
    parser.add_argument("--seed", type=int, help="override seed di config")
    parser.add_argument("--run-name", help="override run_name (folder di runs/)")
    parser.add_argument("--resume", action="store_true", help="lanjutkan dari runs/<run>/last.pt")
    parser.add_argument("--smoke-test", action="store_true", help="pipeline mini, < 2 menit")
    parser.add_argument("--override", nargs="*", default=[], help="contoh: train.lr=1e-4")
    parser.add_argument("--device", help="cuda / cpu (default: otomatis)")
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    smoke = cfg.pop("smoke_test", {})
    if args.smoke_test:
        cfg = deep_merge(cfg, smoke)
    cfg = apply_overrides(cfg, args.override)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.run_name:
        cfg["run_name"] = args.run_name
    return cfg


def infinite_batches(loader: DataLoader) -> Iterator[torch.Tensor]:
    """Ulangi DataLoader tanpa henti (shuffle ulang tiap epoch)."""
    while True:
        yield from loader


def main() -> None:
    args = parse_args()
    cfg = resolve_config(args)
    tcfg = cfg["train"]
    set_seed(cfg["seed"])
    device = get_device(args.device)
    use_amp = tcfg["amp"] and device.type == "cuda"

    run_dir = Path("runs") / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    dataset = build_dataset(cfg["data"])
    if len(dataset) < tcfg["batch_size"]:
        raise ValueError(f"dataset ({len(dataset)}) lebih kecil dari batch_size")
    loader = DataLoader(
        dataset,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=cfg["data"]["num_workers"] > 0,
    )

    model = build_model(cfg).to(device)
    diffusion = build_diffusion(cfg).to(device)
    ema = EMA(model, tcfg["ema_decay"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )
    # Warm-up linear learning rate, lalu konstan (Ho et al. 2020).
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / tcfg["warmup_steps"])
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    step = 0
    if args.resume:
        ckpt_path = run_dir / "last.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        step = ckpt["step"]
        print(f"melanjutkan dari {ckpt_path} (step {step})")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"run: {run_dir} | device: {device} | params: {n_params / 1e6:.2f}M | amp: {use_amp}")

    logger = CSVLogger(run_dir / "log.csv", ["step", "loss", "lr", "time"])
    # Noise tetap untuk grid sampel berkala agar perkembangan antar-step bisa dibandingkan.
    shape = (tcfg["n_samples"], cfg["data"]["channels"], *(2 * [cfg["data"]["image_size"]]))
    fixed_gen = torch.Generator(device).manual_seed(tcfg["sample_seed"])
    fixed_noise = torch.randn(shape, device=device, generator=fixed_gen)

    batches = infinite_batches(loader)
    start_time = time.time()
    model.train()
    while step < tcfg["total_steps"]:
        x0 = next(batches).to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            loss = diffusion.loss(model, x0)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)
        step += 1

        last_step = step == tcfg["total_steps"]
        if step % tcfg["log_every"] == 0 or last_step:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start_time
            logger.log(
                {"step": step, "loss": f"{loss.item():.6f}", "lr": lr, "time": f"{elapsed:.1f}"}
            )
            print(f"step {step:>7d} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.0f}s")

        if step % tcfg["sample_every"] == 0 or last_step:
            gen = torch.Generator(device).manual_seed(tcfg["sample_seed"])
            scfg = cfg["sample"]
            samples = sample(
                ema.model, diffusion, fixed_noise, scfg["sampler"], scfg["steps"], scfg["eta"], gen
            )
            save_grid(samples, run_dir / "samples" / f"step_{step:06d}.png")

        if step % tcfg["save_every"] == 0 or last_step:
            state = {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "config": cfg,
            }
            save_checkpoint(state, run_dir / "last.pt")
            save_checkpoint(
                {"ema": ema.state_dict(), "step": step, "config": cfg}, run_dir / "ema_last.pt"
            )

    print(f"selesai: {step} step, checkpoint di {run_dir}")


if __name__ == "__main__":
    main()
