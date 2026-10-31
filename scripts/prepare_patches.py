"""Ubah foto / gambar pola batik berukuran apa pun menjadi gambar persegi `--size` (default 64×64).

Mode:
- resize : crop tengah menjadi persegi lalu perkecil ke size×size (1 gambar per input).
           Cocok untuk gambar satu motif / ubin pola.
- patch  : (opsional) skalakan sisi terpendek ke --short-side, lalu potong sliding window
           size×size dengan --stride. Cocok untuk foto kain besar.
- auto   : (default) gambar dengan sisi terpendek ≤ --short-side diproses dengan `resize`,
           yang lebih besar diskalakan ke --short-side lalu dipotong dengan `patch`.

Pada semua mode, hasil yang terlalu polos (std piksel < --min-std) dibuang.

Contoh:
    python scripts/prepare_patches.py --input data/raw --output data/patches \
        --size 64 --stride 32 --min-std 0.05
    python scripts/prepare_patches.py --input data/raw --output data/patches --mode resize
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from src.dataset import list_images  # noqa: E402
from src.utils import set_seed  # noqa: E402

MODES = ("auto", "resize", "patch")


def extract_patches(
    image: np.ndarray, size: int, stride: int, min_std: float
) -> list[tuple[int, int, np.ndarray]]:
    """Sliding window di atas gambar (H, W, C) uint8 → daftar (y, x, patch (size, size, C)).
    Patch dengan std piksel (skala [0, 1]) di bawah `min_std` dianggap polos dan dibuang.
    """
    h, w = image.shape[:2]
    patches = []
    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            patch = image[y : y + size, x : x + size]
            if (patch / 255.0).std() >= min_std:
                patches.append((y, x, patch))
    return patches


def center_square_resize(img: Image.Image, size: int) -> Image.Image:
    """Crop tengah menjadi persegi lalu resize ke size×size (LANCZOS, tidak merusak rasio)."""
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    square = img.crop((left, top, left + side, top + side))
    return square if side == size else square.resize((size, size), Image.Resampling.LANCZOS)


def scale_short_side(img: Image.Image, short_side: int) -> Image.Image:
    """Skalakan gambar agar sisi terpendeknya = short_side (rasio aspek dipertahankan)."""
    w, h = img.size
    scale = short_side / min(w, h)
    if scale == 1:
        return img
    new_size = (max(short_side, round(w * scale)), max(short_side, round(h * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def convert_image(
    img: Image.Image, mode: str, size: int, stride: int, min_std: float, short_side: int | None
) -> list[tuple[str, np.ndarray]]:
    """Ubah satu gambar RGB menjadi daftar (sufiks nama, array (size, size, 3) uint8)."""
    if mode == "auto":
        if short_side is None:
            raise ValueError("mode auto membutuhkan short_side")
        mode = "resize" if min(img.size) <= short_side else "patch"

    if mode == "resize":
        out = np.asarray(center_square_resize(img, size))
        return [("r", out)] if (out / 255.0).std() >= min_std else []
    if mode == "patch":
        if short_side is not None:
            img = scale_short_side(img, max(short_side, size))
        elif min(img.size) < size:
            img = scale_short_side(img, size)  # gambar lebih kecil dari patch → perbesar
        patches = extract_patches(np.asarray(img), size, stride, min_std)
        return [(f"y{y:05d}_x{x:05d}", p) for y, x, p in patches]
    raise ValueError(f"mode tidak dikenal: {mode!r} (pilih {', '.join(MODES)})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input", required=True, help="folder foto mentah (dibaca rekursif)")
    parser.add_argument("--output", required=True, help="folder tujuan gambar PNG")
    parser.add_argument("--mode", choices=MODES, default="auto")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--min-std", type=float, default=0.05, help="buang hasil yang polos")
    parser.add_argument(
        "--short-side",
        type=int,
        default=256,
        help="auto: batas resize vs patch dan target skala; patch: target skala (0 = tanpa skala)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)
    short_side = args.short_side or None

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    photos = list_images(args.input)
    if not photos:
        raise SystemExit(f"tidak ada gambar di {args.input}")

    total = 0
    for photo in photos:
        with Image.open(photo) as img:
            results = convert_image(
                img.convert("RGB"), args.mode, args.size, args.stride, args.min_std, short_side
            )
        # Nama memuat path relatif agar foto bernama sama di subfolder berbeda tidak bertabrakan.
        stem = "_".join(photo.relative_to(args.input).with_suffix("").parts)
        for suffix, arr in results:
            Image.fromarray(arr).save(out_dir / f"{stem}_{suffix}.png")
        total += len(results)
        print(f"{photo}: {len(results)} gambar")
    print(f"total {total} gambar {args.size}x{args.size} dari {len(photos)} foto -> {out_dir}")


if __name__ == "__main__":
    main()
