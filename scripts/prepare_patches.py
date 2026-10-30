"""Potong foto kain batik menjadi patch persegi.

Contoh:
    python scripts/prepare_patches.py --input data/raw --output data/patches \
        --size 64 --stride 32 --min-std 0.05
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from src.dataset import list_images  # noqa: E402
from src.utils import set_seed  # noqa: E402


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input", required=True, help="folder foto mentah (dibaca rekursif)")
    parser.add_argument("--output", required=True, help="folder tujuan patch PNG")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--min-std", type=float, default=0.05, help="buang patch polos")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    photos = list_images(args.input)
    if not photos:
        raise SystemExit(f"tidak ada gambar di {args.input}")

    total = 0
    for photo in photos:
        with Image.open(photo) as img:
            image = np.asarray(img.convert("RGB"))
        patches = extract_patches(image, args.size, args.stride, args.min_std)
        # Nama memuat path relatif agar foto bernama sama di subfolder berbeda tidak bertabrakan.
        stem = "_".join(photo.relative_to(args.input).with_suffix("").parts)
        for y, x, patch in patches:
            Image.fromarray(patch).save(out_dir / f"{stem}_y{y:05d}_x{x:05d}.png")
        total += len(patches)
        print(f"{photo}: {len(patches)} patch")
    print(f"total {total} patch dari {len(photos)} foto -> {out_dir}")


if __name__ == "__main__":
    main()
