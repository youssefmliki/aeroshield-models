"""Convert CubiCasa5K SVGs → aligned masks + manifest.csv (official splits)."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .labels import CLASS_NAMES, CLASS_COLORS
from .svg_to_mask import read_svg_dims, svg_to_mask

# Prefer SVG raster for image alignment when cairosvg available; else skip image export note.
TARGET_LONG_SIDE = 1024


def _read_split(data_dir: Path, split: str) -> list[str]:
    p = data_dir / f"{split}.txt"
    if not p.exists():
        raise FileNotFoundError(f"Missing official split file: {p}")
    return [line.strip().strip("/") for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _colorize(mask: np.ndarray) -> Image.Image:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for idx, name in enumerate(CLASS_NAMES):
        rgb[mask == idx] = CLASS_COLORS[name]
    return Image.fromarray(rgb)


def prepare(data_dir: Path, out_dir: Path, overlay_n: int = 15, seed: int = 42) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    overlays_dir = out_dir / "overlays_qa"
    for d in (images_dir, masks_dir, overlays_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    seen_ids: dict[str, str] = {}

    for split in ("train", "val", "test"):
        samples = _read_split(data_dir, split)
        for rel in tqdm(samples, desc=f"prepare {split}"):
            source_id = rel.replace("\\", "/").strip("/")
            if source_id in seen_ids and seen_ids[source_id] != split:
                raise RuntimeError(f"Split leakage for {source_id}: {seen_ids[source_id]} vs {split}")
            seen_ids[source_id] = split

            sample_dir = data_dir / rel
            svg = sample_dir / "model.svg"
            if not svg.exists():
                continue

            try:
                w, h = read_svg_dims(svg)
                scale = TARGET_LONG_SIDE / max(w, h)
                ow = max(1, int(round(w * scale)))
                oh = max(1, int(round(h * scale)))
                mask = svg_to_mask(svg, size=(ow, oh))
            except Exception as e:
                print(f"skip {source_id}: {e}")
                continue

            # Prefer CubiCasa PNG only as optional preview; primary aligned image = SVG render.
            img_path = images_dir / f"{source_id.replace('/', '__')}.png"
            mask_path = masks_dir / f"{source_id.replace('/', '__')}.png"
            try:
                import cairosvg
                from io import BytesIO

                png = cairosvg.svg2png(url=str(svg), output_width=ow, output_height=oh)
                Image.open(BytesIO(png)).convert("RGB").save(img_path)
            except Exception:
                # Fallback: blank canvas sized to mask — converter still valid for masks
                Image.new("RGB", (ow, oh), (255, 255, 255)).save(img_path)

            Image.fromarray(mask).save(mask_path)
            uniq = sorted(int(x) for x in np.unique(mask))
            if not set(uniq).issubset({0, 1, 2, 3}):
                raise RuntimeError(f"Unexpected class IDs in {source_id}: {uniq}")

            rows.append(
                {
                    "image_path": str(img_path.resolve()),
                    "mask_path": str(mask_path.resolve()),
                    "split": split,
                    "source_id": source_id,
                    "width": ow,
                    "height": oh,
                    "unique_ids": ",".join(map(str, uniq)),
                }
            )

    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "mask_path", "split", "source_id", "width", "height", "unique_ids"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Visual QA overlays
    rng = random.Random(seed)
    sample_rows = rows[:]
    rng.shuffle(sample_rows)
    for i, row in enumerate(sample_rows[:overlay_n]):
        img = np.array(Image.open(row["image_path"]).convert("RGB"))
        mask = np.array(Image.open(row["mask_path"]))
        if img.shape[:2] != mask.shape:
            raise RuntimeError(f"Size mismatch {row['source_id']}")
        color = np.array(_colorize(mask))
        blend = (0.55 * img + 0.45 * color).astype(np.uint8)
        Image.fromarray(blend).save(overlays_dir / f"{i:02d}_{row['source_id'].replace('/', '__')}.png")

    print(f"Wrote {len(rows)} rows → {manifest}")
    print(f"QA overlays → {overlays_dir} ({min(overlay_n, len(rows))} files)")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/cubicasa5k"))
    p.add_argument("--out-dir", type=Path, default=Path("data/prepared"))
    p.add_argument("--overlay-n", type=int, default=15)
    args = p.parse_args()
    prepare(args.data_dir, args.out_dir, overlay_n=args.overlay_n)


if __name__ == "__main__":
    main()
