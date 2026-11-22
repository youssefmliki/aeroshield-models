"""Held-out test evaluation — writes metrics only after a real run."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .inference import FloorplanSegmenter, colorize
from .labels import CLASS_NAMES, NUM_CLASSES
from .metrics import confusion_matrix, scores_from_cm
from .preprocess import letterbox_mask


def evaluate(
    manifest: Path,
    weights: Path,
    config: Path,
    out_dir: Path,
    split: str = "test",
    max_examples: int = 6,
    device: str = "auto",
) -> dict:
    rows = [r for r in csv.DictReader(manifest.open(encoding="utf-8")) if r["split"] == split]
    if not rows:
        raise RuntimeError(f"No {split} rows in {manifest}")

    seg = FloorplanSegmenter(weights, config, device=device)
    H, W = seg.image_size
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    latencies: list[float] = []
    examples_dir = out_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for row in tqdm(rows, desc=f"eval {split}"):
        img = np.array(Image.open(row["image_path"]).convert("RGB"))
        gt = np.array(Image.open(row["mask_path"]))
        result = seg.predict(img)
        latencies.append(result["latency_ms"])
        # Align GT to same letterbox canvas as prediction
        from .preprocess import letterbox_rgb

        _, rect = letterbox_rgb(img, size=(H, W))
        gt_lb = letterbox_mask(gt, size=(H, W), content_rect=rect)
        pred = result["mask"]
        cm += confusion_matrix(pred, gt_lb)

        if saved < max_examples:
            panel = Image.new("RGB", (W * 3, H))
            panel.paste(Image.fromarray(letterbox_rgb(img, size=(H, W))[0]), (0, 0))
            panel.paste(colorize(gt_lb), (W, 0))
            panel.paste(result["overlay"], (2 * W, 0))
            panel.save(examples_dir / f"{saved:02d}_{Path(row['source_id']).name}.png")
            saved += 1

    scores = scores_from_cm(cm)
    report = {
        **scores,
        "split": split,
        "n_images": len(rows),
        "latency_ms_mean": float(np.mean(latencies)) if latencies else None,
        "latency_ms_p50": float(np.median(latencies)) if latencies else None,
        "weights": str(weights),
        "notes": {
            "doors": "Inspect false/missed openings in examples; residential CAD ≠ industrial drawings.",
            "domain_gap": "CubiCasa residential; FireTwin industrial plans need operator correction.",
        },
        "status": "COMPUTED",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # confusion figure (simple CSV)
    with (out_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow([""] + list(CLASS_NAMES))
        for i, name in enumerate(CLASS_NAMES):
            wtr.writerow([name] + list(map(int, cm[i])))
    print(json.dumps({k: report[k] for k in ("mean_iou", "per_class_iou", "n_images")}, indent=2))
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("data/prepared/manifest.csv"))
    p.add_argument("--weights", type=Path, default=Path("weights/best.safetensors"))
    p.add_argument("--config", type=Path, default=Path("weights/config.yaml"))
    p.add_argument("--out-dir", type=Path, default=Path("results/floorplan"))
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    evaluate(args.manifest, args.weights, args.config, args.out_dir, split=args.split, device=args.device)


if __name__ == "__main__":
    main()
