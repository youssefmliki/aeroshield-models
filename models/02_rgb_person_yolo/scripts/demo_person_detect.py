#!/usr/bin/env python3
"""
Person-detection demo for jury:
  A) Thermal HIT-UAV test images with pretrained/finetuned YOLO — mark SAMPLE if fails
  B) Separate RGB person demo with pretrained YOLOv8n on COCO (REAL pretrained, not thermal)

Never labels detections as confirmed survivors.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
THERMAL_TEST = ROOT / "data" / "thermal" / "test" / "images"
THERMAL_LBL = ROOT / "data" / "thermal" / "test" / "labels"
WEIGHTS = PROJECT / "weights"
OUT = PROJECT / "results" / "person_demo"
DISCLAIMER = "Person POSSIBLE — JAMAIS un survivant confirmé"


def yolo_to_xyxy(box, w, h):
    cx, cy, bw, bh = box
    return [
        int((cx - bw / 2) * w),
        int((cy - bh / 2) * h),
        int((cx + bw / 2) * w),
        int((cy + bh / 2) * h),
    ]


def load_gt(stem: str, w: int, h: int):
    p = THERMAL_LBL / f"{stem}.txt"
    if not p.exists():
        return []
    boxes = []
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            boxes.append(yolo_to_xyxy(list(map(float, parts[1:])), w, h))
    return boxes


def draw(im: Image.Image, boxes_scores, title: str, mode: str):
    im = im.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    color = (0, 255, 100) if mode.startswith("REAL") else (255, 200, 60)
    for box, score in boxes_scores:
        x1, y1, x2, y2 = box
        d.rectangle([x1, y1, x2, y2], outline=color, width=2)
        tag = f"Person POSSIBLE {score:.2f}" if score is not None else "Person POSSIBLE (SAMPLE GT)"
        d.text((x1, max(0, y1 - 12)), tag, fill=color)
    d.rectangle([0, 0, im.width, 36], fill=(0, 0, 0))
    d.text((4, 4), title, fill=(255, 220, 80))
    d.text((4, 20), f"{mode} | {DISCLAIMER}", fill=(255, 160, 80))
    return im


def iou_box(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = max(0, ax2 - ax1) * max(0, ay2 - ay1) + max(0, bx2 - bx1) * max(0, by2 - by1) - inter
    return inter / union if union else 0.0


def recall_at_iou(preds, gts, thr=0.5):
    if not gts:
        return None
    matched = set()
    for box, _ in preds:
        best_i, best = -1, 0.0
        for i, gt in enumerate(gts):
            if i in matched:
                continue
            v = iou_box(box, gt)
            if v > best:
                best, best_i = v, i
        if best >= thr and best_i >= 0:
            matched.add(best_i)
    return len(matched) / len(gts)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "disclaimer": DISCLAIMER,
        "thermal": {},
        "rgb_pretrained_demo": {},
        "performance_claim": False,
    }

    # --- A) Thermal ---
    thermal_imgs = sorted(THERMAL_TEST.glob("*.jpg"))[:5]
    ckpt = WEIGHTS / "thermal_best.pt"
    model_path = str(ckpt) if ckpt.exists() else "yolov8n.pt"
    model = YOLO(model_path)
    thermal_rows = []
    total_preds = 0
    recalls = []
    for img_path in thermal_imgs:
        im = Image.open(img_path).convert("RGB")
        w, h = im.size
        t0 = time.perf_counter()
        res = model.predict(str(img_path), conf=0.15, verbose=False)[0]
        dt = time.perf_counter() - t0
        preds = []
        if res.boxes is not None and len(res.boxes):
            for box, conf, c in zip(
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.conf.cpu().numpy(),
                res.boxes.cls.cpu().numpy(),
            ):
                if int(c) != 0:
                    continue
                preds.append((list(map(int, box.tolist())), float(conf)))
        total_preds += len(preds)
        gts = load_gt(img_path.stem, w, h)
        rec = recall_at_iou(preds, gts)
        if rec is not None:
            recalls.append(rec)
        mode = "REAL_YOLO_THERMAL_ATTEMPT"
        if len(preds) == 0:
            # SAMPLE GT fallback — clearly marked
            mode = "SAMPLE_GT_THERMAL"
            preds = [(b, None) for b in gts]
        vis = draw(im, preds, f"thermal {img_path.stem} {dt:.3f}s", mode)
        name = f"thermal_{img_path.stem}_{mode}.jpg"
        vis.save(OUT / name, quality=95)
        thermal_rows.append(
            {
                "image": img_path.name,
                "mode": mode,
                "n_model_preds": len(preds) if mode.startswith("REAL") else 0,
                "seconds": round(dt, 4),
                "recall_vs_gt": None if rec is None else round(rec, 4),
                "visual": name,
            }
        )

    thermal_failed = total_preds == 0
    report["thermal"] = {
        "model": model_path,
        "failed_adequate_detection": thermal_failed,
        "mean_recall_model": round(float(np.mean(recalls)), 4) if recalls and not thermal_failed else None,
        "rows": thermal_rows,
        "note": (
            "SAMPLE_GT_THERMAL = annotated HIT-UAV boxes shown because YOLO thermal detections were inadequate. "
            "Not a fake AI claim."
            if thermal_failed
            else "Model produced some person boxes on thermal; treat as POSSIBLE only."
        ),
    }

    # --- B) RGB pretrained demo (separate, honest) ---
    rgb_model = YOLO("yolov8n.pt")
    # Use flood SAMPLE RGB or simulation SVG rasterized via PIL — prefer flood images as "aerial RGB stand-in"
    rgb_sources = sorted((ROOT / "data" / "flood" / "test" / "images").glob("*.png"))[:3]
    # Also try a public COCO-like person image via download optional — use thermal RGB-looking? 
    # Better: generate simple RGB scene is weak. Use urllib to fetch a known person stock? 
    # Keep flood RGB + one downloaded commons image if possible.
    rgb_rows = []
    try:
        import urllib.request
        demo_url = "https://ultralytics.com/images/bus.jpg"
        demo_path = OUT / "rgb_bus_demo.jpg"
        if not demo_path.exists():
            urllib.request.urlretrieve(demo_url, demo_path)
        rgb_sources = [demo_path] + rgb_sources
    except Exception as e:
        report["rgb_download_error"] = str(e)

    for img_path in rgb_sources[:4]:
        im = Image.open(img_path).convert("RGB")
        t0 = time.perf_counter()
        res = rgb_model.predict(str(img_path), conf=0.25, verbose=False)[0]
        dt = time.perf_counter() - t0
        preds = []
        if res.boxes is not None and len(res.boxes):
            for box, conf, c in zip(
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.conf.cpu().numpy(),
                res.boxes.cls.cpu().numpy(),
            ):
                if int(c) != 0:  # COCO person
                    continue
                preds.append((list(map(int, box.tolist())), float(conf)))
        mode = "REAL_PRETRAINED_YOLO_RGB_COCO"
        vis = draw(
            im,
            preds,
            f"RGB demo {img_path.name} {dt:.3f}s n={len(preds)}",
            mode,
        )
        name = f"rgb_{img_path.stem}_{mode}.jpg"
        vis.save(OUT / name, quality=95)
        rgb_rows.append(
            {
                "image": img_path.name,
                "mode": mode,
                "n_preds": len(preds),
                "seconds": round(dt, 4),
                "visual": name,
                "note": "Pretrained COCO YOLO on RGB — NOT thermal HIT-UAV performance.",
            }
        )

    report["rgb_pretrained_demo"] = {
        "model": "yolov8n.pt (COCO pretrained)",
        "rows": rgb_rows,
        "note": "Optional separate RGB person demo for jury. Distinct from thermal SAMPLE.",
    }

    out_json = OUT / "person_demo_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"thermal_failed": thermal_failed, "out": str(OUT)}, indent=2))
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
