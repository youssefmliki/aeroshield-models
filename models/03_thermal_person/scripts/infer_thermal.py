#!/usr/bin/env python3
"""
AeroShield 360 — Étape 09 : tester M3 (personnes thermiques) avec YOLO.

- Essaie poids YOLO (yolov8n) puis affine quelques epochs sur HIT-UAV subset si utile
- 5 prédictions + latence + scores
- precision/rappel seulement si labels test valides
- Ne JAMAIS appeler une détection « survivant confirmé » → « Person POSSIBLE »
- Si aucun poids adéquat : image annotée SAMPLE (GT), pas une fausse IA
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
THERMAL = ROOT / "data" / "thermal"
RESULTS = PROJECT / "results" / "thermal"
WEIGHTS = PROJECT / "weights"
METRICS_PATH = PROJECT / "metrics_thermal.json"
YAML = THERMAL / "thermal.yaml"
DISCLAIMER = "Person POSSIBLE (thermique) — JAMAIS un survivant confirmé"


def yolo_to_xyxy(box: list[float], w: int, h: int) -> tuple[int, int, int, int]:
    cx, cy, bw, bh = box
    x1 = int(round((cx - bw / 2) * w))
    y1 = int(round((cy - bh / 2) * h))
    x2 = int(round((cx + bw / 2) * w))
    y2 = int(round((cy + bh / 2) * h))
    return x1, y1, x2, y2


def load_gt_boxes(label_path: Path, w: int, h: int) -> list[list[int]]:
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        box = list(map(float, parts[1:]))
        boxes.append(list(yolo_to_xyxy(box, w, h)))
    return boxes


def iou_box(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter) / float(union) if union else 0.0


def match_precision_recall(
    preds: list[tuple[list[int], float]], gts: list[list[int]], iou_thr: float = 0.5
) -> tuple[float | None, float | None]:
    if not gts and not preds:
        return 1.0, 1.0
    if not gts:
        return (0.0 if preds else 1.0), None
    # greedy match
    matched_gt = set()
    tp = 0
    for box, _score in sorted(preds, key=lambda x: -x[1]):
        best_i, best_iou = -1, 0.0
        for i, gt in enumerate(gts):
            if i in matched_gt:
                continue
            v = iou_box(box, gt)
            if v > best_iou:
                best_iou, best_i = v, i
        if best_iou >= iou_thr and best_i >= 0:
            matched_gt.add(best_i)
            tp += 1
    fp = len(preds) - tp
    fn = len(gts) - tp
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return precision, recall


def draw_predictions(
    image: Image.Image,
    preds: list[tuple[list[int], float]],
    title: str,
    mode: str,
) -> Image.Image:
    im = image.convert("RGB").copy()
    draw = ImageDraw.Draw(im)
    color = (0, 255, 100) if mode != "SAMPLE_GT" else (255, 200, 60)
    for box, score in preds:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"Person POSSIBLE {score:.2f}" if mode != "SAMPLE_GT" else "Person POSSIBLE (SAMPLE GT)"
        draw.text((x1, max(0, y1 - 12)), label, fill=color)
    draw.rectangle([0, 0, im.width, 36], fill=(0, 0, 0))
    draw.text((4, 4), title, fill=(255, 220, 80))
    draw.text((4, 20), DISCLAIMER, fill=(255, 160, 80))
    return im


def collect_images(n: int = 5) -> list[Path]:
    paths: list[Path] = []
    for split in ("test", "val", "train"):
        d = THERMAL / split / "images"
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*")):
            paths.append(p)
            if len(paths) >= n:
                return paths
    return paths


def write_data_yaml() -> Path:
    # Ultralytics needs absolute path
    content = f"""path: {THERMAL.as_posix()}
train: train/images
val: val/images
test: test/images
names:
  0: person
nc: 1
"""
    YAML.write_text(content, encoding="utf-8")
    return YAML


def run_yolo_on_images(model, images: list[Path], conf: float = 0.15):
    rows = []
    for img_path in images:
        im = Image.open(img_path).convert("RGB")
        w, h = im.size
        t0 = time.perf_counter()
        results = model.predict(str(img_path), conf=conf, verbose=False)
        dt = time.perf_counter() - t0
        preds: list[tuple[list[int], float]] = []
        r0 = results[0]
        if r0.boxes is not None and len(r0.boxes):
            xyxy = r0.boxes.xyxy.cpu().numpy()
            scores = r0.boxes.conf.cpu().numpy()
            cls = r0.boxes.cls.cpu().numpy()
            for box, score, c in zip(xyxy, scores, cls):
                # After fine-tune: class 0 = person. Baseline COCO: class 0 = person too.
                if int(c) != 0:
                    continue
                x1, y1, x2, y2 = map(int, box.tolist())
                preds.append(([x1, y1, x2, y2], float(score)))

        split = img_path.parent.parent.name
        gt_path = THERMAL / split / "labels" / f"{img_path.stem}.txt"
        gts = load_gt_boxes(gt_path, w, h)
        precision = recall = None
        if gts is not None and gt_path.exists():
            precision, recall = match_precision_recall(preds, gts)

        rows.append(
            {
                "image": img_path.name,
                "stem": img_path.stem,
                "split": split,
                "seconds": round(dt, 4),
                "n_preds": len(preds),
                "scores": [round(s, 4) for _, s in preds],
                "boxes": [b for b, _ in preds],
                "precision": None if precision is None else round(precision, 4),
                "recall": None if recall is None else round(recall, 4),
                "has_gt": bool(gts),
                "pil": im,
                "preds": preds,
            }
        )
    return rows


def save_row_visual(row: dict, out_dir: Path, mode: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    title = f"{mode} · {row['stem']} · {row['seconds']}s · n={row['n_preds']}"
    vis = draw_predictions(row["pil"], row["preds"], title, mode)
    name = f"{row['stem']}_{mode}.jpg"
    vis.save(out_dir / name, quality=95)
    return name


def sample_gt_fallback(images: list[Path]) -> list[dict]:
    """Annotated SAMPLE from ground-truth labels — NOT fake AI."""
    rows = []
    for img_path in images:
        im = Image.open(img_path).convert("RGB")
        w, h = im.size
        split = img_path.parent.parent.name
        gt_path = THERMAL / split / "labels" / f"{img_path.stem}.txt"
        gts = load_gt_boxes(gt_path, w, h)
        preds = [(b, 1.0) for b in gts]
        rows.append(
            {
                "image": img_path.name,
                "stem": img_path.stem,
                "split": split,
                "seconds": 0.0,
                "n_preds": len(preds),
                "scores": [],  # pas de score modèle
                "boxes": gts,
                # P/R modèle non applicables — ce sont des boîtes GT, pas une IA
                "precision": None,
                "recall": None,
                "has_gt": bool(gts),
                "pil": im,
                "preds": preds,
                "sample_note": "Boîtes GT SAMPLE — pas une inférence IA",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8, help="fine-tune epochs if baseline weak")
    parser.add_argument("--no-finetune", action="store_true")
    parser.add_argument("--force-sample", action="store_true")
    parser.add_argument("--conf", type=float, default=0.15)
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    images = collect_images(5)
    if len(images) < 5:
        print(f"[ERR] besoin de 5 images, trouvé {len(images)}. Lancez prepare_thermal.py")
        return 1

    report: dict = {
        "disclaimer": DISCLAIMER,
        "performance_claim": False,
        "dataset": "HIT-UAV subset (CC-BY-4.0) via ai/data/thermal",
        "images": [],
        "summary": {},
    }

    mode = "SAMPLE_GT"
    model_path = None
    finetuned = False

    if args.force_sample:
        rows = sample_gt_fallback(images)
        mode = "SAMPLE_GT"
        report["honest_failure"] = True
        report["failure_reason"] = "Mode SAMPLE forcé — annotations GT affichées, pas une fausse IA."
    else:
        try:
            from ultralytics import YOLO

            write_data_yaml()
            print("Baseline YOLO yolov8n.pt (COCO) sur images thermiques…")
            baseline = YOLO("yolov8n.pt")
            rows = run_yolo_on_images(baseline, images, conf=args.conf)
            mean_det = float(np.mean([r["n_preds"] for r in rows]))
            mean_rec = np.mean([r["recall"] for r in rows if r["recall"] is not None])
            print(f"baseline mean_dets={mean_det:.2f} mean_recall={mean_rec:.3f}")

            weak = mean_det < 0.4 or (not np.isnan(mean_rec) and mean_rec < 0.15)
            if weak and not args.no_finetune:
                print(f"Point de départ faible → fine-tune {args.epochs} epochs sur HIT-UAV subset…")
                # Train on train/val; never use held-out test images as train
                train_res = baseline.train(
                    data=str(YAML),
                    epochs=args.epochs,
                    imgsz=640,
                    batch=4,
                    device=0 if __import__("torch").cuda.is_available() else "cpu",
                    project=str(WEIGHTS / "thermal_runs"),
                    name="yolo_person",
                    exist_ok=True,
                    verbose=False,
                    patience=5,
                )
                best = Path(train_res.save_dir) / "weights" / "best.pt"
                if not best.exists():
                    best = WEIGHTS / "thermal_runs" / "yolo_person" / "weights" / "best.pt"
                if best.exists():
                    finetuned = True
                    model_path = str(best)
                    # copy stable path
                    stable = WEIGHTS / "thermal_best.pt"
                    stable.write_bytes(best.read_bytes())
                    model_path = str(stable)
                    model = YOLO(model_path)
                    # evaluate on never-seen: prefer test split images
                    test_imgs = sorted((THERMAL / "test" / "images").glob("*"))[:5]
                    if len(test_imgs) < 5:
                        test_imgs = images
                    rows = run_yolo_on_images(model, test_imgs, conf=args.conf)
                    mode = "YOLO_FINETUNED_HITUAV"
                    print("Fine-tune OK — prédictions sur sous-ensemble non vu.")
                else:
                    print("[WARN] best.pt introuvable après train")
                    mode = "YOLO_BASELINE_WEAK"
            else:
                mode = "YOLO_BASELINE"
                model_path = "yolov8n.pt"

            # If still almost no detections → SAMPLE GT fallback (not fake AI)
            if float(np.mean([r["n_preds"] for r in rows])) < 0.2:
                print("[HONEST] Aucun poids thermique adéquat / détections nulles → SAMPLE GT, pas de fausse IA")
                rows = sample_gt_fallback(images)
                mode = "SAMPLE_GT"
                report["honest_failure"] = True
                report["failure_reason"] = (
                    "Poids YOLO inadaptés au thermique ou détections insuffisantes. "
                    "Affichage des boîtes annotées SAMPLE/GT — pas une fausse IA."
                )
        except Exception as e:
            print(f"[WARN] YOLO indisponible/échec: {e}")
            rows = sample_gt_fallback(images)
            mode = "SAMPLE_GT"
            report["honest_failure"] = True
            report["failure_reason"] = f"Échec YOLO: {e}. SAMPLE GT affiché."

    # Save visuals + strip PIL for JSON
    out_rows = []
    for row in rows:
        fname = save_row_visual(row, RESULTS, mode)
        out = {k: v for k, v in row.items() if k not in ("pil", "preds")}
        out["visual"] = fname
        out["label"] = DISCLAIMER
        out_rows.append(out)

    precisions = [r["precision"] for r in out_rows if r.get("precision") is not None]
    recalls = [r["recall"] for r in out_rows if r.get("recall") is not None]
    latencies = [r["seconds"] for r in out_rows]

    report.update(
        {
            "mode": mode,
            "model_path": model_path,
            "finetuned": finetuned,
            "images": out_rows,
            "summary": {
                "n_images": len(out_rows),
                "mean_seconds": round(float(np.mean(latencies)), 4) if latencies else None,
                "mean_precision": round(float(np.mean(precisions)), 4) if precisions else None,
                "mean_recall": round(float(np.mean(recalls)), 4) if recalls else None,
                "metrics_valid": bool(precisions or recalls),
                "crash": False,
            },
        }
    )

    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (RESULTS / "metrics_thermal.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== infer_thermal done ===")
    print(f"mode           : {mode}")
    print(f"n              : {len(out_rows)}")
    print(f"mean latency   : {report['summary']['mean_seconds']} s")
    print(f"mean precision : {report['summary']['mean_precision']}")
    print(f"mean recall    : {report['summary']['mean_recall']}")
    print(f"results        : {RESULTS}")
    print(f"metrics        : {METRICS_PATH}")
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
