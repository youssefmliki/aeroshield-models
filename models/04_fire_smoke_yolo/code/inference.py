"""Standalone inference helper used by delivery package and local tests."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# HuggingFace SalahALHaismawi/yolov26-fire-detection (verified from checkpoint names)
# 0=fire, 1=other, 2=smoke
DEFAULT_CLASS_NAMES = {0: "fire", 1: "other", 2: "smoke"}
_DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "fire_smoke_best.pt"
_model = None


def load_model(weights: str | Path | None = None):
    global _model
    from ultralytics import YOLO
    import torch

    path = Path(weights) if weights else _DEFAULT_WEIGHTS
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    _model = YOLO(str(path))
    return _model


def _names() -> dict[int, str]:
    if _model is not None and getattr(_model, "names", None):
        raw = _model.names
        if isinstance(raw, dict):
            return {int(k): str(v) for k, v in raw.items()}
        return {i: str(n) for i, n in enumerate(raw)}
    return DEFAULT_CLASS_NAMES


def predict_fire_smoke(
    image: str | Path | np.ndarray,
    confidence_threshold: float = 0.25,
    weights: str | Path | None = None,
) -> dict[str, Any]:
    """
    Run fire/smoke detection.

    Returns structured detections with pixel-space xyxy bboxes.
    """
    global _model
    if _model is None or weights is not None:
        load_model(weights)

    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise ValueError(f"Could not read image: {image}")
    else:
        img = image
        if img is None or not hasattr(img, "shape"):
            raise ValueError("Invalid image array")

    import torch

    device = 0 if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    results = _model.predict(img, conf=confidence_threshold, device=device, verbose=False)
    latency = (time.perf_counter() - t0) * 1000
    r0 = results[0]
    names = _names()

    detections = []
    counts = {"fire": 0, "smoke": 0, "other": 0}
    if r0.boxes is not None and len(r0.boxes):
        for xyxy, cls, score in zip(
            r0.boxes.xyxy.cpu().numpy(),
            r0.boxes.cls.cpu().numpy().astype(int),
            r0.boxes.conf.cpu().numpy(),
        ):
            name = names.get(int(cls), str(int(cls)))
            detections.append(
                {
                    "class_id": int(cls),
                    "class_name": name,
                    "confidence": float(score),
                    "bbox": [float(v) for v in xyxy.tolist()],
                }
            )
            if name in counts:
                counts[name] += 1

    return {
        "source": "REAL",
        "model_version": "fire_smoke_best",
        "model_origin": "huggingface:SalahALHaismawi/yolov26-fire-detection",
        "detections": detections,
        "counts": {"fire": counts["fire"], "smoke": counts["smoke"]},
        "counts_full": counts,
        "latency_ms": float(latency),
        "image_width": int(img.shape[1]),
        "image_height": int(img.shape[0]),
        "bbox_format": "xyxy_pixels",
        "class_mapping": names,
        "disclaimer": "Experimental visual detection; not a certified fire alarm.",
    }


def annotate_image(image: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    out = image.copy()
    colors = {
        "fire": (0, 0, 255),
        "smoke": (255, 180, 0),
        "other": (0, 200, 100),
    }
    for d in result.get("detections", []):
        x1, y1, x2, y2 = map(int, d["bbox"])
        color = colors.get(d["class_name"], (0, 255, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"{d['class_name']} {d['confidence']:.2f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return out


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--weights", default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    res = predict_fire_smoke(args.image, args.conf, args.weights)
    print(json.dumps(res, indent=2))
    if args.out:
        img = cv2.imread(args.image)
        cv2.imwrite(args.out, annotate_image(img, res))
