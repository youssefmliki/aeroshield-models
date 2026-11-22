"""Equipment detector inference. Missing weights are UNAVAILABLE, never a fake sample."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MODEL_VERSION = "firetwin-equipment-yolov8n-v1"
PACKAGE_DIR = Path(__file__).resolve().parent
WEIGHTS = PACKAGE_DIR / "equipment_best.pt"
CLASS_MAP = PACKAGE_DIR / "class_mapping.json"

_model = None
_load_error = None


def _load():
    global _model, _load_error
    if _model is not None or _load_error is not None:
        return _model
    if not WEIGHTS.exists():
        _load_error = "UNAVAILABLE"
        return None
    try:
        from ultralytics import YOLO

        _model = YOLO(str(WEIGHTS))
    except Exception as exc:  # pragma: no cover - environment failure
        _load_error = f"UNAVAILABLE: {exc}"
        return None
    return _model


def status() -> dict:
    model = _load()
    names = {}
    if CLASS_MAP.exists():
        raw = json.loads(CLASS_MAP.read_text(encoding="utf-8"))
        names = {str(k): v for k, v in raw.items()} if isinstance(raw, dict) else {str(i): n for i, n in enumerate(raw)}
    return {
        "model_version": MODEL_VERSION,
        "status": "REAL" if model is not None else "UNAVAILABLE",
        "weights": str(WEIGHTS) if WEIGHTS.exists() else None,
        "classes": names,
    }


def predict_equipment(image_bgr: np.ndarray, confidence: float = 0.25) -> dict:
    """Return class, pixel xyxy, confidence. Provenance is REAL only when weights loaded."""
    model = _load()
    if model is None:
        return {
            "source": "UNAVAILABLE",
            "model_version": MODEL_VERSION,
            "detections": [],
            "detail": _load_error or "UNAVAILABLE",
        }
    results = model.predict(source=image_bgr, conf=confidence, verbose=False)
    detections = []
    names = model.names
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(
                {
                    "class": names.get(cls_id, str(cls_id)),
                    "class_id": cls_id,
                    "confidence": float(box.conf[0]),
                    "xyxy": xyxy,
                }
            )
    return {
        "source": "REAL",
        "model_version": MODEL_VERSION,
        "detections": detections,
        "note": "Class is not an inventory asset id. Match a cabinet to ELEC-CAB-01 only with a trusted camera location or operator review.",
    }
