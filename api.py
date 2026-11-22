#!/usr/bin/env python3
"""
AeroShield 360 — Étape 12 : API FastAPI M2/M3.

POST /predict/flood  — image → source, model_version, class, score, regions
POST /predict/thermal — idem (Person POSSIBLE)

Si poids absents / erreur : répond SAMPLE clairement (pas de fausse IA silencieuse).
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
WEIGHTS = PROJECT / "weights"

app = FastAPI(title="AeroShield 360 AI API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FLOOD_CKPT = WEIGHTS / "flood_best.pt"
THERMAL_CKPT = WEIGHTS / "thermal_best.pt"
RGB_PERSON_CKPT = WEIGHTS / "rgb_person_best.pt"
FIRE_SMOKE_CKPT = WEIGHTS / "fire_smoke_best.pt"

_flood_model = None
_flood_processor = None
_thermal_model = None
_rgb_person_model = None
_device = "cpu"


class Region(BaseModel):
    type: str = "bbox"  # bbox | polygon
    coords: list[float] = Field(description="bbox [x1,y1,x2,y2] normalized 0-1 or polygon pts")
    score: float | None = None
    label: str | None = None


class PredictResponse(BaseModel):
    source: str  # REAL | SAMPLE
    model_version: str
    class_name: str = Field(alias="class")
    score: float
    regions: list[Region]
    latency_ms: float
    disclaimer: str
    note: str | None = None

    model_config = {"populate_by_name": True}


def _load_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _simulated_water_regions(img: Image.Image) -> tuple[float, list[Region]]:
    arr = np.asarray(img, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    water = (b > 100) & (b > r + 15) & (b > g + 5)
    ratio = float(water.mean())
    ys, xs = np.where(water)
    regions: list[Region] = []
    if len(xs) > 0:
        h, w = water.shape
        x1, x2 = xs.min() / w, xs.max() / w
        y1, y2 = ys.min() / h, ys.max() / h
        regions.append(
            Region(type="bbox", coords=[x1, y1, x2, y2], score=ratio, label="water_SIMULATED")
        )
    return ratio, regions


def try_load_flood() -> bool:
    global _flood_model, _flood_processor, _device
    if _flood_model is not None:
        return True
    if not FLOOD_CKPT.exists():
        return False
    try:
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _flood_processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
        ckpt = torch.load(FLOOD_CKPT, map_location=_device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.to(_device)
        model.eval()
        _flood_model = model
        return True
    except Exception as e:
        print(f"[api] flood model load failed: {e}")
        return False


def try_load_thermal() -> bool:
    global _thermal_model
    if _thermal_model is not None:
        return True
    path = THERMAL_CKPT if THERMAL_CKPT.exists() else None
    try:
        from ultralytics import YOLO

        _thermal_model = YOLO(str(path) if path else "yolov8n.pt")
        return True
    except Exception as e:
        print(f"[api] thermal model load failed: {e}")
        return False


def try_load_rgb_person() -> bool:
    global _rgb_person_model
    if _rgb_person_model is not None:
        return True
    path = RGB_PERSON_CKPT if RGB_PERSON_CKPT.exists() else Path("yolov8n.pt")
    try:
        from ultralytics import YOLO

        _rgb_person_model = YOLO(str(path))
        return True
    except Exception as e:
        print(f"[api] rgb person model load failed: {e}")
        return False


def predict_flood_real(img: Image.Image) -> PredictResponse | None:
    if not try_load_flood():
        return None
    import torch
    import torch.nn.functional as F

    t0 = time.perf_counter()
    inputs = _flood_processor(images=img, return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = _flood_model(**inputs).logits
        up = F.interpolate(logits, size=img.size[::-1], mode="bilinear", align_corners=False)
        pred = up.argmax(dim=1)[0].cpu().numpy()
    water = pred == 1
    score = float(water.mean())
    regions: list[Region] = []
    ys, xs = np.where(water)
    if len(xs):
        h, w = water.shape
        regions.append(
            Region(
                type="bbox",
                coords=[xs.min() / w, ys.min() / h, xs.max() / w, ys.max() / h],
                score=score,
                label="water",
            )
        )
    ms = (time.perf_counter() - t0) * 1000
    ckpt_label = "REAL_FLOODNET"
    try:
        import torch

        meta = torch.load(FLOOD_CKPT, map_location="cpu", weights_only=False)
        ckpt_label = str(meta.get("label", ckpt_label))
    except Exception:
        pass
    note = (
        "REAL FloodNet-Supervised fine-tune (subset, official splits). "
        "Binary water+flooded classes. Not a full-dataset SOTA claim."
        if "FLOODNET" in ckpt_label.upper() or "REAL" in ckpt_label.upper()
        else "REAL weights loaded; check metrics_flood.json for dataset label."
    )
    return PredictResponse(
        source="REAL",
        model_version=f"flood_best.pt@segformer-b0-{ckpt_label}",
        class_name="water",
        score=score,
        regions=regions,
        latency_ms=round(ms, 2),
        disclaimer="Inférence SegFormer locale — prototype, pas un système de secours opérationnel.",
        note=note,
    )


def predict_flood_sample(img: Image.Image) -> PredictResponse:
    t0 = time.perf_counter()
    score, regions = _simulated_water_regions(img)
    ms = (time.perf_counter() - t0) * 1000
    return PredictResponse(
        source="SAMPLE",
        model_version="heuristic-blue-v1",
        class_name="water",
        score=score,
        regions=regions,
        latency_ms=round(ms, 2),
        disclaimer="EXEMPLE SIMULÉ — pas une IA de crue opérationnelle.",
        note="Fallback SAMPLE (poids absents ou erreur).",
    )


def predict_thermal_real(img: Image.Image) -> PredictResponse | None:
    if not try_load_thermal():
        return None
    t0 = time.perf_counter()
    # save temp for ultralytics
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img.save(f.name, quality=95)
        results = _thermal_model.predict(f.name, conf=0.15, verbose=False)
    ms = (time.perf_counter() - t0) * 1000
    r0 = results[0]
    regions: list[Region] = []
    scores: list[float] = []
    w, h = img.size
    if r0.boxes is not None and len(r0.boxes):
        xyxy = r0.boxes.xyxy.cpu().numpy()
        confs = r0.boxes.conf.cpu().numpy()
        clss = r0.boxes.cls.cpu().numpy()
        for box, conf, c in zip(xyxy, confs, clss):
            if int(c) != 0:
                continue
            x1, y1, x2, y2 = box.tolist()
            regions.append(
                Region(
                    type="bbox",
                    coords=[x1 / w, y1 / h, x2 / w, y2 / h],
                    score=float(conf),
                    label="Person POSSIBLE",
                )
            )
            scores.append(float(conf))
    if not regions:
        return None  # force SAMPLE rather than empty REAL pretending success
    return PredictResponse(
        source="REAL",
        model_version=str(THERMAL_CKPT.name if THERMAL_CKPT.exists() else "yolov8n.pt"),
        class_name="person_possible",
        score=float(np.mean(scores)),
        regions=regions,
        latency_ms=round(ms, 2),
        disclaimer="Person POSSIBLE — JAMAIS un survivant confirmé.",
        note="REAL = YOLO chargé; fiabilité thermique limitée.",
    )


def predict_thermal_sample(img: Image.Image) -> PredictResponse:
    """No fake detections — return empty SAMPLE with honest note (or center stub)."""
    t0 = time.perf_counter()
    ms = (time.perf_counter() - t0) * 1000
    return PredictResponse(
        source="SAMPLE",
        model_version="none-thermal-adequate",
        class_name="person_possible",
        score=0.0,
        regions=[],
        latency_ms=round(ms, 2),
        disclaimer="EXEMPLE SIMULÉ — aucune fausse détection IA. Voir results/thermal/*_SAMPLE_GT.jpg",
        note="Pas de poids thermique adéquat; utiliser annotations SAMPLE GT côté UI.",
    )


def predict_rgb_person_real(img: Image.Image) -> PredictResponse | None:
    """RGB aerial person detection (VisDrone-finetuned or COCO). NOT thermal."""
    if not try_load_rgb_person():
        return None
    import tempfile

    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img.save(f.name, quality=95)
        results = _rgb_person_model.predict(f.name, conf=0.25, verbose=False)
    ms = (time.perf_counter() - t0) * 1000
    r0 = results[0]
    regions: list[Region] = []
    scores: list[float] = []
    w, h = img.size
    if r0.boxes is not None and len(r0.boxes):
        for box, conf, c in zip(
            r0.boxes.xyxy.cpu().numpy(),
            r0.boxes.conf.cpu().numpy(),
            r0.boxes.cls.cpu().numpy(),
        ):
            if int(c) != 0:
                continue
            x1, y1, x2, y2 = box.tolist()
            regions.append(
                Region(
                    type="bbox",
                    coords=[x1 / w, y1 / h, x2 / w, y2 / h],
                    score=float(conf),
                    label="Person POSSIBLE (RGB)",
                )
            )
            scores.append(float(conf))
    source = "REAL" if RGB_PERSON_CKPT.exists() else "REAL"
    model_version = (
        "rgb_person_best.pt@visdrone-person"
        if RGB_PERSON_CKPT.exists()
        else "yolov8n.pt@coco-person"
    )
    return PredictResponse(
        source=source,
        model_version=model_version,
        class_name="person_possible_rgb",
        score=float(np.mean(scores)) if scores else 0.0,
        regions=regions,
        latency_ms=round(ms, 2),
        disclaimer="Person POSSIBLE (RGB) — JAMAIS un survivant confirmé. NOT a thermal sensor.",
        note="RGB camera workflow only — distinct from /predict/thermal.",
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "flood_checkpoint": FLOOD_CKPT.exists(),
        "flood_label": "check metrics_flood.json",
        "thermal_checkpoint": THERMAL_CKPT.exists(),
        "rgb_person_checkpoint": RGB_PERSON_CKPT.exists(),
        "fire_smoke_checkpoint": FIRE_SMOKE_CKPT.exists(),
        "fire_smoke_status": "READY" if FIRE_SMOKE_CKPT.exists() else "UNAVAILABLE_AWAITING_TEAM_WEIGHTS",
        "label": "AeroShield FireTwin / 360 AI API",
    }


@app.post("/predict/flood")
async def predict_flood(file: UploadFile = File(...)) -> dict[str, Any]:
    data = await file.read()
    img = _load_image(data)
    real = predict_flood_real(img)
    result = real if real is not None else predict_flood_sample(img)
    return result.model_dump(by_alias=True)


@app.post("/predict/thermal")
async def predict_thermal(file: UploadFile = File(...)) -> dict[str, Any]:
    """Thermal path is SAMPLE-only for jury honesty (HIT-UAV model not validated)."""
    data = await file.read()
    img = _load_image(data)
    # Never present thermal as a functioning trained REAL model.
    result = predict_thermal_sample(img)
    result.note = (
        (result.note or "")
        + " | Jury: thermal = SAMPLE (use /predict/rgb-person for REAL RGB detection)."
    ).strip(" |")
    return result.model_dump(by_alias=True)


@app.post("/predict/rgb-person")
async def predict_rgb_person(file: UploadFile = File(...)) -> dict[str, Any]:
    """RGB person detection for EAGLE/RGB camera — never presented as thermal."""
    data = await file.read()
    img = _load_image(data)
    result = predict_rgb_person_real(img)
    if result is None:
        result = PredictResponse(
            source="SAMPLE",
            model_version="none-rgb-person",
            class_name="person_possible_rgb",
            score=0.0,
            regions=[],
            latency_ms=0.0,
            disclaimer="RGB person model unavailable — SAMPLE empty response.",
            note="NOT thermal. VisDrone aerial — not validated for indoor smoke.",
        )
    return result.model_dump(by_alias=True)


@app.post("/predict/fire-smoke")
async def predict_fire_smoke(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Fire/smoke YOLO endpoint.

    Until teammates provide weights/fire_smoke_best.pt this returns source=UNAVAILABLE
    with empty detections — never fabricated REAL boxes.
    """
    t0 = time.perf_counter()
    data = await file.read()
    if len(data) > 15_000_000:
        return {
            "source": "UNAVAILABLE",
            "model_version": "none",
            "class": "fire_smoke",
            "score": 0.0,
            "regions": [],
            "detections": [],
            "latency_ms": 0.0,
            "disclaimer": "Image too large (>15MB).",
            "note": "FireTwin stub",
        }
    try:
        _ = _load_image(data)
    except Exception as exc:  # noqa: BLE001
        return {
            "source": "UNAVAILABLE",
            "model_version": "none",
            "class": "fire_smoke",
            "score": 0.0,
            "regions": [],
            "detections": [],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "disclaimer": f"Invalid image: {exc}",
            "note": "FireTwin stub",
        }

    ms = round((time.perf_counter() - t0) * 1000, 2)

    if not FIRE_SMOKE_CKPT.exists():
        return {
            "source": "UNAVAILABLE",
            "model_version": "awaiting-weights/fire_smoke_best.pt",
            "class": "fire_smoke",
            "score": 0.0,
            "regions": [],
            "detections": [],
            "latency_ms": ms,
            "disclaimer": (
                "Fire/smoke model not installed. Place weights/fire_smoke_best.pt "
                "from team training to enable REAL inference. "
                "NOT certified fire detection."
            ),
            "note": "Stub — teammates train YOLO; no simulated detections as REAL.",
        }

    # Weight present: attempt ultralytics YOLO (optional dependency)
    try:
        from ultralytics import YOLO  # type: ignore

        model = YOLO(str(FIRE_SMOKE_CKPT))
        results = model.predict(source=Image.open(io.BytesIO(data)).convert("RGB"), verbose=False)
        detections: list[dict[str, Any]] = []
        regions: list[dict[str, Any]] = []
        scores: list[float] = []
        r0 = results[0] if results else None
        if r0 is not None and r0.boxes is not None:
            names = r0.names or {}
            wh = r0.orig_shape  # h, w
            h, w = float(wh[0]), float(wh[1])
            for box in r0.boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = str(names.get(cls_id, cls_id))
                scores.append(conf)
                det = {
                    "class": label,
                    "confidence": conf,
                    "bbox_xyxy": xyxy,
                    "bbox_norm": [xyxy[0] / w, xyxy[1] / h, xyxy[2] / w, xyxy[3] / h],
                }
                detections.append(det)
                regions.append(
                    {
                        "type": "bbox",
                        "coords": det["bbox_norm"],
                        "score": conf,
                        "label": label,
                    }
                )
        return {
            "source": "REAL",
            "model_version": f"fire_smoke_best.pt@{FIRE_SMOKE_CKPT.stat().st_mtime_ns}",
            "class": "fire_smoke",
            "score": float(sum(scores) / len(scores)) if scores else 0.0,
            "regions": regions,
            "detections": detections,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "disclaimer": (
                "Experimental fire/smoke detection — NOT certified. "
                "Operator must verify. Hackathon prototype."
            ),
            "note": "YOLOv8n fire/smoke — team-trained weights",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "source": "UNAVAILABLE",
            "model_version": "fire_smoke_best.pt-load-failed",
            "class": "fire_smoke",
            "score": 0.0,
            "regions": [],
            "detections": [],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "disclaimer": f"Weight present but inference failed: {exc}",
            "note": "Check ultralytics install / checkpoint format",
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
