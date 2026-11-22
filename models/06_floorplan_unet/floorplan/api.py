"""Optional FastAPI — POST /predict/floorplan (separate from FireTwin ai/api.py)."""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from .inference import FloorplanSegmenter

MAX_BYTES = 15 * 1024 * 1024
ALLOWED = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".svg"}

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = Path(os.environ.get("FLOORPLAN_WEIGHTS", ROOT / "weights" / "best.safetensors"))
CONFIG = Path(os.environ.get("FLOORPLAN_CONFIG", ROOT / "weights" / "config.yaml"))

app = FastAPI(title="FireTwin FloorPlan Model 5", version="0.1.0")
_segmenter: FloorplanSegmenter | None = None


def get_segmenter() -> FloorplanSegmenter:
    global _segmenter
    if _segmenter is None:
        if not WEIGHTS.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Weights missing: {WEIGHTS}. Run scripts/download_weights.py",
            )
        _segmenter = FloorplanSegmenter(WEIGHTS, CONFIG if CONFIG.exists() else None)
    return _segmenter


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "weights_present": WEIGHTS.exists(),
        "status_contract": "DRAFT_UNVERIFIED",
    }


@app.post("/predict/floorplan")
async def predict_floorplan(file: UploadFile = File(...)):
    suffix = Path(file.filename or "plan.png").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, f"Unsupported type {suffix}. Allowed: {sorted(ALLOWED)}")
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_BYTES} bytes")
    if not raw:
        raise HTTPException(400, "Empty file")

    try:
        if suffix == ".svg":
            import cairosvg

            png = cairosvg.svg2png(bytestring=raw)
            img = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
        else:
            img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception as e:
        raise HTTPException(400, f"Could not decode image: {e}") from e

    try:
        seg = get_segmenter()
        out = seg.predict(img)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}") from e

    buf = io.BytesIO()
    out["overlay"].save(buf, format="PNG")
    mask_buf = io.BytesIO()
    Image.fromarray(out["mask"]).save(mask_buf, format="PNG")

    return JSONResponse(
        {
            "provenance": {
                "model": "resnet34_unet_4class",
                "weights": str(WEIGHTS.name),
                "epoch": out["epoch"],
                "latency_ms": round(out["latency_ms"], 2),
                "filename": file.filename,
            },
            "overlay_png_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
            "mask_png_base64": base64.b64encode(mask_buf.getvalue()).decode("ascii"),
            "draft_geometry": out["draft_geometry"],
            "integration_note": (
                "Main FireTwin app may consume this only after operator approval. "
                "Do not drive evacuation routing from DRAFT_UNVERIFIED geometry."
            ),
        }
    )
