"""Independent FastAPI route for the equipment detector. Not wired into the main app."""
from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile

from inference import predict_equipment, status

app = FastAPI(title="FireTwin Equipment Detector", version="1")


@app.get("/health")
def health():
    info = status()
    return {"ok": info["status"] == "REAL", **info}


@app.post("/predict/equipment")
async def predict_equipment_route(file: UploadFile = File(...), confidence: float = 0.25):
    import cv2
    import numpy as np

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="not an image")
    result = predict_equipment(image, confidence=confidence)
    if result["source"] != "REAL":
        raise HTTPException(status_code=503, detail=result)
    return result
