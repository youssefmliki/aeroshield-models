"""Drop-in helper for AeroShield FastAPI /predict/fire-smoke."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from inference import predict_fire_smoke as _predict

WEIGHTS = Path(__file__).resolve().parent / "fire_smoke_best.pt"


def predict_fire_smoke(image, confidence_threshold: float = 0.25) -> dict[str, Any]:
    """Return structured detections (xyxy pixels) using the delivered checkpoint."""
    return _predict(image, confidence_threshold=confidence_threshold, weights=WEIGHTS)


# Example FastAPI wiring (pseudo):
# from api_integration import predict_fire_smoke
# @app.post("/predict/fire-smoke")
# async def endpoint(file: UploadFile):
#     data = await file.read()
#     import cv2, numpy as np
#     img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
#     return predict_fire_smoke(img)
