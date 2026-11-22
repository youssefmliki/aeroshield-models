"""
Example: merge GasSense /predict/gas into an existing AeroShield FastAPI backend.

Copy inference.py + weights/gas_classifier.joblib into the AeroShield backend
(or add FireTwin-GasSense-AI to PYTHONPATH), then include the router below.
"""

from __future__ import annotations

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference import GasInferenceError, predict_gas

router = APIRouter(prefix="/gassense", tags=["GasSense"])


class GasPredictRequest(BaseModel):
    features: list[float] = Field(..., min_length=1)
    source: str = "AEROSHIELD_BACKEND"


class FactorySimulationAlert(BaseModel):
    """Simulated factory dashboard payload — NOT a live industrial sensor."""

    room: str = "Chemical Storage"
    alarm_source: str = "SIMULATION"
    sensor_status: str = "Warning"
    features: list[float]
    sample_attribution: str = "UCI Gas Sensor Array Drift Dataset sample"


@router.post("/predict/gas")
def aeroshield_predict_gas(payload: GasPredictRequest):
    try:
        return predict_gas(payload.features, source=payload.source)
    except GasInferenceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/factory/chemical-storage-simulation")
def chemical_storage_simulation(payload: FactorySimulationAlert):
    """
    Scenario: Chemical storage room (SIMULATION ONLY).

    The dashboard may display room/alarm metadata alongside an experimental
    ML classification of a separately selected UCI sample.

    This does NOT:
    - read MQ-series / ESP32 sensors
    - claim a real leak detection
    - estimate concentration
    - open valves, cut power, or issue evacuation commands
    """
    try:
        ml = predict_gas(payload.features, source="FACTORY_SIMULATION_UCI_SAMPLE")
    except GasInferenceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "scenario": "Chemical storage room",
        "room": payload.room,
        "alarm_source": payload.alarm_source,
        "sensor_status": payload.sensor_status,
        "sample_attribution": payload.sample_attribution,
        "ml_experimental_classification": ml,
        "suggested_action": (
            "Verification by qualified safety personnel required. "
            "Do not automate safety actuators from this experimental classifier alone."
        ),
        "compatibility_note": (
            "Model expects UCI 16-sensor array feature vectors (128 dims). "
            "Not compatible with generic MQ-series ESP32 readings."
        ),
    }


def attach_to_app(app: FastAPI) -> None:
    """Call from AeroShield main: attach_to_app(app)"""
    app.include_router(router)


if __name__ == "__main__":
    # Standalone smoke demo of the integration router
    app = FastAPI(title="AeroShield + GasSense Integration Example")
    attach_to_app(app)
    print("Routers:", [r.path for r in app.routes])
