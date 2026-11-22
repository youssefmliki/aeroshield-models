"""Inference for synthetic adjacent-room scenario risk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "outputs" / "fire_spread_scenario_rf.joblib"
DEFAULT_SCHEMA = ROOT / "data" / "feature_schema.json"


def predict_scenario_next_room_risk(
    features: dict[str, float],
    model_path: Path | None = None,
    schema_path: Path | None = None,
) -> dict[str, Any]:
    schema_path = schema_path or DEFAULT_SCHEMA
    model_path = model_path or DEFAULT_MODEL
    if not model_path.exists():
        return {
            "source": "UNAVAILABLE",
            "error": f"Missing model {model_path}. Run code/train.py first.",
            "label": None,
            "scenario_risk_score": None,
        }
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    names = schema["features"]
    missing = [n for n in names if n not in features]
    if missing:
        return {
            "source": "SYNTHETIC_MODEL",
            "error": f"Missing features: {missing}",
            "label": None,
            "scenario_risk_score": None,
        }
    x = np.array([[float(features[n]) for n in names]], dtype=np.float64)
    clf = joblib.load(model_path)
    proba = float(clf.predict_proba(x)[0, 1])
    label = int(proba >= 0.5)
    return {
        "source": "SYNTHETIC_MODEL",
        "label": label,
        "label_name": "will_be_affected" if label == 1 else "not_affected",
        "scenario_risk_score": proba,
        "disclaimer": "SIMULATION / NOT PHYSICALLY VALIDATED — toy rules only.",
    }


if __name__ == "__main__":
    demo = {
        "temp_delta_c": 35,
        "smoke_level": 0.8,
        "graph_distance": 1,
        "door_open": 1,
        "fuel_load": 0.9,
        "ventilation": 0.2,
        "sprinkler_active": 0,
        "occupant_count": 4,
        "time_since_ignition_min": 12,
    }
    print(json.dumps(predict_scenario_next_room_risk(demo), indent=2))
