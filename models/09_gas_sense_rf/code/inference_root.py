"""
Reusable inference interface for AeroShield FireTwin GasSense AI.

Usage from another backend:

    from inference import predict_gas
    result = predict_gas([0.1, 0.2, ...])  # exactly 128 floats

Relative paths resolve from this project root (FireTwin-GasSense-AI/).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent


def _default_model_path() -> Path:
    candidates = [
        PROJECT_ROOT / "weights" / "gas_classifier.joblib",
        PROJECT_ROOT / "gas_classifier.joblib",
        PROJECT_ROOT / "delivery" / "FireTwin-GasSense-Model" / "gas_classifier.joblib",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


_BUNDLE: dict[str, Any] | None = None
_MODEL_PATH: Path | None = None


class GasInferenceError(ValueError):
    """Raised when input validation fails or the model cannot run."""


def _load_bundle(model_path: Path | None = None) -> dict[str, Any]:
    global _BUNDLE, _MODEL_PATH
    path = Path(model_path) if model_path else _default_model_path()
    if _BUNDLE is not None and _MODEL_PATH == path:
        return _BUNDLE
    if not path.exists():
        raise GasInferenceError(
            f"Model artifact missing at {path}. Train the model or place gas_classifier.joblib."
        )
    _BUNDLE = joblib.load(path)
    _MODEL_PATH = path
    return _BUNDLE


def reload_model(model_path: str | Path | None = None) -> None:
    """Force reload of the joblib bundle (useful in tests)."""
    global _BUNDLE, _MODEL_PATH
    _BUNDLE = None
    _MODEL_PATH = None
    _load_bundle(Path(model_path) if model_path else None)


def validate_features(features: list[float] | np.ndarray, n_features: int) -> np.ndarray:
    if features is None:
        raise GasInferenceError("features is required")

    if isinstance(features, str):
        raise GasInferenceError("features must be a list of numbers, not a string")

    try:
        arr = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GasInferenceError(
            f"features must be numerical; invalid values present ({exc})"
        ) from exc

    if arr.ndim != 1:
        raise GasInferenceError(f"features must be a 1-D vector, got shape {arr.shape}")

    if arr.size != n_features:
        raise GasInferenceError(
            f"Incorrect feature count: expected {n_features}, got {arr.size}. "
            "Input must match the UCI 16-sensor array feature schema."
        )

    if not np.isfinite(arr).all():
        bad = np.where(~np.isfinite(arr))[0][:10].tolist()
        raise GasInferenceError(
            f"features contain NaN or infinity at indices {bad}. Inference refused."
        )

    return arr


def predict_gas(
    features: list[float],
    *,
    source: str = "UCI_SENSOR_SAMPLE",
    model_path: str | Path | None = None,
    uncertainty_threshold: float | None = None,
) -> dict[str, Any]:
    """
    Validate features, run the stored preprocessing + Random Forest pipeline,
    and return gas label, probabilities, uncertainty, and provenance.
    """
    bundle = _load_bundle(Path(model_path) if model_path else None)
    pipe = bundle["pipeline"]
    class_mapping = {int(k): v for k, v in bundle["class_mapping"].items()}
    n_features = int(bundle.get("n_features") or len(bundle["feature_names"]))
    threshold = (
        float(uncertainty_threshold)
        if uncertainty_threshold is not None
        else float(bundle.get("uncertainty_threshold", 0.45))
    )

    x = validate_features(features, n_features)

    t0 = time.perf_counter()
    proba = pipe.predict_proba(x.reshape(1, -1))[0]
    pred_id = int(pipe.predict(x.reshape(1, -1))[0])
    latency_ms = (time.perf_counter() - t0) * 1000.0

    classes_ = [int(c) for c in pipe.named_steps["clf"].classes_]
    class_probabilities = {
        class_mapping[cid]: float(proba[idx]) for idx, cid in enumerate(classes_)
    }
    # Ensure all known classes present (zeros if somehow missing)
    for cid, name in class_mapping.items():
        class_probabilities.setdefault(name, 0.0)

    predicted_gas = class_mapping[pred_id]
    confidence = float(max(class_probabilities.values()))
    uncertain = confidence < threshold
    decision_state = "UNCERTAIN" if uncertain else predicted_gas

    return {
        "source": source if source else "REAL_MODEL_ON_UCI_SAMPLE",
        "model_version": bundle.get("model_version", "gas_classifier_v1"),
        "predicted_gas": predicted_gas,
        "decision_state": decision_state,
        "class_probabilities": class_probabilities,
        "confidence": confidence,
        "uncertain": uncertain,
        "latency_ms": round(latency_ms, 4),
        "provenance": {
            "dataset": bundle.get("dataset", {}).get("name", "UCI Gas Sensor Array Drift Dataset"),
            "training_batches": bundle.get("training_batches"),
            "test_batch_held_out": bundle.get("test_batch"),
            "algorithm": bundle.get("algorithm"),
            "n_features": n_features,
            "preprocessing": bundle.get("preprocessing"),
            "model_path": str(_MODEL_PATH),
        },
        "disclaimer": bundle.get(
            "disclaimer",
            "Experimental classification of compatible sensor measurements; "
            "not a certified gas leak detector.",
        ),
    }


if __name__ == "__main__":
    # Smoke test with a processed demo sample if available
    demo_path = PROJECT_ROOT / "data" / "processed" / "demo_samples.json"
    if demo_path.exists():
        import json

        samples = json.loads(demo_path.read_text(encoding="utf-8"))
        sample = samples[0]
        out = predict_gas(sample["features"], source="UCI_DEMO_SAMPLE")
        print(json.dumps(out, indent=2))
        print("true_gas:", sample["true_gas"])
    else:
        print("No demo samples found. Train/prepare data first.")
