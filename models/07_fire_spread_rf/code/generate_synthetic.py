"""Generate toy adjacent-room fire-scenario features (NOT physical fire data)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "temp_delta_c",
    "smoke_level",
    "graph_distance",
    "door_open",
    "fuel_load",
    "ventilation",
    "sprinkler_active",
    "occupant_count",
    "time_since_ignition_min",
]

SCHEMA = {
    "features": FEATURE_NAMES,
    "label": "will_be_affected",
    "id_columns": ["scenario_id", "source_room", "candidate_room"],
    "disclaimer": "SYNTHETIC toy rules — not NIST CFAST/FDS, not real fire events.",
}


def _label_row(r: dict) -> int:
    """Transparent toy rule the RF will try to approximate."""
    score = (
        0.35 * r["temp_delta_c"] / 40.0
        + 0.25 * r["smoke_level"]
        + 0.20 * (1.0 / max(r["graph_distance"], 1))
        + 0.15 * r["door_open"] * r["fuel_load"]
        - 0.25 * r["sprinkler_active"]
        - 0.10 * r["ventilation"]
        + 0.05 * min(r["time_since_ignition_min"] / 30.0, 1.0)
    )
    return int(score > 0.45)


def generate(n_scenarios: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    rooms = [f"R{i:02d}" for i in range(1, 13)]
    for s in range(n_scenarios):
        scenario_id = f"S{s:04d}"
        source = rooms[int(rng.integers(0, len(rooms)))]
        n_cand = int(rng.integers(3, 7))
        candidates = rng.choice([r for r in rooms if r != source], size=n_cand, replace=False)
        for cand in candidates:
            row = {
                "scenario_id": scenario_id,
                "source_room": source,
                "candidate_room": cand,
                "temp_delta_c": float(rng.uniform(0, 60)),
                "smoke_level": float(rng.uniform(0, 1)),
                "graph_distance": float(rng.integers(1, 5)),
                "door_open": float(rng.integers(0, 2)),
                "fuel_load": float(rng.uniform(0.1, 1.0)),
                "ventilation": float(rng.uniform(0, 1)),
                "sprinkler_active": float(rng.integers(0, 2)),
                "occupant_count": float(rng.integers(0, 15)),
                "time_since_ignition_min": float(rng.uniform(0, 45)),
            }
            row["will_be_affected"] = _label_row(row)
            rows.append(row)
    return pd.DataFrame(rows)


def split_by_scenario(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    scenarios = np.array(df["scenario_id"].unique(), dtype=object)
    rng.shuffle(scenarios)
    n = len(scenarios)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train_s = set(scenarios[:n_train])
    val_s = set(scenarios[n_train : n_train + n_val])
    test_s = set(scenarios[n_train + n_val :])
    out = df.copy()
    out["split"] = out["scenario_id"].map(
        lambda x: "train" if x in train_s else "val" if x in val_s else "test"
    )
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    df = split_by_scenario(generate())
    df.to_csv(data_dir / "synthetic_scenarios.csv", index=False)
    (data_dir / "feature_schema.json").write_text(json.dumps(SCHEMA, indent=2), encoding="utf-8")
    print(df["split"].value_counts().to_dict())
    print("wrote", data_dir / "synthetic_scenarios.csv")


if __name__ == "__main__":
    main()
