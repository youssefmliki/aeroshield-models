"""Train Random Forest on synthetic scenario data — honest synthetic metrics only."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from generate_synthetic import FEATURE_NAMES, generate, split_by_scenario

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    data_dir = ROOT / "data"
    out_dir = ROOT / "outputs"
    metrics_dir = ROOT / "metrics"
    for d in (data_dir, out_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    df = split_by_scenario(generate())
    df.to_csv(data_dir / "synthetic_scenarios.csv", index=False)

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    X_train, y_train = train[FEATURE_NAMES], train["will_be_affected"]
    X_test, y_test = test[FEATURE_NAMES], test["will_be_affected"]

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "status": "COMPUTED_SYNTHETIC_ONLY",
        "disclaimer": "Scores measure agreement with toy labeling rules — NOT real fire spread skill.",
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "n_train_scenarios": int(train["scenario_id"].nunique()),
        "n_test_scenarios": int(test["scenario_id"].nunique()),
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "classification_report": classification_report(y_test, pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "feature_importance": {
            n: float(v) for n, v in zip(FEATURE_NAMES, clf.feature_importances_)
        },
        "source": "SYNTHETIC_MODEL",
    }
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (data_dir / "feature_schema.json").write_text(
        json.dumps(
            {
                "features": FEATURE_NAMES,
                "label": "will_be_affected",
                "disclaimer": metrics["disclaimer"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    joblib.dump(clf, out_dir / "fire_spread_scenario_rf.joblib")

    # Confusion matrix plot
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["not_affected", "affected"])
    ax.set_yticklabels(["not_affected", "affected"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Toy label")
    ax.set_title("Synthetic test confusion")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(metrics_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    # Feature importance
    imp = sorted(metrics["feature_importance"].items(), key=lambda x: -x[1])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh([k for k, _ in imp][::-1], [v for _, v in imp][::-1], color="#c45c26")
    ax.set_title("Feature importance (synthetic RF)")
    fig.tight_layout()
    fig.savefig(metrics_dir / "feature_importance.png", dpi=120)
    plt.close(fig)

    (out_dir / "MODEL_REPORT.md").write_text(
        f"""# Fire spread scenario RF — run report

- source: SYNTHETIC_MODEL
- test accuracy: {metrics['accuracy']:.3f}
- macro F1: {metrics['macro_f1']:.3f}
- test scenarios: {metrics['n_test_scenarios']}
- disclaimer: {metrics['disclaimer']}
""",
        encoding="utf-8",
    )
    print(json.dumps({k: metrics[k] for k in ("accuracy", "macro_f1", "n_test_rows", "source")}, indent=2))


if __name__ == "__main__":
    main()
