"""Per-class IoU, Dice, confusion — computed only from real preds/GTs."""

from __future__ import annotations

from typing import Any

import numpy as np

from .labels import CLASS_NAMES, NUM_CLASSES


def confusion_matrix(pred: np.ndarray, gt: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    pred = pred.ravel().astype(np.int64)
    gt = gt.ravel().astype(np.int64)
    mask = (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    pred, gt = pred[mask], gt[mask]
    cm = np.bincount(gt * num_classes + pred, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def scores_from_cm(cm: np.ndarray) -> dict[str, Any]:
    iou: dict[str, float] = {}
    dice: dict[str, float] = {}
    for i, name in enumerate(CLASS_NAMES):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - tp)
        fn = float(cm[i, :].sum() - tp)
        denom_iou = tp + fp + fn
        denom_dice = 2 * tp + fp + fn
        iou[name] = (tp / denom_iou) if denom_iou > 0 else float("nan")
        dice[name] = (2 * tp / denom_dice) if denom_dice > 0 else float("nan")
    valid_iou = [v for v in iou.values() if v == v]
    return {
        "per_class_iou": iou,
        "per_class_dice": dice,
        "mean_iou": float(np.mean(valid_iou)) if valid_iou else float("nan"),
        "confusion_matrix": cm.tolist(),
        "class_names": list(CLASS_NAMES),
    }
