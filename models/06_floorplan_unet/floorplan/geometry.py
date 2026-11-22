"""Mask → draft wall/door/window polygons for operator review."""

from __future__ import annotations

from typing import Any, TypedDict

import cv2
import numpy as np

from .labels import CLASS_TO_ID

EXTRACT_CLASSES: tuple[str, ...] = ("wall", "door", "window")
CLOSING_KERNEL_PX = 3
APPROX_EPSILON_PX = 1.5
MIN_POLYGON_AREA_PX = 30.0


class Polygon(TypedDict):
    outer: list[list[float]]
    holes: list[list[list[float]]]
    confidence: float


def _approx(contour: np.ndarray) -> list[list[float]]:
    simplified = cv2.approxPolyDP(contour, APPROX_EPSILON_PX, closed=True)
    return simplified.reshape(-1, 2).astype(float).tolist()


def _polygons_for_class(mask: np.ndarray, class_id: int, conf_map: np.ndarray | None) -> list[Polygon]:
    binary = (mask == class_id).astype(np.uint8)
    if binary.sum() == 0:
        return []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSING_KERNEL_PX, CLOSING_KERNEL_PX))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    holes_by_parent: dict[int, list[int]] = {}
    for i, (_, _, _, parent) in enumerate(hierarchy):
        if parent != -1:
            holes_by_parent.setdefault(parent, []).append(i)

    polygons: list[Polygon] = []
    for i, (_, _, _, parent) in enumerate(hierarchy):
        if parent != -1:
            continue
        if cv2.contourArea(contours[i]) < MIN_POLYGON_AREA_PX:
            continue
        outer = _approx(contours[i])
        if len(outer) < 3:
            continue
        holes: list[list[list[float]]] = []
        for j in holes_by_parent.get(i, []):
            if cv2.contourArea(contours[j]) < MIN_POLYGON_AREA_PX:
                continue
            ring = _approx(contours[j])
            if len(ring) >= 3:
                holes.append(ring)
        conf = 1.0
        if conf_map is not None:
            ys = [p[1] for p in outer]
            xs = [p[0] for p in outer]
            y0, y1 = max(0, int(min(ys))), min(mask.shape[0], int(max(ys)) + 1)
            x0, x1 = max(0, int(min(xs))), min(mask.shape[1], int(max(xs)) + 1)
            region = conf_map[y0:y1, x0:x1][binary[y0:y1, x0:x1].astype(bool)]
            if region.size:
                conf = float(region.mean())
        polygons.append({"outer": outer, "holes": holes, "confidence": conf})
    return polygons


def mask_to_draft_geometry(
    mask: np.ndarray,
    conf_map: np.ndarray | None = None,
    content_rect: tuple[int, int, int, int] | None = None,
    scale_m_per_px: float | None = None,
) -> dict[str, Any]:
    """Export draft primitives with provenance. status always DRAFT_UNVERIFIED."""
    H, W = mask.shape
    polys = {
        cls: _polygons_for_class(mask, CLASS_TO_ID[cls], conf_map) for cls in EXTRACT_CLASSES
    }
    return {
        "status": "DRAFT_UNVERIFIED",
        "approved": False,
        "provenance": "floorplan_model5_resnet34_unet",
        "canvas_size": [W, H],
        "content_rect": list(content_rect) if content_rect else [0, 0, W, H],
        "scale_m_per_px": scale_m_per_px,
        "classes": list(EXTRACT_CLASSES),
        "polygons": polys,
        "notes": (
            "Operator must calibrate scale, edit walls/doors/windows, then approve "
            "before 2D/3D twin generation. Not an evacuation map."
        ),
    }
