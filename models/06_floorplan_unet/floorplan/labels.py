"""4-class taxonomy matching Yytsi/floorplan-to-3d (checkpoint-compatible)."""

from __future__ import annotations

from collections import OrderedDict

CLASS_NAMES: tuple[str, ...] = ("floor", "wall", "door", "window")
CLASS_TO_ID: dict[str, int] = {n: i for i, n in enumerate(CLASS_NAMES)}
NUM_CLASSES: int = len(CLASS_NAMES)
FLOOR_ID: int = CLASS_TO_ID["floor"]

TOKEN_TO_CLASS: OrderedDict[str, str] = OrderedDict(
    [
        ("Window", "window"),
        ("Door", "door"),
        ("Wall", "wall"),
    ]
)

PAINT_ORDER: tuple[str, ...] = ("wall", "door", "window")

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "floor": (240, 240, 235),
    "wall": (40, 40, 45),
    "door": (230, 120, 50),
    "window": (60, 150, 220),
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
