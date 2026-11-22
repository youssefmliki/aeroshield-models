"""Letterbox + ImageNet normalize matching HF config.yaml."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .labels import FLOOR_ID, IMAGENET_MEAN, IMAGENET_STD


def letterbox_rgb(
    image: np.ndarray,
    size: tuple[int, int] = (512, 512),
    fill: tuple[int, int, int] = (122, 116, 104),
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Resize RGB HxWx3 into HxW canvas with aspect preserved. Returns image, (left, top, iw, ih)."""
    H, W = size
    h, w = image.shape[:2]
    scale = min(W / w, H / h)
    iw = max(1, int(round(w * scale)))
    ih = max(1, int(round(h * scale)))
    resized = np.array(Image.fromarray(image).resize((iw, ih), Image.BILINEAR))
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:] = fill
    top = (H - ih) // 2
    left = (W - iw) // 2
    canvas[top : top + ih, left : left + iw] = resized
    return canvas, (left, top, iw, ih)


def letterbox_mask(
    mask: np.ndarray,
    size: tuple[int, int],
    content_rect: tuple[int, int, int, int],
) -> np.ndarray:
    H, W = size
    left, top, iw, ih = content_rect
    resized = np.array(
        Image.fromarray(mask.astype(np.uint8), mode="L").resize((iw, ih), Image.NEAREST)
    )
    canvas = np.full((H, W), FLOOR_ID, dtype=np.uint8)
    canvas[top : top + ih, left : left + iw] = resized
    return canvas


def to_model_tensor(image_rgb: np.ndarray, normalize: bool = True) -> torch.Tensor:
    t = torch.from_numpy(image_rgb).permute(2, 0, 1).contiguous().float().div_(255.0)
    if normalize:
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        t = (t - mean) / std
    return t


def load_plan_image(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".svg":
        try:
            import cairosvg
            from io import BytesIO

            png = cairosvg.svg2png(url=str(path))
            return np.array(Image.open(BytesIO(png)).convert("RGB"))
        except Exception as e:
            raise RuntimeError(
                f"SVG rasterize failed ({e}). Install Cairo + cairosvg, or upload PNG/JPG."
            ) from e
    return np.array(Image.open(path).convert("RGB"))
