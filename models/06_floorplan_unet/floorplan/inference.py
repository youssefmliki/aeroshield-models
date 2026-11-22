"""Run 4-class floorplan segmentation with verified preprocessing."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from .checkpoint import load_inference_checkpoint
from .geometry import mask_to_draft_geometry
from .labels import CLASS_COLORS, CLASS_NAMES, FLOOR_ID
from .model import build_model
from .preprocess import letterbox_rgb, load_plan_image, to_model_tensor


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def colorize(mask: np.ndarray) -> Image.Image:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for idx, name in enumerate(CLASS_NAMES):
        rgb[mask == idx] = CLASS_COLORS[name]
    return Image.fromarray(rgb)


class FloorplanSegmenter:
    def __init__(
        self,
        weights: Path,
        config: Path | None = None,
        device: str = "auto",
    ) -> None:
        self.device = resolve_device(device)
        cfg: dict = {}
        if config and Path(config).exists():
            with open(config, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        self.cfg = cfg
        self.image_size = tuple(cfg.get("data", {}).get("image_size", [512, 512]))  # H, W
        self.normalize = cfg.get("data", {}).get("normalize", True)
        encoder = cfg.get("model", {}).get("encoder_name", "resnet34")

        self.model = build_model(encoder_name=encoder, encoder_weights=None).to(self.device)
        state, self.epoch = load_inference_checkpoint(Path(weights), self.device)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(
                f"Incompatible checkpoint keys (unexpected={list(unexpected)[:5]}…). "
                "Use HF best.safetensors or a matching floorplan_best.pth — not CubiCasa .pkl."
            )
        if missing:
            raise RuntimeError(f"Checkpoint missing keys: {list(missing)[:8]}")
        self.model.eval()

    @torch.no_grad()
    def predict(self, image_rgb: np.ndarray) -> dict:
        H, W = self.image_size
        canvas, rect = letterbox_rgb(image_rgb, size=(H, W))
        tensor = to_model_tensor(canvas, normalize=self.normalize).unsqueeze(0).to(self.device)
        t0 = time.perf_counter()
        logits = self.model(tensor)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        probs = torch.softmax(logits, dim=1)[0]
        conf = probs.max(dim=0).values.cpu().numpy()
        mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        left, top, iw, ih = rect
        if (ih, iw) != mask.shape:
            cleaned = np.full_like(mask, FLOOR_ID)
            cleaned[top : top + ih, left : left + iw] = mask[top : top + ih, left : left + iw]
            mask = cleaned
        draft = mask_to_draft_geometry(mask, conf_map=conf, content_rect=rect)
        return {
            "mask": mask,
            "confidence": conf,
            "overlay": colorize(mask),
            "draft_geometry": draft,
            "latency_ms": latency_ms,
            "content_rect": rect,
            "epoch": self.epoch,
        }

    def predict_path(self, path: Path) -> dict:
        img = load_plan_image(Path(path))
        return self.predict(img)


def main() -> None:
    p = argparse.ArgumentParser(description="Floorplan 4-class inference")
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--weights", default=Path("weights/best.safetensors"), type=Path)
    p.add_argument("--config", default=Path("weights/config.yaml"), type=Path)
    p.add_argument("--out-dir", default=Path("results/floorplan/examples"), type=Path)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    seg = FloorplanSegmenter(args.weights, args.config, device=args.device)
    out = seg.predict_path(args.image)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    Image.fromarray(load_plan_image(args.image)).save(args.out_dir / f"{stem}_input.png")
    out["overlay"].save(args.out_dir / f"{stem}_overlay.png")
    Image.fromarray(out["mask"]).save(args.out_dir / f"{stem}_mask.png")
    (args.out_dir / f"{stem}_draft.json").write_text(
        json.dumps(out["draft_geometry"], indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "latency_ms": round(out["latency_ms"], 2),
                "epoch": out["epoch"],
                "status": out["draft_geometry"]["status"],
                "polygon_counts": {k: len(v) for k, v in out["draft_geometry"]["polygons"].items()},
                "out_dir": str(args.out_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
