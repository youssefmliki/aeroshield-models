"""Load .safetensors (HF) or training .pt / .pth checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch


def load_inference_checkpoint(
    path: Path, device: torch.device
) -> tuple[dict[str, torch.Tensor], int | None]:
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file

        target = "cuda" if device.type == "cuda" else "cpu"
        state = load_file(str(path), device=target)
        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata() or {}
        epoch = int(meta["epoch"]) if meta.get("epoch") else None
        return state, epoch

    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"], ckpt.get("epoch")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"], ckpt.get("epoch")
    if isinstance(ckpt, dict):
        # bare state_dict
        return ckpt, None
    raise ValueError(f"Unrecognized checkpoint format: {path}")


def export_pth(state: dict[str, torch.Tensor], out_path: Path, epoch: int | None = None) -> None:
    """Write FireTwin-friendly floorplan_best.pth (model key)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state, "epoch": epoch, "num_classes": 4, "encoder": "resnet34"}, out_path)
