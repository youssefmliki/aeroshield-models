"""Download verified HF ResNet-34 U-Net checkpoint + config."""

from __future__ import annotations

import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "weights"
FILES = {
    "best.safetensors": "https://huggingface.co/Yytsi/floorplan-to-3d-walls/resolve/main/best.safetensors",
    "config.yaml": "https://huggingface.co/Yytsi/floorplan-to-3d-walls/resolve/main/config.yaml",
}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"exists: {dest}")
        return
    print(f"Downloading {url} → {dest}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
        tmp.replace(dest)
    print(f"OK {dest.stat().st_size} bytes")


def main() -> int:
    for name, url in FILES.items():
        download(url, OUT / name)
    # Also write FireTwin-named alias as .pth for integration docs
    try:
        import torch
        from floorplan.checkpoint import export_pth, load_inference_checkpoint

        state, epoch = load_inference_checkpoint(OUT / "best.safetensors", torch.device("cpu"))
        export_pth(state, OUT / "floorplan_best.pth", epoch=epoch)
        print(f"Exported compatible alias: {OUT / 'floorplan_best.pth'}")
    except Exception as e:
        print(f"Note: could not export floorplan_best.pth yet ({e}). safetensors is enough for inference.")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
