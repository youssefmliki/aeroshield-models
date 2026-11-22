"""Predict visible_change vs no_visible_change for a before/after pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]


class PairNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(nn.Linear(512 * 3, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, b, a):
        fb = self.encoder(b).flatten(1)
        fa = self.encoder(a).flatten(1)
        x = torch.cat([fb, fa, (fb - fa).abs()], dim=1)
        return self.head(x)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=ROOT / "results" / "damage_best.pt")
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "metrics" / "prediction.json")
    p.add_argument("--threshold", type=float, default=0.65)
    args = p.parse_args()

    if not args.checkpoint.exists():
        payload = {
            "source": "MODEL_UNAVAILABLE",
            "status": "DATASET_MISSING_OR_NOT_TRAINED",
            "message": "Train with real paired images first, or compare manually in the Twin UI.",
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 2

    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = PairNet().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    inv = {v: k for k, v in ckpt["label_map"].items()}
    b = tfm(Image.open(args.before).convert("RGB")).unsqueeze(0).to(device)
    a = tfm(Image.open(args.after).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.softmax(model(b, a), dim=1)[0]
        conf, pred = float(prob.max()), int(prob.argmax())
    label = inv[pred]
    payload = {
        "source": "REAL_MODEL",
        "label": label,
        "confidence": conf,
        "review_required": conf < args.threshold,
        "note": "PENDING observation until human review. Not a structural safety verdict.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
