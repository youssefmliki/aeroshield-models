"""Train paired ResNet-18 damage classifier — refuses to invent metrics without real pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.csv")
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output", type=Path, default=ROOT / "results")
    args = p.parse_args()

    metrics_path = ROOT / "metrics" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.manifest.exists() or args.manifest.name.endswith("template.csv"):
        report = {
            "status": "DATASET_MISSING",
            "message": (
                "No real indoor before/after manifest found. "
                "Copy dataset_manifest_template.csv → manifest.csv and add licensed image pairs. "
                "Do not train on xBD satellite imagery for this indoor use case."
            ),
            "architecture": "ResNet-18 shared encoder + concat(|diff|) binary head",
            "metrics": None,
        }
        metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 2

    # Real training path (only when user supplies data)
    import pandas as pd
    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
    from sklearn.metrics import classification_report, f1_score, accuracy_score
    import numpy as np

    df = pd.read_csv(args.manifest)
    required = {"before_path", "after_path", "label", "split", "facility_id", "incident_id", "asset_id"}
    if not required.issubset(df.columns):
        raise SystemExit(f"Manifest missing columns: {required - set(df.columns)}")

    label_map = {"no_visible_change": 0, "visible_change": 1}
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    class PairDS(Dataset):
        def __init__(self, frame):
            self.frame = frame.reset_index(drop=True)

        def __len__(self):
            return len(self.frame)

        def __getitem__(self, i):
            r = self.frame.iloc[i]
            b = Image.open(args.data_root / r["before_path"]).convert("RGB")
            a = Image.open(args.data_root / r["after_path"]).convert("RGB")
            y = label_map[str(r["label"])]
            return tfm(b), tfm(a), y

    class PairNet(nn.Module):
        def __init__(self):
            super().__init__()
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.encoder = nn.Sequential(*list(backbone.children())[:-1])
            self.head = nn.Sequential(nn.Linear(512 * 3, 128), nn.ReLU(), nn.Linear(128, 2))

        def forward(self, b, a):
            fb = self.encoder(b).flatten(1)
            fa = self.encoder(a).flatten(1)
            x = torch.cat([fb, fa, (fb - fa).abs()], dim=1)
            return self.head(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PairNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    if train_df.empty or test_df.empty:
        raise SystemExit("Need non-empty train and test splits")

    train_loader = DataLoader(PairDS(train_df), batch_size=args.batch_size, shuffle=True)
    args.output.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for b, a, y in train_loader:
            b, a, y = b.to(device), a.to(device), y.to(device)
            logits = model(b, a)
            loss = nn.functional.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print(f"epoch {epoch} loss={np.mean(losses):.4f}")

    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for b, a, y in DataLoader(PairDS(test_df), batch_size=args.batch_size):
            logits = model(b.to(device), a.to(device))
            pred = logits.argmax(1).cpu().numpy()
            ys.extend(y.numpy().tolist())
            ps.extend(pred.tolist())

    ckpt = args.output / "damage_best.pt"
    torch.save({"model": model.state_dict(), "label_map": label_map}, ckpt)
    report = {
        "status": "COMPUTED",
        "n_test": len(ys),
        "accuracy": float(accuracy_score(ys, ps)),
        "macro_f1": float(f1_score(ys, ps, average="macro")),
        "classification_report": classification_report(
            ys, ps, target_names=["no_visible_change", "visible_change"], output_dict=True
        ),
        "checkpoint": str(ckpt),
        "note": "Visible change ≠ fire causation or structural safety.",
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
