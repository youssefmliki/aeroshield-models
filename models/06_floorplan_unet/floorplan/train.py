"""Optional fine-tune on prepared CubiCasa masks. Only run after data prep."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .checkpoint import export_pth, load_inference_checkpoint
from .labels import NUM_CLASSES
from .metrics import confusion_matrix, scores_from_cm
from .model import build_model
from .preprocess import letterbox_mask, letterbox_rgb, to_model_tensor


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ManifestDataset(Dataset):
    def __init__(self, rows: list[dict], size: tuple[int, int] = (512, 512), augment: bool = False):
        self.rows = rows
        self.size = size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img = np.array(Image.open(row["image_path"]).convert("RGB"))
        mask = np.array(Image.open(row["mask_path"]))
        canvas, rect = letterbox_rgb(img, size=self.size)
        mask_lb = letterbox_mask(mask, size=self.size, content_rect=rect)
        if self.augment and random.random() < 0.5:
            canvas = np.ascontiguousarray(canvas[:, ::-1])
            mask_lb = np.ascontiguousarray(mask_lb[:, ::-1])
        return to_model_tensor(canvas), torch.from_numpy(mask_lb.astype(np.int64))


def run_train(
    manifest: Path,
    out_dir: Path,
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-4,
    seed: int = 42,
    init_weights: Path | None = None,
    device: str = "auto",
) -> Path:
    set_seed(seed)
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    if not train_rows or not val_rows:
        raise RuntimeError("Need train and val in manifest")

    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu")
    model = build_model(encoder_weights="imagenet").to(dev)
    if init_weights and Path(init_weights).exists():
        state, _ = load_inference_checkpoint(Path(init_weights), dev)
        model.load_state_dict(state, strict=True)

    train_loader = DataLoader(ManifestDataset(train_rows, augment=True), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ManifestDataset(val_rows, augment=False), batch_size=batch_size, shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    out_dir.mkdir(parents=True, exist_ok=True)
    curves: list[dict] = []
    best_miou = -1.0
    best_path = out_dir / "floorplan_best.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for images, masks in tqdm(train_loader, desc=f"train {epoch}"):
            images, masks = images.to(dev), masks.to(dev)
            logits = model(images)
            loss = F.cross_entropy(logits, masks)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        model.eval()
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(dev)
                pred = model(images).argmax(1).cpu().numpy()
                gt = masks.numpy()
                for p, g in zip(pred, gt):
                    cm += confusion_matrix(p, g)
        scores = scores_from_cm(cm)
        miou = scores["mean_iou"]
        curves.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_miou": miou})
        print(f"epoch {epoch}: loss={np.mean(losses):.4f} val_mIoU={miou:.4f}")
        if miou > best_miou:
            best_miou = miou
            export_pth(model.state_dict(), best_path, epoch=epoch)

    import json

    (out_dir / "training_curves.json").write_text(json.dumps(curves, indent=2), encoding="utf-8")
    print(f"Best mIoU={best_miou:.4f} → {best_path}")
    return best_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("data/prepared/manifest.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("weights"))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init-weights", type=Path, default=Path("weights/best.safetensors"))
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    run_train(
        args.manifest,
        args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        init_weights=args.init_weights,
        device=args.device,
    )


if __name__ == "__main__":
    main()
