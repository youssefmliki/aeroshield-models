#!/usr/bin/env python3
"""
AeroShield 360 — Étape 08 : préparer M3 (personnes thermiques).

Modes:
  --source hituav  : télécharge un petit sous-ensemble HIT-UAV (CC-BY-4.0) + boîtes Person
  --source sample  : exemples locaux illustratifs (si HIT-UAV inaccessible)

Sortie: ai/data/thermal/{train,val,test}/{images,labels} + previews
Labels YOLO: class 0 = Person (normalisé cx cy w h). Conversion seulement pour Person.
Ne lance PAS d'entraînement ici. Annotations incompatibles → refus + SAMPLE.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "thermal"
RAW = ROOT / "data" / "hituav_raw"
PREVIEWS = OUT / "previews"
GH_RAW = "https://raw.githubusercontent.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset/main"
PERSON_ID = 0
USER_AGENT = "AeroShield360-Hackathon/1.0 (research demo)"


def ensure_dirs() -> None:
    for split in ("train", "val", "test"):
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)


def clear_outputs() -> None:
    if OUT.exists():
        for split in ("train", "val", "test"):
            for sub in ("images", "labels"):
                d = OUT / split / sub
                if d.exists():
                    shutil.rmtree(d)
        if PREVIEWS.exists():
            shutil.rmtree(PREVIEWS)


def http_get(url: str, dest: Path | None = None, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    if dest is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return data


def load_coco_person_index(split: str) -> dict[str, list[list[float]]]:
    """
    Parse HIT-UAV normal_json annotations.
    Returns filename -> list of COCO bboxes [x,y,w,h] for Person only.
    """
    # Prefer local raw, else GitHub
    local = RAW / "normal_json" / "annotations" / f"{split}.json"
    if local.exists():
        data = json.loads(local.read_text(encoding="utf-8"))
    else:
        url = f"{GH_RAW}/normal_json/annotations/{split}.json"
        print(f"Downloading annotations {split}.json …")
        data = json.loads(http_get(url).decode("utf-8"))

    images = {im["id"]: im for im in data.get("images", [])}
    # HIT-UAV uses singular key "annotation"
    anns = data.get("annotation") or data.get("annotations") or []
    if not anns:
        raise ValueError(f"{split}: no annotation array (incompatible format)")

    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    if PERSON_ID not in cats and not any(n.lower() == "person" for n in cats.values()):
        raise ValueError(f"{split}: Person class not found in categories={cats}")

    person_cat = PERSON_ID
    for cid, name in cats.items():
        if name.lower() == "person":
            person_cat = cid
            break

    by_file: dict[str, list[list[float]]] = {}
    for ann in anns:
        if int(ann.get("category_id", -1)) != int(person_cat):
            continue
        im = images.get(ann["image_id"])
        if not im:
            continue
        bbox = ann.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x, y, w, h = map(float, bbox[:4])
        if w <= 1 or h <= 1:
            continue
        fn = im["file_name"] if "file_name" in im else im.get("filename")
        if not fn:
            continue
        by_file.setdefault(fn, []).append([x, y, w, h])
    return by_file


def coco_to_yolo(bbox: list[float], img_w: int, img_h: int) -> list[float] | None:
    x, y, w, h = bbox
    # clamp to image
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    nw = bw / img_w
    nh = bh / img_h
    # valid YOLO ranges
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < nw <= 1 and 0 < nh <= 1):
        return None
    return [cx, cy, nw, nh]


def yolo_to_xyxy(box: list[float], img_w: int, img_h: int) -> tuple[int, int, int, int]:
    cx, cy, w, h = box
    bw, bh = w * img_w, h * img_h
    x1 = int(round(cx * img_w - bw / 2))
    y1 = int(round(cy * img_h - bh / 2))
    x2 = int(round(cx * img_w + bw / 2))
    y2 = int(round(cy * img_h + bh / 2))
    return x1, y1, x2, y2


def download_hituav_subset(max_images: int, seed: int) -> list[dict]:
    """Download images that have ≥1 Person box from val+test (train listing often huge)."""
    rng = random.Random(seed)
    collected: list[dict] = []
    for split in ("val", "test"):
        try:
            index = load_coco_person_index(split)
        except Exception as e:
            print(f"[WARN] annotations {split}: {e}", file=sys.stderr)
            continue
        files = [f for f, boxes in index.items() if boxes]
        rng.shuffle(files)
        for fn in files:
            if len(collected) >= max_images:
                break
            url = f"{GH_RAW}/normal_json/{split}/{fn}"
            dest = RAW / "images" / split / fn
            try:
                if not dest.exists():
                    print(f"  get {split}/{fn}")
                    http_get(url, dest, timeout=90)
                img = Image.open(dest).convert("RGB")
                w, h = img.size
                yolo_boxes = []
                for bb in index[fn]:
                    yb = coco_to_yolo(bb, w, h)
                    if yb:
                        yolo_boxes.append(yb)
                if not yolo_boxes:
                    continue
                collected.append(
                    {
                        "stem": Path(fn).stem,
                        "path": dest,
                        "boxes": yolo_boxes,
                        "source_split": split,
                        "label": "HIT-UAV",
                    }
                )
            except Exception as e:
                print(f"[WARN] skip {fn}: {e}", file=sys.stderr)
                continue
        if len(collected) >= max_images:
            break
    return collected


def make_sample_thermal(idx: int, size: int = 320, seed: int = 0) -> tuple[Image.Image, list[list[float]]]:
    """Illustrative thermal-like image + aligned Person boxes (SAMPLE)."""
    rng = random.Random(seed + idx)
    img = Image.new("RGB", (size, size), (20, 20, 24))
    px = img.load()
    # noisy background
    for y in range(size):
        for x in range(size):
            v = 18 + (x * 3 + y * 2 + rng.randint(0, 12)) % 40
            px[x, y] = (v, v, int(v * 0.9))

    draw = ImageDraw.Draw(img)
    boxes: list[list[float]] = []
    n_persons = rng.randint(1, 3)
    for _ in range(n_persons):
        pw = rng.randint(10, 22)
        ph = rng.randint(22, 40)
        x1 = rng.randint(20, size - pw - 20)
        y1 = rng.randint(20, size - ph - 20)
        x2, y2 = x1 + pw, y1 + ph
        # bright "thermal" blob
        draw.ellipse([x1, y1, x2, y2], fill=(220, 120, 40))
        draw.ellipse([x1 + 2, y1 + 2, x1 + pw // 2, y1 + ph // 3], fill=(250, 200, 80))
        cx = ((x1 + x2) / 2) / size
        cy = ((y1 + y2) / 2) / size
        boxes.append([cx, cy, pw / size, ph / size])

    draw.rectangle([0, 0, size, 18], fill=(0, 0, 0))
    draw.text((4, 2), f"SAMPLE thermal #{idx:02d} — NOT HIT-UAV", fill=(255, 200, 80))
    return img, boxes


def generate_samples(n: int = 15) -> list[dict]:
    items = []
    sample_dir = RAW / "sample_gen"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img, boxes = make_sample_thermal(i)
        path = sample_dir / f"sample_thermal_{i:03d}.png"
        img.save(path)
        items.append(
            {
                "stem": path.stem,
                "path": path,
                "boxes": boxes,
                "source_split": "sample",
                "label": "SAMPLE",
            }
        )
    return items


def split_items(items: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    order = list(items)
    rng.shuffle(order)
    n = len(order)
    n_train = max(1, int(n * 0.6))
    n_val = max(1, int(n * 0.2)) if n >= 5 else max(1, n - n_train - 1)
    if n_train + n_val >= n and n >= 3:
        n_val = max(1, n - n_train - 1)
    train = order[:n_train]
    val = order[n_train : n_train + n_val]
    test = order[n_train + n_val :]
    stems = lambda xs: {x["stem"] for x in xs}
    assert stems(train).isdisjoint(stems(val))
    assert stems(train).isdisjoint(stems(test))
    assert stems(val).isdisjoint(stems(test))
    return {"train": train, "val": val, "test": test}


def write_split(split: str, items: list[dict]) -> None:
    for it in items:
        img = Image.open(it["path"]).convert("RGB")
        out_img = OUT / split / "images" / f"{it['stem']}.jpg"
        out_lbl = OUT / split / "labels" / f"{it['stem']}.txt"
        img.save(out_img, quality=95)
        lines = []
        for b in it["boxes"]:
            cx, cy, w, h = b
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        out_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def validate_boxes() -> list[str]:
    errors: list[str] = []
    seen: dict[str, str] = {}
    for split in ("train", "val", "test"):
        img_dir = OUT / split / "images"
        lbl_dir = OUT / split / "labels"
        for img_path in img_dir.glob("*"):
            stem = img_path.stem
            if stem in seen:
                errors.append(f"doublon {stem} dans {seen[stem]} et {split}")
            seen[stem] = split
            lbl = lbl_dir / f"{stem}.txt"
            if not lbl.exists():
                errors.append(f"{split}/{stem}: image sans label")
                continue
            im = Image.open(img_path)
            W, H = im.size
            content = lbl.read_text(encoding="utf-8").strip()
            if not content:
                errors.append(f"{split}/{stem}: label vide (pas de Person)")
                continue
            for line in content.splitlines():
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{split}/{stem}: ligne YOLO invalide")
                    continue
                cls, cx, cy, w, h = parts[0], *map(float, parts[1:])
                if cls != "0":
                    errors.append(f"{split}/{stem}: classe {cls} ≠ Person(0)")
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    errors.append(f"{split}/{stem}: box hors normes YOLO")
                x1, y1, x2, y2 = yolo_to_xyxy([cx, cy, w, h], W, H)
                # allow 1px clamp tolerance
                if x1 < -1 or y1 < -1 or x2 > W + 1 or y2 > H + 1:
                    errors.append(f"{split}/{stem}: box décalée hors image ({x1},{y1},{x2},{y2})")
                if x2 - x1 < 2 or y2 - y1 < 2:
                    errors.append(f"{split}/{stem}: box trop petite / décalée")
    return errors


def write_previews(n: int = 5) -> list[Path]:
    collected: list[tuple[Path, Path]] = []
    for split in ("train", "val", "test"):
        for img_path in sorted((OUT / split / "images").glob("*")):
            lbl = OUT / split / "labels" / f"{img_path.stem}.txt"
            if lbl.exists():
                collected.append((img_path, lbl))
            if len(collected) >= n:
                break
        if len(collected) >= n:
            break

    outs: list[Path] = []
    for i, (img_path, lbl) in enumerate(collected[:n], 1):
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        draw = ImageDraw.Draw(im)
        for line in lbl.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            box = list(map(float, parts[1:]))
            x1, y1, x2, y2 = yolo_to_xyxy(box, W, H)
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 80), width=2)
            draw.text((x1, max(0, y1 - 12)), "Person POSSIBLE", fill=(0, 255, 80))
        draw.rectangle([0, 0, W, 18], fill=(0, 0, 0))
        draw.text((4, 2), f"preview {i} · {img_path.stem} · M3 thermal", fill=(255, 220, 80))
        out = PREVIEWS / f"preview_{i:02d}_{img_path.stem}.jpg"
        im.save(out, quality=95)
        outs.append(out)
    return outs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("hituav", "sample"), default="hituav")
    parser.add_argument("--max-images", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    clear_outputs()
    ensure_dirs()
    RAW.mkdir(parents=True, exist_ok=True)

    meta: dict = {
        "person_class_id": 0,
        "person_note": "Personne POSSIBLE en thermique — jamais survivant confirmé",
        "yolo_format": "class cx cy w h (normalized)",
        "license_notes": str((ROOT / "LICENSE_HITUAV.md").as_posix()),
        "performance_claim": False,
        "blind_training_allowed": False,
    }

    items: list[dict] = []
    source = args.source
    if source == "hituav":
        try:
            items = download_hituav_subset(args.max_images, args.seed)
            if len(items) < 5:
                raise RuntimeError(f"seulement {len(items)} images Person téléchargées")
            meta["source"] = "HIT-UAV subset"
            meta["license"] = "CC-BY-4.0"
            meta["attribution"] = "HIT-UAV Infrared Thermal Dataset (suojiashun et al.)"
        except Exception as e:
            print(f"[WARN] HIT-UAV inaccessible/incompatible: {e}", file=sys.stderr)
            print("[WARN] Pas de formation aveugle. Fallback SAMPLE illustratif.", file=sys.stderr)
            source = "sample"

    if source == "sample":
        items = generate_samples(max(15, min(args.max_images, 40)))
        meta["source"] = "SAMPLE illustrative"
        meta["license"] = "local SAMPLE (not HIT-UAV imagery)"
        meta["note"] = "Annotations locales alignées — démonstration seulement"

    splits = split_items(items, seed=args.seed)
    for split, xs in splits.items():
        write_split(split, xs)

    errors = validate_boxes()
    previews = write_previews(5)
    meta["splits"] = {k: len(v) for k, v in splits.items()}
    meta["previews"] = [p.name for p in previews]
    meta["validation_errors"] = errors
    meta["compatible_for_training"] = len(errors) == 0

    manifest = OUT / "manifest.json"
    manifest.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("=== prepare_thermal.py ===")
    print(f"source   : {meta['source']}")
    print(f"splits   : {meta['splits']}")
    print(f"previews : {len(previews)} → {PREVIEWS}")
    if errors:
        print("VALIDATION FAILED — ne pas lancer de formation aveugle:")
        for e in errors:
            print(" -", e)
        return 1
    print("VALIDATION OK — 5 previews avec boîtes Person alignées, aucune box décalée")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
