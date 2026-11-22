# CubiCasa5K + floorplan-to-3d — Dataset / Checkpoint Audit

**Date:** 2026-11-21  
**Status:** Audit complete from official sources. Full dataset **not yet present** on this machine (~5.5 GB zip).

## Sources inspected

| Resource | URL | Role |
|----------|-----|------|
| CubiCasa5K repo | https://github.com/CubiCasa/CubiCasa5k | Official dataset + SVG parser reference |
| CubiCasa5K paper | https://arxiv.org/abs/1904.01920 | Split sizes, annotation protocol |
| Zenodo record | https://zenodo.org/record/2613548 | Download: `cubicasa5k.zip` (~5.47 GB, md5 `0ce0b203d1e3c125b51087b219bd23b9`) |
| floorplan-to-3d | https://github.com/Yytsi/floorplan-to-3d | ResNet-34 U-Net baseline + SVG→mask + polygon extract |
| HF weights | https://huggingface.co/Yytsi/floorplan-to-3d-walls | `best.safetensors` + `config.yaml` |

## Dataset layout (after unzip)

Expected under `data/cubicasa5k/`:

- Official split lists: `train.txt`, `val.txt`, `test.txt` (paths relative to dataset root)
- Per sample folder containing at least:
  - `model.svg` — dense vector annotations (canonical label source)
  - Raster PNG(s) such as `F1_scaled.png` / `F1_original.png` — **do not use as GT alignment source** without verifying coordinate space

**Critical (verified in floorplan-to-3d `data.py`):** training/inference rasterizes **`model.svg` via cairosvg**, not the CubiCasa PNG. PNGs may not share the SVG coordinate space.

## Official splits (paper)

| Split | Count (paper) |
|-------|----------------|
| train | 4200–4400 (paper text cites 4400) |
| val | 100 |
| test | 500 |
| **total** | **5000** |

Always preserve the **source** `train.txt` / `val.txt` / `test.txt` from the zip. Do not re-split randomly.

## SVG label semantics (official CubiCasa)

From CubiCasa `floortrans/loaders/house.py` and floorplan-to-3d `svg_to_mask.py`:

| SVG token | Meaning | Used in our 4-class task? |
|-----------|---------|---------------------------|
| `id`/`class` **Wall** | Wall polygons | Yes → `wall` |
| **Railing** | Railings (official wall channel) | Optional; baseline TOKEN map omits Railing |
| **Door** | Door openings | Yes → `door` |
| **Window** | Window openings | Yes → `window` |
| `class="Space …"` | Room polygons (80+ room types) | Not a separate class; residual → `floor` |
| Furniture / icons | Fixed furniture etc. | Ignored → `floor` |
| `display:none` / `visibility:hidden` | Hidden floors/layers | Must skip (else multi-floor bleed into mask) |

Doors/windows often nest inside walls. Paint priority (baseline): **wall → door → window** so openings overwrite walls.

## Verified class schema (floorplan-to-3d) — AUTHORITATIVE for checkpoint

```
0 = floor   (default fill for every pixel; acts as “background” of structure)
1 = wall
2 = door
3 = window
```

**No separate 5th background class.** Adding one would break HF `best.safetensors` compatibility.

Reported HF validation **mIoU ≈ 0.983** (4 classes, epoch 26) — cite as upstream claim, not as our held-out run until we evaluate locally.

## Preprocessing (must match checkpoint)

From HF `config.yaml` / repo:

- Image size: **512×512**
- **Letterbox:** aspect-preserving scale + center pad
- ImageNet normalize: mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`
- Encoder: **resnet34**, U-Net via `segmentation_models_pytorch`
- Output: raw logits, 4 channels

## Checkpoint compatibility

| Artifact | Compatible with 4-class ResNet34-UNet? | Notes |
|----------|----------------------------------------|-------|
| `weights/best.safetensors` (HF) | **Yes** — preferred baseline | Load via `safetensors`; metadata may include `epoch`, `best_miou` |
| `weights/config.yaml` | Required companion | Must travel with weights |
| CubiCasa official `model_best_val_loss_var.pkl` | **No** — different multi-task architecture | Do not load into smp U-Net |
| `floorplan_best.pth` | Only if **we** train/export with identical schema | Convert safetensors→pt for FireTwin naming if needed |

## Domain gap (industrial FireTwin)

CubiCasa5K is dominated by **Nordic residential** CAD-style plans. Industrial factory drawings (heavy dimensions, equipment symbols, multi-sheet PDFs) will degrade wall/door IoU. Operator correction is mandatory before any 2D/3D twin geometry is treated as approved.

## Local status

| Item | Status |
|------|--------|
| Audit docs | Done |
| CubiCasa zip on disk | **Missing** — run `python scripts/download_cubicasa.py` |
| Prepared masks / `manifest.csv` | Pending dataset |
| HF baseline weights | Download via `python scripts/download_weights.py` |
| Our held-out metrics | **Not computed** (no fabricated numbers) |
