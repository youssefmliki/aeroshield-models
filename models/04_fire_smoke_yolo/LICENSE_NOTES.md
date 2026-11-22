# License & Attribution — FireTwin Fire/Smoke AI

## Delivered checkpoint
- File: `weights/fire_smoke_best.pt`
- Origin: https://huggingface.co/SalahALHaismawi/yolov26-fire-detection
- Architecture: YOLOv26-S (Ultralytics)
- Classes: `0=fire`, `1=other`, `2=smoke`
- Author metrics (self-reported): mAP50=94.9%, mAP50-95=68.0%
- Cite/credit the Hugging Face model card when redistributing weights.

## D-Fire dataset (prepared locally for optional fine-tuning)
- Official: https://github.com/gaia-solutions-on-demand/DFireDataset
- Mirror used: https://huggingface.co/datasets/hanvithSai/gavin-dfire-raw
- License: **CC BY 4.0**
- Label convention in D-Fire raw labels: `0=smoke`, `1=fire` (different from the delivered HF model!)
- Citation: Venâncio, Lisboa, Barbosa — Neural Computing and Applications, 2022.

## Disclaimer
Experimental visual detection only. Not a certified industrial fire alarm.
Does not detect gas or measure temperature.
