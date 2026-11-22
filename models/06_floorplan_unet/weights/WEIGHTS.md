# Poids FloorPlan

Les fichiers `best.safetensors` (~98 Mo) et `floorplan_best.pth` ne sont **pas** commités (limite GitHub).

```powershell
cd ai/models/06_floorplan_unet
pip install -r requirements.txt
python scripts/download_weights.py
```

Source : https://huggingface.co/Yytsi/floorplan-to-3d-walls (MIT)  
Dataset d’entraînement amont : CubiCasa5K (CC BY-NC) — ne pas redistribuer le zip.
