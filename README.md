# AeroShield FireTwin — AI

## Pour le jury

**Ouvrir :** [`models/00_INDEX_JURY.md`](models/00_INDEX_JURY.md)

```
ai/
├── README.md              ← vous êtes ici
├── api.py                 ← API FastAPI (runtime)
├── requirements.txt
├── licenses/              ← licences datasets
├── models/                ← 1 dossier = 1 modèle
│   ├── 00_INDEX_JURY.md
│   ├── 01_flood_segformer/
│   ├── 02_rgb_person_yolo/
│   ├── 03_thermal_person/
│   ├── 04_fire_smoke_yolo/
│   ├── 05_equipment_yolo/
│   ├── 06_floorplan_unet/
│   ├── 07_fire_spread_rf/
│   ├── 08_damage_resnet/
│   └── 09_gas_sense_rf/
└── data/                  ← local only (gitignored)
```

Dans chaque modèle : `FICHE_CARACTERISTIQUES.md` → `RAPPORT_MODELE.pdf` → `illustrations/` → `notebook/` / `scripts/`.

## API

```powershell
cd ai
pip install -r requirements.txt
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Poids : `../weights/`
