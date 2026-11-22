# INDEX JURY — Modèles AeroShield FireTwin

Un **dossier = un modèle**. Ouvrir le dossier → lire `FICHE_CARACTERISTIQUES.md` → ouvrir `RAPPORT_MODELE.pdf`.

| # | Dossier | Modèle | Algorithme | Statut | PDF | Illustrations |
|---|---------|--------|------------|--------|-----|---------------|
| 01 | [`01_flood_segformer/`](./01_flood_segformer/) | Segmentation d'inondation | SegFormer-B0 | **REAL** | oui | 11 |
| 02 | [`02_rgb_person_yolo/`](./02_rgb_person_yolo/) | Détection de personnes (RGB) | YOLOv8n | **REAL** | oui | 3 |
| 03 | [`03_thermal_person/`](./03_thermal_person/) | Personne thermique | YOLO (slot thermique) | **SAMPLE** | oui | 0 |
| 04 | [`04_fire_smoke_yolo/`](./04_fire_smoke_yolo/) | Détection feu / fumée | YOLOv26-S | **REAL (checkpoint public + smoke tests)** | oui | 15 |
| 05 | [`05_equipment_yolo/`](./05_equipment_yolo/) | Détection d'équipements industriels | YOLOv8n (9 classes) | **REAL** | oui | 9 |
| 06 | [`06_floorplan_unet/`](./06_floorplan_unet/) | Reconnaissance de plan d'étage | U-Net + ResNet-34 | **Baseline HF + pipeline** | oui | 1 |
| 07 | [`07_fire_spread_rf/`](./07_fire_spread_rf/) | Propagation scénario (synthétique) | Random Forest | **SYNTHETIC_MODEL (pas physique)** | oui | 2 |
| 08 | [`08_damage_resnet/`](./08_damage_resnet/) | Dommages avant / après | ResNet-18 pairé | **DATASET_MISSING** | oui | 0 |
| 09 | [`09_gas_sense_rf/`](./09_gas_sense_rf/) | Identification de gaz (GasSense) | Random Forest (128 features) | **REAL (expérimental, pas certifié)** | oui | 3 |

## Structure identique dans chaque dossier

```
XX_nom_modele/
├── FICHE_CARACTERISTIQUES.md   ← résumé jury (caractéristiques)
├── RAPPORT_MODELE.pdf         ← rapport détaillé
├── RAPPORT_MODELE.md
├── caracteristiques.json
├── illustrations/             ← résultats visuels
├── notebook/
├── metrics/
└── weights/ ou code/ …
```

## Honesté

- **REAL** = métriques mesurées / checkpoint validé pour la démo
- **SAMPLE / SYNTHETIC / DATASET_MISSING** = clairement non opérationnel
- Aucun modèle n'est une alarme incendie certifiée
