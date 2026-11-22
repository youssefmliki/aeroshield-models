# Rapport du modèle — 05 Détection d’équipements industriels

## Objectif
Reconnaître des **catégories visibles** d’équipements (bouteilles, cuves, coffrets, outils…) sur image caméra, comme indice visuel dans FireTwin.  
**Ne fournit pas** l’ID d’inventaire usine (ex. ELEC-CAB-01).

## Architecture
- **Algo :** YOLOv8n (Ultralytics), ~3,01 M paramètres
- **Préentraînement :** `yolov8n.pt` (COCO)
- **Entrée :** 640×640
- **Sortie :** boîtes multi-classes (9 classes)
- **Checkpoint :** `weights/equipment_best.pt` (~6 Mo)

### Classes
`cylinder`, `cap`, `marked`, `tank`, `electrical_cabinet`, `electrical_device`, `bolts_nuts_washer`, `industrial_tools`, `power_tools`

## Pourquoi YOLOv8n
Nano = faible VRAM (RTX 2050 4 Go, batch 4), inférence ~9 ms/image, stack déjà utilisée (person / fire). Transfer learning COCO accélère le fine-tune sur 15 époques.

## Dataset

### Provenance — fusion multi-sources
Les instances d’équipements industriels ont été assemblées en **fusionnant quatre jeux de détection Roboflow Universe**, chacun sous **CC BY 4.0** (créditer les auteurs de chaque projet en cas de redistribution) :

| Source (Roboflow Universe) | Rôle dans la fusion |
|----------------------------|---------------------|
| industrial-equipment-annotation | `cylinder`, `cap`, `marked` |
| detect-t (Tank) | `tank` |
| Electrical equipment | `electrical_cabinet`, `electrical_device` |
| FA Industrial Objects | `bolts_nuts_washer`, `industrial_tools`, `power_tools` |

Les IDs de classes ont été harmonisés dans un seul `data.yaml` YOLO aux neuf noms FireTwin. Images sans boîte après remapping retirées ; quasi-doublons JPEG (hash) retirés ; **fuites train/test évidentes issues de rafales vidéo** corrigées pour qu’aucune paire de frames temporelles proches ne se retrouve des deux côtés du split.

### Effectifs (run de livraison)
| Split | Images |
|-------|--------|
| train | **1745** |
| val | **280** |
| test | **232** |

Les métriques primaires de `metrics/metrics.json` sont calculées sur le split **test** uniquement (`evaluation_split: test`). La val a guidé l’early stopping (patience 8).

### Bruit d’annotation / déséquilibre
- `marked` et classes visserie/outils = longue traîne → mAP50-95 par classe faible **attendu et reporté**, pas lissé.
- Certaines boîtes `tank` viennent de pipelines semi-automatiques → bruit de localisation possible.
- La catégorie prédite est un **indice visuel**, pas un ID CMMS, un inventaire chimique, ni un état « sous tension / sûr ».

### Licence
CC BY 4.0 sur les jeux Roboflow constitutifs : conserver la chaîne d’attribution si le jeu fusionné ou les poids sortent du hackathon.

## Entraînement
- 15 époques, patience 8, seed 42, AdamW
- Durée ~23–24 min sur RTX 2050
- Notebook : `notebook/FireTwin_Equipment_Training.ipynb`

## Métriques (test jamais vu à l’entraînement)
| Métrique | Valeur |
|----------|--------|
| Precision (moy.) | 0.721 |
| Recall (moy.) | 0.439 |
| **mAP50** | **0.507** |
| mAP50-95 | 0.380 |

Courbes : `metrics/results.png`, `PR_curve.png`, `confusion_matrix.png`.  
Exemples : `examples/`.

## Intégration
API standalone : `code/api.py` / `code/inference.py` (module séparé, non cassant pour `ai/api.py`).

## Limites
- `marked` et petites classes outils : rappel faible / peu d’images test.
- Cuves partiellement auto-annotées → bruit possible.
- Classe prédite ≠ contenu chimique, ≠ état de marche, ≠ décision pompier.
