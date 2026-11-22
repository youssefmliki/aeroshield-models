# Rapport du modèle — 08 Évaluation visuelle de dommages (avant / après)

## Objectif
Comparer deux photos du **même équipement / zone** (avant vs après incident) et classer :
- `visible_change`
- `no_visible_change`  
Faible confiance → `REVIEW_REQUIRED`.  
**Ne diagnostique pas** cause incendie, tenue structurelle, ni aptitude à réutiliser l’équipement.

## Architecture
- **Algo :** encodeur **ResNet-18** (torchvision, ImageNet) partagé
- **Fusion :** features before ∥ features after ∥ |before − after|
- **Tête :** classification binaire
- **Checkpoint cible :** `results/damage_best.pt` (après entraînement réel)

## Pourquoi ResNet-18 pairé
Architecture classique et légère pour similarité / changement ; préentraînement ImageNet ; adaptée à un petit jeu de paires indoor. Plus honnête que d’utiliser xBD (satellite) pour une usine.

## Dataset

### Schéma requis (pas encore rempli)
L’entraînement refuse de tourner sans vrai manifeste. Copier `data/dataset_manifest_template.csv` → `data/manifest.csv` et renseigner des paires **indoor** licenciées :

| Colonne | Sens |
|---------|------|
| `before_path` | Chemin relatif photo RGB pré-incident |
| `after_path` | Chemin relatif photo post-incident du **même** équipement / zone |
| `label` | `visible_change` \| `no_visible_change` |
| `facility_id` | Clé site (groupement) |
| `incident_id` | Clé incident (groupement) |
| `asset_id` | ID équipement / zone (métadonnée API, pas une classe) |
| `split` | `train` / `val` / `test` |

### Hygiène des splits
Assigner des groupes entiers **`facility_id` / `incident_id`** à un seul split. Découper aléatoirement des frames de la même pièce ou rafale caméra ferait fuiter l’apparence et gonflerait les métriques.

### Corpus explicitement rejetés
Les jeux satellite **xBD / xView2** (dommages de bâtiments) sont **hors domaine** pour l’inspection indoor industrielle. Les utiliser produirait de beaux chiffres qui ne transfèrent pas aux photos facility FireTwin — `code/train.py` et ce rapport interdisent ce raccourci.

### Statut livraison actuelle
`metrics/metrics.json` → `status: DATASET_MISSING`, `metrics: null`. L’architecture (ResNet-18 partagé + tête \|diff\| concaténée) est implémentée derrière le gate dans `code/train.py` / `code/predict.py`, mais **aucune performance n’est revendiquée**. Quand de vraies paires arriveront, relancer l’entraînement et remplacer cette section par P/R/F1 mesurés sur les facilities de test groupées.

## Entraînement
Uniquement si `data/manifest.csv` + images réelles sont fournis :

```powershell
cd ai/models/08_damage_resnet
python code/train.py --manifest data/manifest.csv
```

Sinon : `MODEL_UNAVAILABLE` + comparaison manuelle opérateur.

## Métriques
`metrics/metrics.json` → statut `DATASET_MISSING` jusqu’à un run réel.

## Intégration prévue
`POST /predict/damage` : fichiers before/after + IDs room/asset fournis séparément.  
Observation **PENDING** jusqu’à revue humaine ; ne change pas accessibilité des routes automatiquement.

## Limites
- Différence visuelle ≠ preuve de dommage feu
- Pas de polygones de dégâts
- Décisions sécurité = inspection humaine
