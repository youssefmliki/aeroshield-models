# Rapport du modèle — 06 Reconnaissance de plan d’étage

## Objectif
Proposer un **brouillon** de segmentation (sol / murs / portes / fenêtres) à partir d’un plan, puis laisser un opérateur corriger l’échelle et la géométrie avant génération 2D/3D.  
Statut de sortie : toujours `DRAFT_UNVERIFIED` jusqu’à validation humaine.  
**Pas** une carte d’évacuation certifiée.

## Architecture
- **Algo :** U-Net + encodeur **ResNet-34** (`segmentation_models_pytorch`)
- **Classes (IDs) :** `floor=0`, `wall=1`, `door=2`, `window=3`  
  → pas de 5ᵉ classe « background » : le sol est le remplissage par défaut (compatibilité checkpoint).
- **Entrée :** 512×512, letterbox, normalisation ImageNet
- **Post-traitement :** contours → polygones mur/porte/fenêtre (`code/geometry.py`)
- **Baseline :** [floorplan-to-3d](https://github.com/Yytsi/floorplan-to-3d) + poids HF

## Pourquoi ce modèle
Pipeline public déjà entraîné sur CubiCasa5K, schéma de classes clair, extrude 3D possible après vectorisation. ResNet-34 U-Net = bon équilibre précision murs / coût compute. Évite de partir de zéro sous deadline.

## Dataset

### Provenance
Les classes structurelles suivent **CubiCasa5K** (Kalervo et al., arXiv:1904.01920) : **5 000** plans résidentiels avec annotations vectorielles **SVG** denses. Téléchargement canonique : Zenodo **10.5281/zenodo.2613548** (`cubicasa5k.zip`, ~5,5 Go, md5 dans `DATASET_AUDIT.md`). Licence : **CC BY-NC** — voir `LICENSE_NOTES.md` (pas de redistribution commerciale du dataset).

### Splits officiels (papier / listes du zip)
| Split | Effectif (papier) |
|-------|-------------------|
| train | ~4200–4400 |
| val | 100 |
| test | 500 |
| **total** | **5000** |

Toujours préférer `train.txt` / `val.txt` / `test.txt` du zip. **Ne pas** inventer un re-split aléatoire.

### Sémantique labels & rasterisation (critique)
Les PNG CubiCasa ne sont **pas** des sources GT de confiance pour l’alignement. La lignée floorplan-to-3d rasterise **`model.svg` via cairosvg**. Jetons SVG utiles : Wall, Door, Window ; polygones de pièces `Space …` et meubles → **floor**. Couches cachées (`display:none`) à ignorer. Priorité de peinture si chevauchement : **wall → door → window** (ouvertures écrasent les murs). IDs figés pour le checkpoint HF : `0=floor, 1=wall, 2=door, 3=window` (**pas** de 5ᵉ classe background).

### Contrat de preprocess
512×512 **letterbox**, mean/std ImageNet, logits U-Net ResNet-34 (4 canaux). Documenté dans `config.yaml` HF / `CLASS_MAPPING.md`.

### Statut local
CubiCasa complet peut être absent sur une machine donnée. `metrics/test_metrics.json` reste donc **`NOT_COMPUTED`** tant que `scripts/download_cubicasa.py` → `floorplan.prepare_data` → `floorplan.evaluate` n’ont pas réussi. Le mIoU val HF ≈ 0.983 est une **claim auteur**, pas notre mesure held-out.

### Écart de domaine
Plans résidentiels nordiques ≠ plans d’usine (épaisseurs de trait, symboles, cartouches). FireTwin renvoie toujours **DRAFT_UNVERIFIED** jusqu’à correction échelle/géométrie par un opérateur.

## Poids
| Fichier | Rôle |
|---------|------|
| HF `best.safetensors` | Baseline vérifiée (époque 26, mIoU val **upstream** 0.983) |
| `floorplan_best.pth` | Alias exporté compatible |

Téléchargement (poids ~98 Mo, non commités) :

```powershell
cd ai/models/06_floorplan_unet
python scripts/download_weights.py
```

## Métriques locales
`metrics/test_metrics.json` : **NOT_COMPUTED** tant que CubiCasa n’est pas préparé ici.  
Ne pas confondre la mIoU HF (auteur) avec une évaluation jury locale.

## Pipeline FireTwin
Upload plan → prédiction → **correction manuelle** → génération 2D/3D.

## Intégration
```powershell
uvicorn code.api:app --app-dir . --port 8010
# ou: python -m code.inference --image plan.png --weights weights/best.safetensors
```

## Limites
- Domaine résidentiel nordique ≠ plans industriels usine.
- CubiCasa NC : pas de redistribution commerciale du dataset.
- Géométrie non validée ≠ routage d’évacuation.
