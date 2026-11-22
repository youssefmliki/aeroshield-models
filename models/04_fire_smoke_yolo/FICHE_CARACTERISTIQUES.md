# FICHE JURY — Détection feu / fumée

> **Ouvrir d'abord :** [`RAPPORT_MODELE.pdf`](./RAPPORT_MODELE.pdf)

| Caractéristique | Détail |
|-----------------|--------|
| **Nom** | Détection feu / fumée |
| **Algorithme** | YOLOv26-S |
| **Tâche** | Détection — fire / other / smoke |
| **Dataset** | Checkpoint HF (~8939 img auteur) + audit D-Fire |
| **Métriques clés** | mAP50 0.949 · mAP50-95 0.680 (métriques auteur) · tests locaux OK |
| **Statut** | **REAL (checkpoint public + smoke tests)** |
| **Checkpoint** | `weights/fire_smoke_best.pt` |
| **Dossier** | `04_fire_smoke_yolo/` |

## Contenu de ce dossier

| Élément | Chemin |
|---------|--------|
| Rapport PDF (détaillé) | `RAPPORT_MODELE.pdf` |
| Rapport texte | `RAPPORT_MODELE.md` |
| Illustrations résultats | `illustrations/` |
| Notebook | `notebook/` |
| Métriques JSON / courbes | `metrics/` |
| Code / poids | `code/` · `weights/` · `scripts/` (selon modèle) |

## Limite (honesté)

Prototype hackathon FireTwin / AeroShield. Ne remplace pas une certification incendie ni une inspection humaine.
