# FICHE JURY — Détection de personnes (RGB)

> **Ouvrir d'abord :** [`RAPPORT_MODELE.pdf`](./RAPPORT_MODELE.pdf)

| Caractéristique | Détail |
|-----------------|--------|
| **Nom** | Détection de personnes (RGB) |
| **Algorithme** | YOLOv8n |
| **Tâche** | Détection d'objets — classe person |
| **Dataset** | VisDrone2019-DET (person-only) |
| **Métriques clés** | Test mAP50 0.462 · P 0.609 · R 0.422 (n=200) |
| **Statut** | **REAL** |
| **Checkpoint** | `../../../weights/rgb_person_best.pt` |
| **Dossier** | `02_rgb_person_yolo/` |

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
