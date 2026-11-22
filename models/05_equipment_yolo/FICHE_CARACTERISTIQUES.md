# FICHE JURY — Détection d'équipements industriels

> **Ouvrir d'abord :** [`RAPPORT_MODELE.pdf`](./RAPPORT_MODELE.pdf)

| Caractéristique | Détail |
|-----------------|--------|
| **Nom** | Détection d'équipements industriels |
| **Algorithme** | YOLOv8n (9 classes) |
| **Tâche** | Détection multi-classes équipements |
| **Dataset** | Roboflow ×4 (CC BY 4.0) — 1745/280/232 |
| **Métriques clés** | Test mAP50 0.507 · mAP50-95 0.380 · P 0.721 · R 0.439 |
| **Statut** | **REAL** |
| **Checkpoint** | `weights/equipment_best.pt` |
| **Dossier** | `05_equipment_yolo/` |

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
