# FICHE JURY — Reconnaissance de plan d'étage

> **Ouvrir d'abord :** [`RAPPORT_MODELE.pdf`](./RAPPORT_MODELE.pdf)

| Caractéristique | Détail |
|-----------------|--------|
| **Nom** | Reconnaissance de plan d'étage |
| **Algorithme** | U-Net + ResNet-34 |
| **Tâche** | Segmentation floor / wall / door / window |
| **Dataset** | CubiCasa5K (CC BY-NC) |
| **Métriques clés** | mIoU val auteur HF ≈ 0.983 · sortie DRAFT_UNVERIFIED |
| **Statut** | **Baseline HF + pipeline** |
| **Checkpoint** | `scripts/download_weights.py → best.safetensors` |
| **Dossier** | `06_floorplan_unet/` |

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
