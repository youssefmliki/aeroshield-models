# FICHE JURY — Dommages avant / après

> **Ouvrir d'abord :** [`RAPPORT_MODELE.pdf`](./RAPPORT_MODELE.pdf)

| Caractéristique | Détail |
|-----------------|--------|
| **Nom** | Dommages avant / après |
| **Algorithme** | ResNet-18 pairé |
| **Tâche** | visible_change vs no_visible_change |
| **Dataset** | DATASET_MISSING (paires indoor requises) |
| **Métriques clés** | Non calculées (pas de dataset) |
| **Statut** | **DATASET_MISSING** |
| **Checkpoint** | `results/damage_best.pt (après train)` |
| **Dossier** | `08_damage_resnet/` |

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
