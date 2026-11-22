# FICHE JURY — Identification de gaz (GasSense)

> **Ouvrir d'abord :** [`RAPPORT_MODELE.pdf`](./RAPPORT_MODELE.pdf)

| Caractéristique | Détail |
|-----------------|--------|
| **Nom** | Identification de gaz (GasSense) |
| **Algorithme** | Random Forest (128 features) |
| **Tâche** | Classification 6 gaz |
| **Dataset** | UCI Gas Sensor Array Drift — test batch 10 |
| **Métriques clés** | Accuracy 0.602 · Macro F1 0.600 (batch-separated) |
| **Statut** | **REAL (expérimental, pas certifié)** |
| **Checkpoint** | `weights/gas_classifier.joblib` |
| **Dossier** | `09_gas_sense_rf/` |

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
