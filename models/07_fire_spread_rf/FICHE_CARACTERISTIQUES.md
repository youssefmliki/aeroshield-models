# FICHE JURY — Propagation scénario (synthétique)

> **Ouvrir d'abord :** [`RAPPORT_MODELE.pdf`](./RAPPORT_MODELE.pdf)

| Caractéristique | Détail |
|-----------------|--------|
| **Nom** | Propagation scénario (synthétique) |
| **Algorithme** | Random Forest |
| **Tâche** | Risque salles adjacentes — SIMULATION |
| **Dataset** | Scénarios 100% synthétiques (règles jouet) |
| **Métriques clés** | Accuracy synthétique 0.934 · Macro F1 0.927 |
| **Statut** | **SYNTHETIC_MODEL (pas physique)** |
| **Checkpoint** | `outputs/fire_spread_scenario_rf.joblib` |
| **Dossier** | `07_fire_spread_rf/` |

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
