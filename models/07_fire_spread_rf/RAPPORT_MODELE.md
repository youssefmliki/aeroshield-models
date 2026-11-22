# Rapport du modèle — 07 Risque de propagation scénario (Random Forest)

## Objectif
Classer / scorer des **salles adjacentes candidates** dans un scénario d’incident **synthétique** (jouet), pour enrichir la démo visuelle FireTwin.  
**Ce n’est pas** un prédicteur physique de propagation du feu.

## Architecture
- **Algo :** Random Forest (scikit-learn)
- **Entrée :** vecteur de features (température simulée, fumée, distance graphe, portes ouvertes, matériel, …)
- **Sortie :** label salle candidate + `scenario_risk_score` + `source: SYNTHETIC_MODEL`
- **Sérialisation :** `outputs/fire_spread_scenario_rf.joblib` + `feature_schema.json`

## Pourquoi Random Forest
- Interprétable (importance des variables) pour le jury
- Robuste sur petits tableaux tabulaires synthétiques
- Entraînement instantané CPU — idéal sous deadline
- Ne prétend pas remplacer NIST CFAST / FDS

## Dataset

### Provenance — 100 % synthétique
Il n’y a **aucune** télémétrie d’incendie réel dans ce dossier. Les lignes sont produites par `code/generate_synthetic.py` : pour chaque `scenario_id` jouet, une salle source et plusieurs `candidate_room` adjacentes sont tirées sur un graphe à 12 salles. Features continues / binaires (`temp_delta_c`, `smoke_level`, `graph_distance`, `door_open`, `fuel_load`, `ventilation`, `sprinkler_active`, `occupant_count`, `time_since_ignition_min`) tirées dans des plages explicites, puis labellisées par une **règle de score transparente** (`will_be_affected` si score pondéré > 0,45). Le rôle de la Random Forest est d’approximer cette règle — rien de plus.

### Volume & protocole de split
Générateur par défaut : **400 scénarios** → de l’ordre de ~1,5k lignes candidates. Split **groupé par `scenario_id`** (≈70 % / 15 % / 15 % d’affectation de scénarios) pour qu’aucun incident ne fuite entre train et test. Les métriques livrées (`metrics/metrics.json`) portent sur la tranche **test** held-out par scénario (ex. 273 lignes / 60 scénarios sur le run figé).

### Artifacts de schéma
- `data/synthetic_scenarios.csv` — matérialisation du générateur
- `data/feature_schema.json` — liste des features + disclaimer
- Colonne label : `will_be_affected` ∈ {0,1}

### Ce que mesurent (et ne mesurent pas) les métriques
Accuracy / macro-F1 mesurent l’**accord avec la règle jouet**, autrement dit « la RF a-t-elle appris notre simulateur ? ». Ce n’est **pas** une compétence de propagation physique. NIST **CFAST / FDS** sont cités seulement comme références physiques *futures* et n’ont **pas** été exécutés ici. L’UI / API doit conserver `source: SYNTHETIC_MODEL` et ne doit pas piloter l’évacuation certifiée (l’A* déterministe reste maître).

## Entraînement / évaluation
```powershell
cd ai/models/07_fire_spread_rf
pip install -r requirements.txt
python code/train.py
```
Notebook : `notebook/FireTwin_FireSpread_Training.ipynb`

## Métriques (exécutées — synthétiques uniquement)
Fichier : `metrics/metrics.json`

| Métrique | Valeur |
|----------|--------|
| Accuracy test | **0.934** |
| Macro F1 | **0.927** |
| Lignes test | 273 |
| Source | `SYNTHETIC_MODEL` |

Libellé obligatoire : *synthetic scenario accuracy* — **pas** une performance monde réel.

## Intégration
- `code/inference.py` → `predict_scenario_next_room_risk(features)`
- Ne pilote **pas** les routes d’évacuation (moteur A* déterministe reste maître)
- Affichage Twin : `SIMULATION / NOT PHYSICALLY VALIDATED`

## Limites
- Zéro validation monde réel
- Interdit pour décisions de lutte incendie
- Optionnel face au moteur d’incident déjà en place
