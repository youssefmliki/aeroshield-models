# Rapport du modèle — 09 GasSense (Random Forest)

## Objectif
Classifier **6 gaz** à partir d’un vecteur de 128 features (réseau de 16 capteurs chimiques), pour enrichir FireTwin.
**Expérimental** — pas un détecteur de fuite certifié.

## Architecture
- **Algo :** RandomForestClassifier (scikit-learn)
- **Entrée :** 128 features (16 capteurs × 8)
- **Classes :** Ethanol, Ethylene, Ammonia, Acetaldehyde, Acetone, Toluene
- **Checkpoint :** `weights/gas_classifier.joblib` (alias `best_gas_model.pkl`)

## Pourquoi Random Forest
Robuste sur données tabulaires UCI, entraînement CPU rapide, importances interprétables, adapté au drift partiel vs réseaux profonds lourds.

## Dataset

### Provenance
Corpus primaire : **UCI Gas Sensor Array Drift Dataset** (UCI ML Repository **ID 224**), introduit par Vergara (2012), DOI **10.24432/C5RP6W**.  
URL : https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset  

La release brute contient **10 batches d’acquisition** enregistrés sur une longue période sur un **réseau de 16 capteurs chimiques à oxyde métallique**. Chaque présentation est encodée en **128 features LIBSVM densifiées** (8 réponses ingénierie × 16 capteurs), **pas** des traces ADC brutes MQ / ESP32. FireTwin GasSense suppose donc un vecteur **128-D compatible schéma** à l’inférence.

### Échelle & espace de labels
| Item | Valeur |
|------|--------|
| Samples conservés (tous batches) | **13 910** |
| Features / sample | **128** (cohérent entre batches) |
| Classes (IDs source 1-based) | Ethanol, Ethylene, Ammonia, Acetaldehyde, Acetone, Toluene |
| Batches | 1 … 10 (fortement déséquilibrés tôt ; batch 10 équilibré 600×6) |

Comptes globaux (voir `DATASET_AUDIT.md`) : Ethanol 2565, Ethylene 2926, Ammonia 1641, Acetaldehyde 1936, Acetone 3009, Toluene 1833. Plusieurs batches intermédiaires n’ont **aucun** Toluene — autre raison de **ne pas** mélanger les batches au hasard pour revendiquer la généralisation.

### Protocole de split primaire (anti-fuite / drift-aware)
Les surfaces de capteurs vieillissent et les conditions environnementales dérivent ; les signatures d’un même gaz changent dans le temps. L’évaluation **primaire** utilise donc un protocole **séparé par batch** :

- **Train :** batches **1–9** (n≈10 310)
- **Test :** batch **10** (n=**3600**, 600 par gaz)

Le preprocess (`StandardScaler`) est fit **uniquement** sur les batches d’entraînement. Labels cibles et IDs de batch ne sont **jamais** des features d’entrée. Une CV groupée par batch (macro-F1 RF ≈ 0,86 ± 0,09) est journalisée comme contrôle de stabilité secondaire.

### Split exploratoire (non revendiqué)
Un mélange aléatoire stratifié 80/20 sur tous les batches donne accuracy≈0,995 — enregistré sous `evaluation_A_random_stratified_exploratory` comme **avertissement**. Ce chiffre **surestime** sous drift et n’est **pas** la métrique headline GasSense.

### Filtres qualité
Le `gas_data.npz` traité ne retient que des valeurs finies ; l’audit reporte **0** ligne exclue pour NaN/Inf dans la livraison nettoyée. Histogrammes batch par batch : `DATASET_AUDIT.md`.

### Avertissement de compatibilité
Des vecteurs issus de cartes MQ low-cost ne sont **pas** des remplacements plug-and-play. Cross-sensibilité, humidité et concentration ne sont **pas** récupérées — la tâche est uniquement la **classification d’identité** de mesures compatibles UCI.

## Métriques PRIMAIRES (batch-separated)
| Métrique | Valeur |
|----------|--------|
| Accuracy | **0.602** |
| Macro F1 | **0.600** |
| Macro Precision | 0.643 |
| Macro Recall | 0.602 |

⚠️ Split aléatoire 80/20 donne ~0.995 — **surestimé** (drift) ; non revendiqué.

## Illustrations
`illustrations/confusion_matrix.png`, `class_distribution.png`

## Limites
Schéma features spécifique à ce réseau de capteurs. Probabilité ≠ concentration / toxicité / fuite.
