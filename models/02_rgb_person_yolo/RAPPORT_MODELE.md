# Rapport du modèle — 02 Personne RGB (YOLOv8n)

## Objectif
Détecter des personnes sur images RGB (vue drone / VisDrone) pour signaler une présence **possible** dans la zone.

## Architecture
- **Modèle :** YOLOv8n (Ultralytics)
- **Tâche :** détection d’objets 1 classe (`person`)
- **Entrée :** image RGB
- **Sortie :** boîtes + scores
- **Préentraînement :** COCO → fine-tune VisDrone person-only

## Pourquoi ce modèle
YOLOv8n est le nano Ultralytics : rapide (~ms), peu de VRAM, standard industrie pour la détection. Suffisant pour un prototype de cartographie de présence, sans prétendre à un comptage certifié.

## Dataset

### Provenance
Le détecteur est entraîné sur une **projection person-only** de **VisDrone2019-DET** (AISKYEYE / Zhu et al.), benchmark standard de détection aérienne collecté depuis des UAV en scènes urbaines et péri-urbaines. Les frames présentent une forte variation d’échelle (piétons très petits en altitude), flou de mouvement, occlusions (végétation, véhicules) et angles non nadir — conditions plus proches des flux drone FireTwin que des photos sol COCO.

Les labels VisDrone distinguent `0 = pedestrian` et `1 = people` (groupes). Pour FireTwin on **fusionne les deux en une seule classe YOLO `0 = person`** et on **supprime** voitures / vélos / tricycles. Cela colle à la question produit (« une silhouette humaine est-elle possiblement présente ? ») et évite de diluer un petit budget GPU sur trop de classes.

### Protocole de préparation
Script : `scripts/prepare_visdrone_person.py`.

1. Résoudre / télécharger VisDrone via les helpers Ultralytics `VisDrone.yaml` si besoin.
2. Parcourir les arbres officiels `images/{train,val,test}` + `labels/` (test-dev mappé sur `test`).
3. Réécrire chaque label en ne gardant que les IDs `{0,1}` → remappés en `0`.
4. Plafonner pour le run de livraison : **train 800 / val 200 / test 200** images (VisDrone complet est plus large ; les caps sont documentés, pas cachés).
5. Émettre `person.yaml` consommé par l’entraînement Ultralytics.

### Hygiène des splits
Les partitions officielles VisDrone sont **conservées** — pas de reshuffle aléatoire inter-splits. Les caps s’appliquent **à l’intérieur** de chaque partition. Les métriques de `metrics/metrics.json` portent sur le **test** tenu à l’écart (n=200) ; la val sert uniquement au monitoring d’entraînement.

### Licence / citation
Respecter les conditions recherche du challenge VisDrone. Citer les publications VisDrone / AISKYEYE en cas de redistribution des labels person-only dérivés. Sous-ensemble destiné à la démo R&D / jury FireTwin — **pas** une certification SAR opérationnelle.

### Biais connus
- Scènes aériennes rue / place ≠ usine indoor (domain shift attendu).
- Forte proportion de toutes petites boîtes → le rappel est structurellement plus dur que la précision.
- Fusionner `people` (groupes) dans `person` produit parfois de grandes boîtes au comportement différent d’un piéton isolé.

## Entraînement
- 30 époques, `scripts/train_rgb_person.py`
- Checkpoint : `weights/rgb_person_best.pt`

## Métriques (test)
| Métrique | Valeur |
|----------|--------|
| mAP50 | **0.462** |
| Precision | 0.609 |
| Recall | 0.422 |
| n images | 200 |

## Intégration
`POST /predict/rgb-person` — **distinct** du thermique.

## Limites
- Petites silhouettes / occlusions : rappel limité.
- Ce n’est **pas** le modèle thermique.
- Ne certifie ni identité ni état de santé.
