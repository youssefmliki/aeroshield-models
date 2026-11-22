# Rapport du modèle — 04 Feu / Fumée (YOLOv26-S)

## Objectif
Détecter visuellement **feu** et **fumée** sur images caméra pour FireTwin.
Expérimental — **pas** une alarme incendie certifiée.

## Architecture
- **Algo :** YOLOv26-S (Ultralytics)
- **Classes :** `fire` (0), `other` (1), `smoke` (2)
- **Checkpoint :** `weights/fire_smoke_best.pt`
- **Origine :** Hugging Face `SalahALHaismawi/yolov26-fire-detection` (utilisé prêt pour le hackathon)

## Pourquoi ce modèle
Checkpoint public performant, classes feu/fumée déjà alignées, intégration YOLO homogène avec équipements / personnes. Évite un long fine-tune sous deadline.

## Dataset

### Provenance du checkpoint livré
La livraison hackathon utilise les poids **publiés** YOLOv26-S Ultralytics depuis Hugging Face  
[`SalahALHaismawi/yolov26-fire-detection`](https://huggingface.co/SalahALHaismawi/yolov26-fire-detection), copiés vers `weights/fire_smoke_best.pt`.  
Nous n’avons **pas** ré-entraîné ce réseau de bout en bout pour le package jury. Les têtes de classes sont déjà alignées FireTwin : **`0=fire`, `1=other`, `2=smoke`**.

### Jeu d’entraînement de l’auteur (cité, non re-hébergé)
D’après la model card amont, l’entraînement porte sur de l’ordre de **~8 939** images feu/fumée (collection Roboflow, **CC BY 4.0** — citer les auteurs Roboflow d’origine en cas de redistribution). Recette rapportée : **100 époques**, batch **16**, imgsz **640**. Les métriques headline (mAP50≈0.949, etc.) sont **déclarées par l’auteur** ; ce dépôt les enregistre dans `metrics/metrics.json` sous `author_reported_metrics` et **ne prétend pas** les avoir recalculées sur un split D-Fire officiel ici.

### Audit local / fine-tune futur
Un extrait **D-Fire** a été audité sur disque pour un éventuel fine-tune adapté au domaine (stills industrielles, variabilité fumée). Ce fine-tune a été **démarré puis abandonné sur demande** sous deadline — voir `training_note` dans les métriques. Les **smoke tests** locaux (image feu → boîtes ; négatif → vide ; fumée → variable) valident le câblage et attrapent les échecs grossiers, pas une eval style COCO complète.

### Limites de domaine
Images web/Roboflow feu ≠ chaque hall industriel (vapeur, poussière, glare LED, EPI orange). L’apparence de la fumée dérive particulièrement. Traiter les sorties API comme **indices visuels expérimentaux**, jamais comme boucle d’alarme certifiée.

## Métriques publiées par l'auteur (non recalculées end-to-end ici)
| Métrique | Valeur |
|----------|--------|
| mAP50 | **0.949** |
| mAP50-95 | 0.680 |
| Precision | 0.896 |
| Recall | 0.888 |

## Tests locaux
Inférence réelle OK sur images feu / négatif / fumée (voir `illustrations/`).
Fumée D-Fire : variable (décalage de domaine possible).

## Illustrations
Dossier `illustrations/` — prédictions feu, fumée, négatif, cas industriels.

## Limites
Ne mesure pas température ni gaz. Ne remplace pas détecteurs certifiés.
