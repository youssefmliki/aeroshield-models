# Rapport du modèle — 03 Personne thermique

## Objectif
Slot API thermique pour démontrer le parcours UI (caméra thermique → présence possible).

## Architecture
- Détecteur YOLO chargé depuis `weights/thermal_best.pt` **si** disponible.
- Sinon réponse explicite **SAMPLE** (régions simulées / fallback), jamais présentée comme modèle entraîné fiable.

## Pourquoi ce choix
Le jeu HIT-UAV local était trop petit pour des métriques honnêtes. Priorité au demo jury fiable : flood + RGB person en REAL, thermique clairement **SAMPLE**.

## Dataset

### Provenance (tentative)
Le corpus thermique visé est un sous-ensemble **HIT-UAV** (CC BY 4.0) : imagerie infrarouge UAV avec annotations personne. Les stems dans `metrics/metrics.json` suivent la convention HIT-UAV (`altitude_angle_…`). L’extrait local disponible pendant le hackathon était **beaucoup trop petit** pour une claim crédible (fine-tune journalisé ≈ 14 images train, 8 époques).

### Pourquoi SAMPLE
À ce volume, tout mAP publié serait statistiquement non significatif et facilement overfitté. FireTwin conserve donc le **slot API/UI thermique** mais étiquette les sorties `source: SAMPLE` / `mode: SAMPLE_GT` : des boîtes GT HIT-UAV peuvent être affichées pour démontrer le parcours, **explicitement pas** comme prédictions modèle.

### Scripts préparation / inférence
- `scripts/prepare_thermal.py` — helpers de layout / conversion pour un futur jeu thermique adéquat.
- `scripts/infer_thermal.py` — charge `weights/thermal_best.pt` **si** présent ; sinon bascule sur le chemin SAMPLE documenté dans les métriques.

### Politique split / métriques
Aucune précision/rappel REAL n’est revendiquée. `metrics/metrics.json` fixe `performance_claim: false`, `metrics_valid: false`, et trace l’échec (`baseline yolov8n.pt` faible sur IR ; finetune court insuffisant). Ne **pas** confondre avec `02_rgb_person_yolo` (VisDrone RGB).

### Exigences futures (honnêtes)
Un modèle thermique crédible demanderait : (1) des centaines–milliers de frames IR avec protocole d’annotation stable, (2) splits sensibles aux séquences (pas de fuite frame-à-frame), (3) métadonnées de calibration (palette / radiométrique vs pseudo-couleur), (4) négatifs (machines chaudes) pour contrôler les faux positifs. Jusque-là, cette fiche reste SAMPLE.

## Métriques
**Non présentées comme REAL.** Voir `metrics/metrics.json` et label SAMPLE dans `jury_metrics_summary.json`.

## Intégration
`POST /predict/thermal` → `source: SAMPLE` si charge réelle non validée.

## Limites
- Ne pas confondre avec le détecteur RGB VisDrone.
- Aucune validation opérationnelle thermique.
