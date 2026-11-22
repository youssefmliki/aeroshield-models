# Notes de licence — données M3 (PERSONNES THERMIQUES)

## HIT-UAV Infrared Thermal Dataset

- **Source** : [suojiashun/HIT-UAV-Infrared-Thermal-Dataset](https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset)
- **Archive** : Zenodo v1.2 (~815 MB) — [DOI record](https://zenodo.org/records/7633120)
- **Licence** : **CC-BY-4.0** (attribution requise)
- **Contenu** : ~2898 images thermiques IR UAV + boîtes (Person, Bicycle, Car, OtherVehicle, …)
- **Annotations** :
  - COCO-like JSON (`normal_json/annotations/*.json`) — attention : clé `annotation` (singulier)
  - XML (`normal_xml`)
  - YOLO labels (`yolo_labels/{train,val,test}`)
  - Bbox standard : pixels `[x, y, w, h]` (COCO) ou YOLO normalisé
- **Classe Person** : id **0** (personnes POSSIBLES en thermique — jamais « survivant confirmé »)

### Citation / attribution

Utiliser HIT-UAV avec mention CC-BY-4.0 et citation du dépôt / article associés.

## Jeu local de ce prototype

Si le ZIP HIT-UAV n’est pas extrait dans `ai/data/hituav_raw/`,  
`prepare_thermal.py` crée un sous-ensemble **SAMPLE** illustratif :

- images thermiques synthétiques + boîtes Person alignées
- **aucune formation aveugle** sur annotations incompatibles
- **pas d’annonce de performance** terrain

Pour importer HIT-UAV réel :

```bash
# Extraire le ZIP dans ai/data/hituav_raw/
python ai/prepare_thermal.py --source hituav --max-images 60
```
