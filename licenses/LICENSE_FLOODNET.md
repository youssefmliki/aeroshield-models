# Notes de licence — données M2 (EAU)

## FloodNet Supervised v1.0 (dataset officiel)

- **Source** : [BinaLab/FloodNet-Supervised_v1.0](https://github.com/BinaLab/FloodNet-Supervised_v1.0)
- **Licence** : [Community Data License Agreement — Permissive (CDLA-Permissive-1.0)](https://cdla.io/permissive-1-0/)
- **Contenu** : 2343 images UAV RGB post-Hurricane Harvey + masques de segmentation
- **Téléchargement** : lien Dropbox indiqué dans le README GitHub (jeu complet volumineux)
- **Citation requise** si utilisation des données FloodNet :

```
Rahnemoonfar et al., "FloodNet: A High Resolution Aerial Imagery Dataset
for Post Flood Scene Understanding", IEEE Access, 2021.
DOI: 10.1109/ACCESS.2021.3090981
```

### Classes (segmentation)

| ID | Classe |
|----|--------|
| 0 | Background |
| 1 | Building Flooded |
| 2 | Building Non-Flooded |
| 3 | Road Flooded |
| 4 | Road Non-Flooded |
| 5 | Water |
| 6 | Tree |
| 7 | Vehicle |
| 8 | Pool |
| 9 | Grass |

Format attendu FloodNet : dossiers `*-org-img` (RGB) + `*-label-img` (masques pixel).

## Jeu local utilisé dans ce prototype (hackathon)

Le téléchargement complet FloodNet n’est **pas** embarqué ici (taille / temps).  
`prepare_flood.py` crée un **sous-ensemble illustratif local** (`SAMPLE`) :

- couples image RGB / masque alignés
- classes alignées sur la table FloodNet ci-dessus (eau = classe 5)
- **aucune performance / IoU ne doit être annoncée** sur ce jeu SAMPLE
- usage : démonstration illustrative AeroShield 360 uniquement

Si vous placez une extraction FloodNet dans `ai/data/floodnet_raw/`, relancez :

```bash
python ai/prepare_flood.py --source floodnet --max-pairs 100
```

pour construire `ai/data/flood/{train,val,test}` à partir de FloodNet (toujours sous CDLA-Permissive).
