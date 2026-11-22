# Dataset Audit — UCI Gas Sensor Array Drift

- **Source:** UCI Gas Sensor Array Drift Dataset (UCI ID 224)
- **URL:** https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset
- **Citation:** Vergara, A. (2012). Gas Sensor Array Drift Dataset. UCI ML Repository. https://doi.org/10.24432/C5RP6W

## Summary

- Raw rows (sum of batches): **13910**
- Kept rows after validity filters: **13910**
- Excluded rows: **0**
- Features per sample: **128** (expected 128)
- Feature count consistent across batches: **True**

## Class mapping (source labels, 1-based)

| Class ID | Gas |
|----------|-----|
| 1 | Ethanol |
| 2 | Ethylene |
| 3 | Ammonia |
| 4 | Acetaldehyde |
| 5 | Acetone |
| 6 | Toluene |

## Global class distribution

| Gas | Count |
|-----|------:|
| Ethanol | 2565 |
| Ethylene | 2926 |
| Ammonia | 1641 |
| Acetaldehyde | 1936 |
| Acetone | 3009 |
| Toluene | 1833 |

## Batch distribution

| batch | n_samples | Ethanol | Ethylene | Ammonia | Acetaldehyde | Acetone | Toluene |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 445 | 90 | 98 | 83 | 30 | 70 | 74 |
| 2 | 1244 | 164 | 334 | 100 | 109 | 532 | 5 |
| 3 | 1586 | 365 | 490 | 216 | 240 | 275 | 0 |
| 4 | 161 | 64 | 43 | 12 | 30 | 12 | 0 |
| 5 | 197 | 28 | 40 | 20 | 46 | 63 | 0 |
| 6 | 2300 | 514 | 574 | 110 | 29 | 606 | 467 |
| 7 | 3613 | 649 | 662 | 360 | 744 | 630 | 568 |
| 8 | 294 | 30 | 30 | 40 | 33 | 143 | 18 |
| 9 | 470 | 61 | 55 | 100 | 75 | 78 | 101 |
| 10 | 3600 | 600 | 600 | 600 | 600 | 600 | 600 |

## Missing / invalid values

- No NaN retained after filtering; non-finite rows excluded.
- Rows with NaN/Inf excluded before save.

No rows were excluded.

## Per-batch detail

### batch1

- Raw rows: 445
- Kept rows: 445
- Excluded: 0
- Features: 128
- Has NaN in raw: False
- Has Inf in raw: False
- Unique labels (raw): [1, 2, 3, 4, 5, 6]
- Class counts (kept): {'Ethanol': 90, 'Ethylene': 98, 'Ammonia': 83, 'Acetaldehyde': 30, 'Acetone': 70, 'Toluene': 74}

### batch2

- Raw rows: 1244
- Kept rows: 1244
- Excluded: 0
- Features: 128
- Has NaN in raw: False
- Has Inf in raw: False
- Unique labels (raw): [1, 2, 3, 4, 5, 6]
- Class counts (kept): {'Ethanol': 164, 'Ethylene': 334, 'Ammonia': 100, 'Acetaldehyde': 109, 'Acetone': 532, 'Toluene': 5}

### batch3

- Raw rows: 1586
- Kept rows: 1586
- Excluded: 0
- Features: 128
- Has NaN in raw: False
- Has Inf in raw: False
- Unique labels (raw): [1, 2, 3, 4, 5]
- Class counts (kept): {'Ethanol': 365, 'Ethylene': 490, 'Ammonia': 216, 'Acetaldehyde': 240, 'Acetone': 275, 'Toluene': 0}

### batch4

- Raw rows: 161
- Kept rows: 161
- Excluded: 0
- Features: 128
- Has NaN in raw: False
- Has Inf in raw: False
- Unique labels (raw): [1, 2, 3, 4, 5]
- Class counts (kept): {'Ethanol': 64, 'Ethylene': 43, 'Ammonia': 12, 'Acetaldehyde': 30, 'Acetone': 12, 'Toluene': 0}

### batch5

- Raw rows: 197
- Kept rows: 197
- Excluded: 0
- Features: 128
- Has NaN in raw: False
- Has Inf in raw: False
- Unique labels (raw): [1, 2, 3, 4, 5]
- Class counts (kept): {'Ethanol': 28, 'Ethylene': 40, 'Ammonia': 20, 'Acetaldehyde': 46, 'Acetone': 63, 'Toluene': 0}

### batch6

- Raw rows: 2300
- Kept rows: 2300
- Excluded: 0
- Features: 128
- Has NaN in raw: False
- Has Inf in raw: False
- Unique labels (raw): [1, 2, 3, 4, 5, 6]
- Class counts (kept): {'Ethanol': 514, 'Ethylene': 574, 'Ammonia': 110, 'Acetaldehyde': 29, 'Acetone': 606, 'Toluene': 467}

### batch7

- Raw rows: 3613
- Kept rows: 3613
- Excluded: 0
- Features: 128
- Has NaN in raw: False
- Has Inf in raw: False
- Unique labels (raw): [1, 2, 3, 4, 5, 6]
- Class counts (kept): {'Ethanol': 649, 'Ethylene': 662, 'Ammonia': 360, 'Acetaldehyde': 744, 'Acetone': 630, 'Toluene': 568}

### batch8

- Raw rows: 294
- Kept rows: 294
- Excluded: 0
- Features: 128
- Has NaN in raw: False
- Has Inf in raw: False
- Unique labels (raw): [1, 2, 3, 4, 5, 6]
- Class counts (kept): {'Ethanol': 30, 'Ethylene': 30, 'Ammonia': 40, 'Acetaldehyde': 33, 'Acetone': 143, 'Toluene': 18}

### batch9

- Raw rows: 470
- Kept rows: 470
- Excluded: 0
- Features: 128
- Has NaN in raw: False
- Has Inf in raw: False
- Unique labels (raw): [1, 2, 3, 4, 5, 6]
- Class counts (kept): {'Ethanol': 61, 'Ethylene': 55, 'Ammonia': 100, 'Acetaldehyde': 75, 'Acetone': 78, 'Toluene': 101}

### batch10

- Raw rows: 3600
- Kept rows: 3600
- Excluded: 0
- Features: 128
- Has NaN in raw: False
- Has Inf in raw: False
- Unique labels (raw): [1, 2, 3, 4, 5, 6]
- Class counts (kept): {'Ethanol': 600, 'Ethylene': 600, 'Ammonia': 600, 'Acetaldehyde': 600, 'Acetone': 600, 'Toluene': 600}

## Evaluation caution

Batches were NOT randomly mixed. Train/test batch separation is defined in scripts/train.py.

Target class and batch ID are **not** used as input features.
