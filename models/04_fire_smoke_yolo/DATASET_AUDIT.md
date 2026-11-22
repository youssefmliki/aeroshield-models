# D-Fire Dataset Audit Report

## Source
- Official: https://github.com/gaia-solutions-on-demand/DFireDataset
- Mirror used: https://huggingface.co/datasets/hanvithSai/gavin-dfire-raw
- License: **CC BY 4.0**
- Attribution: Pedro Vinícius Almeida Borges de Venâncio, Adriano Chaves Lisboa, Adriano Vilela Barbosa (Neural Computing and Applications, 2022)

## Verified class mapping
| class_id | class_name |
|----------|------------|
| 0 | smoke |
| 1 | fire |

Source of truth: HF mirror README (`Raw label convention as shipped by the original authors: 0=smoke, 1=fire`) + label inspection during this audit.

## Counts
| Split | Images |
|-------|--------|
| test | 4306 |
| train | 17221 |
| **Total** | **21527** |

### Images by category
| Category | Count |
|----------|-------|
| Only fire | 1164 |
| Only smoke | 5857 |
| Fire and smoke | 4636 |
| Neither (negatives) | 9838 |

### Bounding boxes per class
| Class | Boxes |
|-------|-------|
| fire | 14607 |
| smoke | 11816 |

## Quality checks
| Check | Result |
|-------|--------|
| Readable images | 21527 |
| Unreadable images | 0 |
| Missing labels | 0 |
| Invalid labels | 32 |
| Empty labels (negatives) | 9838 |
| Exact duplicate extras | 0 |
| Duplicate hash groups | 0 |
| Cross-split exact-hash groups | 0 |

Empty labels are **valid** (negative examples: sunlight, lamps, steam, clouds without fire/smoke).

## Final usable dataset size
- Usable paired samples: **21495**
- Invalid items excluded: **32**
- Annotated visual examples saved: **12** under `reports/audit_examples/`

## Notes
- Official train/test partitions preserved when present.
- Near-duplicate check used content MD5 of first 1MB + file size (practical proxy; not perceptual hash).
- Do not fabricate labels. Negatives retained intentionally to reduce false alarms.
