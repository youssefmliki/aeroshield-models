# License notes — FloorPlan Model 5

## CubiCasa5K dataset

| Field | Value |
|-------|-------|
| Source | https://github.com/CubiCasa/CubiCasa5k · Zenodo 10.5281/zenodo.2613548 |
| Repo LICENSE | Creative Commons **Attribution-NonCommercial 4.0** (CC BY-NC 4.0) |
| Zenodo metadata | **cc-by-nc-sa-4.0** (NC + ShareAlike) |

**Implication:** Research / hackathon demo use is appropriate under NonCommercial terms. **Do not** ship CubiCasa images, SVGs, or derived masks into a commercial product without separate rights. Prefer citing the paper and linking Zenodo rather than redistributing the zip.

Always keep attribution: Kalervo et al., CubiCasa5K, SCIA 2019.

## floorplan-to-3d code + HF weights

| Field | Value |
|-------|-------|
| Code | MIT — https://github.com/Yytsi/floorplan-to-3d |
| Weights | MIT — https://huggingface.co/Yytsi/floorplan-to-3d-walls |

MIT allows reuse with copyright notice retained. Weights were trained on CubiCasa5K; **dataset NC terms still apply to data**. Shipping only model weights for research is common practice but does not grant CubiCasa redistribution rights.

## FireTwin / AeroShield integration

- Predictions = **DRAFT_UNVERIFIED** until an operator approves geometry/scale.
- Do not claim certified evacuation routing from unverified floorplan AI.
- Do not bundle CubiCasa raw data inside the HACK_INSAT submission zip if license review is incomplete; ship code + optional weights + attribution instead.
