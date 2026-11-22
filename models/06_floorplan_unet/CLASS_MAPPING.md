# Exact class mapping

Aligned with **Yytsi/floorplan-to-3d** (checkpoint-compatible). Do not remap silently.

| ID | Name | Source SVG tokens | Notes |
|----|------|-------------------|-------|
| 0 | floor | *(default fill)* | All pixels start as floor; rooms/furniture/text residual |
| 1 | wall | `Wall` | Painted first among structure |
| 2 | door | `Door` | Overwrites wall where nested |
| 3 | window | `Window` | Overwrites wall/door paint order last |

**Not used (would break HF checkpoint):** separate `background` class, CubiCasa room taxonomy, railing-as-own-class, furniture icons.

## FireTwin operator contract

```json
{
  "status": "DRAFT_UNVERIFIED",
  "classes": ["floor", "wall", "door", "window"],
  "scale_m_per_px": null,
  "approved": false
}
```

Only after manual scale calibration + geometry edit + operator approval may status become `APPROVED_FOR_TWIN` and feed 2D/3D generation.
