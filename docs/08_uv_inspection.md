# Chapter 8: UV Inspection

## 8.1 Overview

UV inspection detects polymer fiber mixup (wrong material blended in) by analyzing UV fluorescence patterns. Different polymers fluoresce differently under UV light, creating concentric bands visible as a local dip in the radial log(G/B) profile.

**Source:** `src/inspection/uv_inspection.py` — `UVInspection` class

**No training required** — the algorithm is physics-based. Only a scalar threshold needs calibration.

## 8.2 Physics Rationale

- Pure yarn has a smooth, monotonic radial fluorescence profile under UV light
- Polymer mixup creates concentric fluorescence bands (different polymer fluoresces at different intensity)
- The log(G/B) ratio amplifies small fluorescence differences — sensitive to subtle bands
- A degree-2 polynomial baseline captures the natural radial gradient (not the defect bands)
- Maximum negative deviation from baseline = the "dip" metric

## 8.3 Algorithm

```
UV Frame (1920×1200)
        │
        ▼
YOLO (UV model) → yarn_cone bbox + yarn_tube bbox
        │
        ▼
Crop cone from bbox (clamp to frame bounds)
        │
        ▼
Derive geometry:
  - tube center (crop coordinates)
  - inner_r = tube radius
  - outer_r = cone radius × (1 - outer_margin)
        │
        ▼
Create radial distance map from tube center
        │
        ▼
Extract B and G channels, validate:
  - b > 5 (dark pixel guard)
  - g > 0 (log guard)
  - within annular region [inner_r, outer_r]
        │
        ▼
Check minimum valid pixels (< 100 = invalid frame)
        │
        ▼
Normalize radial distance to [0, 1]:
  0 = tube edge, 1 = outer cone edge
        │
        ▼
Compute per-pixel log(G/B) on valid pixels
        │
        ▼
Bin into 100 radial rings, compute mean log(G/B) per ring
        │
        ▼
Fit degree-2 polynomial baseline
        │
        ▼
max_dip = max(baseline - profile)  (positive = dip depth)
        │
        ▼
max_dip > radial_dip_threshold → DEFECT (has_mixup=True)
max_dip ≤ threshold → PASS (has_mixup=False)
```

## 8.4 NaN Guard

```python
valid_mask = (G > 0) & (B > 5)
```

Without this guard:
- `log(0)` = `-inf` propagates through `polyfit` → baseline becomes NaN
- NaN comparison always returns False → silent Good verdict on defective cones

The `B > 5` threshold filters dark pixels at cone edges that produce unreliable ratios.

## 8.5 Validation Results

Validated on 1950 good + 9 polymer-mixup images:

| Class | max_dip p1 | max_dip p99 |
|-------|-----------|------------|
| Good | — | 0.0195 |
| Defect | 0.0374 | — |

- Clean separation gap: +0.018 in log(G/B) domain
- Threshold 0.024 sits at the midpoint — validated with margin on both sides

## 8.6 Consecutive Detection Failure

If YOLO fails to detect cone or tube in the UV frame:

- `detection_failed=True` returned — UV check is **skipped** for this cone (not counted as Good or Defect)
- VL and Tail results still determine the final verdict
- Consecutive failure counter increments
- At 5 consecutive failures → `logger.error()` fires (likely camera/hardware issue, not a real defect)
- Counter resets on any successful detection

## 8.7 Result Fields

```python
@dataclass
class UVResult:
    has_mixup: bool              # max_dip > threshold
    radial_dip: float            # max negative dip value (monitoring)
    gb_ratio: float              # mean G/B ratio (monitoring)
    detection_failed: bool       # YOLO/compute failed
    cone_bbox: Optional[tuple]   # cone bbox in UV frame
```

## 8.8 Configuration

```json
{
    "uv_inspection": {
        "yolo_weights": "weights/uv_yolo.pt",
        "yolo_conf": 0.3,
        "radial_dip_threshold": 0.024,
        "outer_margin": 0.10
    }
}
```

| Key | Default | Description |
|-----|---------|-------------|
| `radial_dip_threshold` | 0.024 | Max allowed radial dip before flagging mixup |
| `outer_margin` | 0.10 | Fraction of radius to exclude at cone edge (noisy pixels) |
| `yolo_conf` | 0.3 | Lower than VL (UV images are noisier) |

### Constants (hardcoded)

| Constant | Value | Description |
|----------|-------|-------------|
| `RADIAL_BINS` | 100 | Number of radial rings from tube to outer edge |
| `_UV_DETECTION_FAIL_THRESHOLD` | 5 | Consecutive failures before operator alert |

## 8.9 Calibration

UV calibration is installation-only (no model training):

1. Run 10+ known-good cones, check `radial_dip` values via `GET /results`
2. Typical good cone dip: 0.003 – 0.015
3. If good cones all < 0.018 → default threshold 0.024 is correct
4. If good cones up to 0.020-0.022 → raise threshold slightly
5. Rule: `threshold = max(good_cone_dips) + 0.005`
6. `POST /teaching/uv` with new threshold

Recalibrate only if UV camera is replaced.
