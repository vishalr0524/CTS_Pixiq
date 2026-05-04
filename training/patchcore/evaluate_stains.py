"""
Evaluate stain images: YOLO detect → annular crop → PatchCore score.

Processes full-frame stain images through the same pipeline as inference:
1. YOLO detect cone + tube
2. Crop cone bbox
3. Apply annular mask (donut)
4. Run PatchCore → raw heatmap max within donut

Saves intermediate images for visual inspection.

Usage:
    uv run python training/patchcore/evaluate_stains.py
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np

os.environ["TRUST_REMOTE_CODE"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from inspection.yolo_detector import YOLODetector


def find_geometry(cone_bbox, tube_bbox):
    """Derive cone geometry from YOLO bboxes."""
    cx1, cy1, cx2, cy2 = cone_bbox
    tx1, ty1, tx2, ty2 = tube_bbox

    tube_cx = (tx1 + tx2) / 2 - cx1
    tube_cy = (ty1 + ty2) / 2 - cy1
    center = (int(tube_cx), int(tube_cy))

    tube_w = tx2 - tx1
    tube_h = ty2 - ty1
    inner_r = float(min(tube_w, tube_h)) / 2

    cone_w = cx2 - cx1
    cone_h = cy2 - cy1
    outer_r = float(min(cone_w, cone_h)) / 2

    return center, inner_r, outer_r


def create_annular_mask(shape, center, inner_r, outer_r):
    """Create donut mask."""
    h, w = shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = (dist >= inner_r) & (dist <= outer_r)
    return mask.astype(np.uint8) * 255


def overlay_heatmap(image, heatmap, mask=None, alpha=0.5):
    """Overlay anomaly heatmap on image within mask region."""
    h, w = image.shape[:2]
    hm = cv2.resize(heatmap, (w, h))

    # Normalize
    hm_min, hm_max = hm.min(), hm.max()
    if hm_max > hm_min:
        hm = (hm - hm_min) / (hm_max - hm_min)
    else:
        hm = np.zeros_like(hm)

    hm_uint8 = (np.clip(hm, 0, 1) * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)

    blended = image.copy()
    if mask is not None:
        region = mask > 0
        blended[region] = cv2.addWeighted(image, 1 - alpha, hm_color, alpha, 0)[region]
    else:
        blended = cv2.addWeighted(image, 1 - alpha, hm_color, alpha, 0)

    return blended


def main():
    stain_dir = Path("training/patchcore/stain_images")
    model_path = Path("training/patchcore/results/Patchcore/cone_surface/exported/torch/weights/torch/model.pt")
    output_dir = Path("training/patchcore/stain_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLO
    print("Loading YOLO...")
    detector = YOLODetector("weights/visible_yolo.pt", conf_threshold=0.6)

    # Load PatchCore
    print(f"Loading PatchCore: {model_path}")
    from anomalib.deploy import TorchInferencer
    engine = TorchInferencer(path=str(model_path))

    images = sorted(stain_dir.glob("*.png"))
    print(f"\nProcessing {len(images)} stain images...")
    print()

    results = []
    for img_path in images:
        name = img_path.stem
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  SKIP: cannot read {img_path.name}")
            continue

        # YOLO detect
        detections = detector.detect(frame)
        cone_det = detector.get_detection_by_class(detections, "yarn_cone")
        tube_det = detector.get_detection_by_class(detections, "yarn_tube")

        if cone_det is None or tube_det is None:
            missing = "cone" if cone_det is None else "tube"
            print(f"  {name}: SKIP (no {missing} detected)")
            continue

        # Crop cone
        cone_crop = detector.extract_roi(frame, cone_det)

        # Geometry
        center, inner_r, outer_r = find_geometry(cone_det.bbox, tube_det.bbox)

        # Annular mask
        mask = create_annular_mask(cone_crop.shape, center, inner_r, outer_r)

        # Apply mask to input (same as training + inference)
        donut = cone_crop.copy()
        donut[mask == 0] = 0

        # Run PatchCore
        rgb = cv2.cvtColor(donut, cv2.COLOR_BGR2RGB)
        prediction = engine.predict(rgb)

        # Get raw heatmap
        raw_map = prediction.anomaly_map
        if raw_map is not None:
            if hasattr(raw_map, 'cpu'):
                raw_map = raw_map.cpu().numpy()
            if raw_map.ndim == 3:
                raw_map = raw_map[0]
            heatmap = cv2.resize(raw_map, (cone_crop.shape[1], cone_crop.shape[0]))

            donut_scores = heatmap[mask > 0]
            raw_max = float(np.max(donut_scores)) if len(donut_scores) > 0 else 0.0
            raw_mean = float(np.mean(donut_scores)) if len(donut_scores) > 0 else 0.0
            raw_p99 = float(np.percentile(donut_scores, 99)) if len(donut_scores) > 0 else 0.0
        else:
            raw_max = raw_mean = raw_p99 = 0.0
            heatmap = np.zeros(cone_crop.shape[:2], dtype=np.float32)

        norm_score = float(prediction.pred_score.item() if hasattr(prediction.pred_score, 'item') else prediction.pred_score)

        results.append({
            "name": name,
            "raw_max": raw_max,
            "raw_mean": raw_mean,
            "raw_p99": raw_p99,
            "norm_score": norm_score,
            "cone_size": f"{cone_crop.shape[1]}x{cone_crop.shape[0]}",
            "center": center,
            "inner_r": inner_r,
            "outer_r": outer_r,
        })

        print(f"  {name}: raw_max={raw_max:.4f}  raw_mean={raw_mean:.4f}  norm={norm_score:.4f}  "
              f"cone={cone_crop.shape[1]}x{cone_crop.shape[0]}  center={center}  r={inner_r:.0f}/{outer_r:.0f}")

        # Save intermediate images
        cv2.imwrite(str(output_dir / f"{name}_1_cone_crop.jpg"), cone_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(output_dir / f"{name}_2_donut.jpg"), donut, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Heatmap overlay on donut
        heatmap_vis = overlay_heatmap(cone_crop, heatmap, mask=mask, alpha=0.6)
        cv2.imwrite(str(output_dir / f"{name}_3_heatmap.jpg"), heatmap_vis, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Draw geometry circles
        debug = cone_crop.copy()
        cv2.circle(debug, center, int(inner_r), (0, 0, 255), 2)
        cv2.circle(debug, center, int(outer_r), (0, 255, 0), 2)
        cv2.circle(debug, center, 5, (255, 0, 0), -1)
        cv2.imwrite(str(output_dir / f"{name}_4_geometry.jpg"), debug, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Summary
    print(f"\n{'='*60}")
    print(f"STAIN Score Summary — {len(results)} images")
    print(f"{'='*60}")

    if results:
        raw_maxes = np.array([r["raw_max"] for r in results])
        print(f"\n  Raw MAX (our scoring):")
        print(f"    Min:  {raw_maxes.min():.4f}")
        print(f"    Max:  {raw_maxes.max():.4f}")
        print(f"    Mean: {raw_maxes.mean():.4f}")

        print(f"\n  Comparison with GOOD test images:")
        print(f"    GOOD range:  0.708 - 1.000 (mean 0.801)")
        print(f"    STAIN range: {raw_maxes.min():.3f} - {raw_maxes.max():.3f} (mean {raw_maxes.mean():.3f})")

        gap = raw_maxes.min() - 1.000
        if gap > 0:
            print(f"\n    GAP between max GOOD and min STAIN: {gap:.3f} — CLEAN SEPARATION!")
        else:
            overlap = 1.000 - raw_maxes.min()
            print(f"\n    OVERLAP: stain scores overlap with good scores by {overlap:.3f}")
            print(f"    Some stains may be indistinguishable from normal images.")

    print(f"\nSaved to: {output_dir}/")


if __name__ == "__main__":
    main()
