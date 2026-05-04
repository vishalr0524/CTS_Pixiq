"""
Analyze PatchCore anomaly score spread on test images.

Uses the same scoring method as inference (stain_detector.py):
    - Raw anomaly heatmap max within the annular mask region
    - NOT anomalib's normalized pred_score

This gives the true score distribution to set the threshold.

Usage:
    uv run python training/patchcore/score_spread.py
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np

os.environ["TRUST_REMOTE_CODE"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))


def create_annular_mask(shape, center, inner_r, outer_r):
    """Create donut mask (same as stain_detector.py)."""
    h, w = shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = (dist >= inner_r) & (dist <= outer_r)
    return mask.astype(np.uint8) * 255


def main():
    model_path = Path("training/patchcore/results/Patchcore/cone_surface/exported/torch/weights/torch/model.pt")
    test_dir = Path("training/patchcore/dataset/test/good")

    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    # Load model
    from anomalib.deploy import TorchInferencer
    print(f"Loading model: {model_path}")
    engine = TorchInferencer(path=str(model_path))

    # We also need YOLO + geometry to create annular masks for each image
    # But test images are already donut-masked crops, so pixels that are
    # black (outside mask) will have low anomaly scores.
    # For the spread analysis, we use:
    #   1. normalized pred_score (anomalib's internal)
    #   2. raw max(anomaly_map) over non-zero pixels (our inference method)

    images = sorted(test_dir.glob("*.png"))
    print(f"Test images: {len(images)}")
    print()

    results = []
    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Create mask from non-zero pixels (donut region)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        donut_mask = (gray > 5).astype(np.uint8) * 255  # non-black pixels

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        prediction = engine.predict(rgb)

        # Method 1: anomalib normalized score
        score_val = prediction.pred_score
        norm_score = float(score_val.item() if hasattr(score_val, 'item') else score_val)

        # Method 2: raw max(heatmap) within donut (our inference method)
        raw_map = prediction.anomaly_map
        if raw_map is not None:
            if hasattr(raw_map, 'cpu'):
                raw_map = raw_map.cpu().numpy()
            if raw_map.ndim == 3:
                raw_map = raw_map[0]
            # Resize to image size
            heatmap = cv2.resize(raw_map, (img.shape[1], img.shape[0]))

            # Score only within donut (non-black pixels)
            donut_scores = heatmap[donut_mask > 0]
            if len(donut_scores) > 0:
                raw_max = float(np.max(donut_scores))
                raw_mean = float(np.mean(donut_scores))
                raw_p99 = float(np.percentile(donut_scores, 99))
            else:
                raw_max = raw_mean = raw_p99 = 0.0
        else:
            raw_max = raw_mean = raw_p99 = 0.0

        results.append({
            "name": img_path.name,
            "norm_score": norm_score,
            "raw_max": raw_max,
            "raw_mean": raw_mean,
            "raw_p99": raw_p99,
        })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(images)}] processed...")

    print(f"\n{'='*60}")
    print(f"Score Spread — {len(results)} GOOD test images")
    print(f"{'='*60}")

    # Raw max scores (our inference method)
    raw_maxes = np.array([r["raw_max"] for r in results])
    raw_means = np.array([r["raw_mean"] for r in results])
    raw_p99s = np.array([r["raw_p99"] for r in results])
    norm_scores = np.array([r["norm_score"] for r in results])

    print(f"\n--- Raw heatmap MAX within donut (our inference scoring) ---")
    print(f"  Min:    {raw_maxes.min():.4f}")
    print(f"  Max:    {raw_maxes.max():.4f}")
    print(f"  Mean:   {raw_maxes.mean():.4f}")
    print(f"  Std:    {raw_maxes.std():.4f}")
    print(f"  Median: {np.median(raw_maxes):.4f}")
    print(f"  P95:    {np.percentile(raw_maxes, 95):.4f}")
    print(f"  P99:    {np.percentile(raw_maxes, 99):.4f}")

    print(f"\n--- Raw heatmap P99 within donut ---")
    print(f"  Min:    {raw_p99s.min():.4f}")
    print(f"  Max:    {raw_p99s.max():.4f}")
    print(f"  Mean:   {raw_p99s.mean():.4f}")

    print(f"\n--- Raw heatmap MEAN within donut ---")
    print(f"  Min:    {raw_means.min():.4f}")
    print(f"  Max:    {raw_means.max():.4f}")
    print(f"  Mean:   {raw_means.mean():.4f}")

    print(f"\n--- Anomalib normalized score (for reference) ---")
    print(f"  Min:    {norm_scores.min():.4f}")
    print(f"  Max:    {norm_scores.max():.4f}")
    print(f"  Mean:   {norm_scores.mean():.4f}")

    # Top 10 highest raw_max scores
    sorted_results = sorted(results, key=lambda x: x["raw_max"], reverse=True)
    print(f"\nTop 10 highest raw_max GOOD images:")
    for r in sorted_results[:10]:
        print(f"  raw_max={r['raw_max']:.4f}  raw_mean={r['raw_mean']:.4f}  norm={r['norm_score']:.4f}  {r['name']}")

    print(f"\nBottom 5 lowest raw_max GOOD images:")
    for r in sorted_results[-5:]:
        print(f"  raw_max={r['raw_max']:.4f}  raw_mean={r['raw_mean']:.4f}  norm={r['norm_score']:.4f}  {r['name']}")

    # Threshold suggestion based on raw_max
    print(f"\nThreshold suggestions (based on raw_max of GOOD images):")
    max_good = raw_maxes.max()
    p99_good = np.percentile(raw_maxes, 99)
    p95_good = np.percentile(raw_maxes, 95)
    print(f"  Conservative (max + 20%):  {max_good * 1.2:.4f}")
    print(f"  Moderate (max + 10%):      {max_good * 1.1:.4f}")
    print(f"  Aggressive (P95 + 10%):    {p95_good * 1.1:.4f}")


if __name__ == "__main__":
    main()
