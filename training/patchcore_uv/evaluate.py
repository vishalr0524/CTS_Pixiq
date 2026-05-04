"""
Evaluate UV PatchCore model on good and defect images.

Runs inference on each image and reports anomaly scores.
Defect images should score higher than good images.

Usage:
    uv run python training/patchcore_uv/evaluate.py
"""

import os
os.environ["TRUST_REMOTE_CODE"] = "1"

import cv2
import numpy as np
from pathlib import Path

from anomalib.deploy import TorchInferencer


def evaluate(model_path, good_dir, defect_dir=None):
    """Run PatchCore inference on good and defect images, report scores."""
    inferencer = TorchInferencer(path=str(model_path))

    def score_images(image_dir, label):
        images = sorted(image_dir.glob("*.png"))
        scores = []
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  SKIP (unreadable): {img_path.name}")
                continue
            # PatchCore expects RGB; UV unwrapped strips are grayscale → convert
            if len(img.shape) == 2 or img.shape[2] == 1:
                rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            prediction = inferencer.predict(rgb)
            score = float(prediction.pred_score.item() if hasattr(prediction.pred_score, 'item') else prediction.pred_score)
            scores.append(score)
            print(f"  {label:6s} | {score:8.4f} | {img_path.name}")
        return scores

    print(f"Model: {model_path}")
    print(f"{'':=<60}")

    # Good images
    print(f"\n--- Good images ({good_dir}) ---")
    good_scores = score_images(good_dir, "GOOD")

    # Defect images
    defect_scores = []
    if defect_dir and defect_dir.exists():
        defect_files = list(defect_dir.glob("*.png"))
        if defect_files:
            print(f"\n--- Defect images ({defect_dir}) ---")
            defect_scores = score_images(defect_dir, "DEFECT")

    # Summary
    print(f"\n{'':=<60}")
    print("SUMMARY")
    print(f"{'':=<60}")
    if good_scores:
        print(f"  Good:   n={len(good_scores):3d}  mean={np.mean(good_scores):.4f}  "
              f"min={np.min(good_scores):.4f}  max={np.max(good_scores):.4f}  "
              f"std={np.std(good_scores):.4f}")
    if defect_scores:
        print(f"  Defect: n={len(defect_scores):3d}  mean={np.mean(defect_scores):.4f}  "
              f"min={np.min(defect_scores):.4f}  max={np.max(defect_scores):.4f}  "
              f"std={np.std(defect_scores):.4f}")

    if good_scores and defect_scores:
        good_max = np.max(good_scores)
        defect_min = np.min(defect_scores)
        margin = defect_min - good_max
        print(f"\n  Margin (defect_min - good_max): {margin:.4f}")
        if margin > 0:
            threshold = (good_max + defect_min) / 2
            print(f"  Classes separable! Suggested threshold: {threshold:.4f}")
            # Check accuracy at this threshold
            tp = sum(1 for s in defect_scores if s > threshold)
            tn = sum(1 for s in good_scores if s <= threshold)
            acc = (tp + tn) / (len(good_scores) + len(defect_scores))
            print(f"  Accuracy at threshold={threshold:.4f}: {acc:.1%} (TP={tp}/{len(defect_scores)}, TN={tn}/{len(good_scores)})")
        else:
            print(f"  WARNING: Classes overlap (margin={margin:.4f})")
            # Find best threshold anyway
            all_scores = [(s, 0) for s in good_scores] + [(s, 1) for s in defect_scores]
            all_scores.sort()
            best_acc, best_thresh = 0, 0
            for i in range(len(all_scores) - 1):
                t = (all_scores[i][0] + all_scores[i+1][0]) / 2
                tp = sum(1 for s in defect_scores if s > t)
                tn = sum(1 for s in good_scores if s <= t)
                acc = (tp + tn) / (len(good_scores) + len(defect_scores))
                if acc > best_acc:
                    best_acc = acc
                    best_thresh = t
            print(f"  Best threshold: {best_thresh:.4f} (accuracy={best_acc:.1%})")


if __name__ == "__main__":
    model_path = Path("models/patchcore_uv/weights/torch/model.pt")
    good_dir = Path("training/patchcore_uv/dataset/good")
    defect_dir = Path("training/patchcore/training/patchcore_uv/dataset/defect")

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        exit(1)

    evaluate(model_path, good_dir, defect_dir)
