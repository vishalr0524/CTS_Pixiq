"""
Visualize PatchCore UV anomaly heatmaps.

For each image runs inference and saves a side-by-side panel:
    [Original crop] | [Heatmap overlay] | [Heatmap only]

Saves results to training/patchcore_uv/results/heatmaps/{good,defect}/

Usage:
    python training/patchcore_uv/visualize_heatmaps.py
    python training/patchcore_uv/visualize_heatmaps.py --good-n 20
"""

import argparse
import os
import random
os.environ["TRUST_REMOTE_CODE"] = "1"

import cv2
import numpy as np
from pathlib import Path


MODEL_PATH  = "training/patchcore_uv/results/Patchcore/uv_cone/exported/torch/weights/torch/model.pt"
GOOD_DIR    = "training/patchcore_uv/dataset/good"
DEFECT_DIR  = "training/patchcore_uv/dataset/defect"
OUT_DIR     = "training/patchcore_uv/results/heatmaps"


def make_heatmap_overlay(image_bgr: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Blend anomaly heatmap over the original image.

    Args:
        image_bgr: Original BGR image.
        heatmap: Anomaly map (H, W) float32, values 0–1.

    Returns:
        BGR overlay image.
    """
    # Normalize heatmap to 0-255
    h_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    h_uint8 = (h_norm * 255).astype(np.uint8)

    # Apply JET colormap
    h_color = cv2.applyColorMap(h_uint8, cv2.COLORMAP_JET)
    h_color = cv2.resize(h_color, (image_bgr.shape[1], image_bgr.shape[0]))

    # Blend with original
    overlay = cv2.addWeighted(image_bgr, 0.5, h_color, 0.5, 0)
    return overlay, h_color


def make_panel(original: np.ndarray, overlay: np.ndarray, heatmap_color: np.ndarray,
               score: float, label: str, filename: str) -> np.ndarray:
    """Create a 3-panel visualization with score label."""
    h, w = original.shape[:2]

    # Resize all panels to same height
    target_h = 400
    scale = target_h / h
    target_w = int(w * scale)

    orig_r   = cv2.resize(original, (target_w, target_h))
    over_r   = cv2.resize(overlay, (target_w, target_h))
    heat_r   = cv2.resize(heatmap_color, (target_w, target_h))

    # Add label bar on top of each panel
    def add_label(img, text, color):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (target_w, 30), (0, 0, 0), -1)
        cv2.putText(out, text, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        return out

    color = (0, 255, 0) if label == "GOOD" else (0, 0, 255)
    p1 = add_label(orig_r,  f"Original  [{label}]", color)
    p2 = add_label(over_r,  f"Overlay   score={score:.3f}", color)
    p3 = add_label(heat_r,  f"Heatmap   {filename}", (255, 255, 255))

    # Horizontal concat with separator
    sep = np.zeros((target_h, 4, 3), dtype=np.uint8)
    panel = np.hstack([p1, sep, p2, sep, p3])
    return panel


def run_inference_and_save(
    image_paths: list,
    inferencer,
    output_dir: Path,
    label: str,
) -> list[float]:
    """Run inference on images, save heatmap panels, return scores."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = []

    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  SKIP (unreadable): {img_path.name}")
            continue

        # PatchCore expects RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        prediction = inferencer.predict(img_rgb)

        score = float(
            prediction.pred_score.item()
            if hasattr(prediction.pred_score, "item")
            else prediction.pred_score
        )
        scores.append(score)

        # Get anomaly map
        if hasattr(prediction, "anomaly_map") and prediction.anomaly_map is not None:
            amap = prediction.anomaly_map
            if hasattr(amap, "numpy"):
                amap = amap.cpu().numpy() if hasattr(amap, "cpu") else amap.numpy()
            amap = np.squeeze(amap).astype(np.float32)
        else:
            # Fallback: blank heatmap
            amap = np.zeros(img_bgr.shape[:2], dtype=np.float32)

        overlay, heatmap_color = make_heatmap_overlay(img_bgr, amap)
        panel = make_panel(img_bgr, overlay, heatmap_color, score, label, img_path.name)

        out_path = output_dir / f"{img_path.stem}_heatmap.jpg"
        cv2.imwrite(str(out_path), panel)
        print(f"  {label:6s} | score={score:.4f} | {img_path.name} → {out_path.name}")

    return scores


def main():
    parser = argparse.ArgumentParser(description="Visualize UV PatchCore heatmaps")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--good-dir", default=GOOD_DIR)
    parser.add_argument("--defect-dir", default=DEFECT_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--good-n", type=int, default=20,
                        help="Number of good images to sample for visualization (default: 20)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    model_path  = project_root / args.model
    good_dir    = project_root / args.good_dir
    defect_dir  = project_root / args.defect_dir
    out_dir     = project_root / args.out_dir

    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return

    print(f"Loading model: {model_path}")
    from anomalib.deploy import TorchInferencer
    inferencer = TorchInferencer(path=str(model_path))

    # Sample good images for visualization
    good_images = sorted(good_dir.glob("*.png"))
    if args.good_n < len(good_images):
        random.seed(42)
        good_images = random.sample(good_images, args.good_n)
        good_images = sorted(good_images)

    print(f"\nGood images ({len(good_images)} sampled):")
    good_scores = run_inference_and_save(good_images, inferencer, out_dir / "good", "GOOD")

    # All defect images
    defect_images = sorted(defect_dir.glob("*.png")) if defect_dir.exists() else []
    defect_scores = []
    if defect_images:
        print(f"\nDefect images ({len(defect_images)}):")
        defect_scores = run_inference_and_save(defect_images, inferencer, out_dir / "defect", "DEFECT")

    # Summary
    print(f"\n{'='*55}")
    if good_scores:
        print(f"Good   n={len(good_scores):3d}  mean={np.mean(good_scores):.4f}  "
              f"min={np.min(good_scores):.4f}  max={np.max(good_scores):.4f}")
    if defect_scores:
        print(f"Defect n={len(defect_scores):3d}  mean={np.mean(defect_scores):.4f}  "
              f"min={np.min(defect_scores):.4f}  max={np.max(defect_scores):.4f}")
    if good_scores and defect_scores:
        margin = np.min(defect_scores) - np.max(good_scores)
        print(f"Margin (defect_min - good_max): {margin:.4f}")
        threshold = (np.max(good_scores) + np.min(defect_scores)) / 2
        print(f"Suggested threshold: {threshold:.4f}")

    print(f"\nHeatmaps saved to: {out_dir}")


if __name__ == "__main__":
    main()
