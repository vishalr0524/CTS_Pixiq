"""
Evaluate a trained PatchCore model on test images.

Can process either:
    - Pre-unwrapped texture strips (direct evaluation)
    - Full-frame images (YOLO detect → crop → unwarp → evaluate → inverse warp heatmap)

Usage:
    # Evaluate on a pre-unwrapped texture
    uv run python training/patchcore/evaluate.py \\
        --model training/patchcore/results/Patchcore/cone_surface/exported/torch \\
        --image training/patchcore/dataset/good/3536.png

    # Evaluate on a full-frame image (uses YOLO + unwarp)
    uv run python training/patchcore/evaluate.py \\
        --model training/patchcore/results/Patchcore/cone_surface/exported/torch \\
        --image data/visible/good/3536.png \\
        --full-frame

    # Batch evaluate a directory
    uv run python training/patchcore/evaluate.py \\
        --model training/patchcore/results/Patchcore/cone_surface/exported/torch \\
        --input-dir data/visible/stain \\
        --full-frame
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from unwarp import find_geometry, unwarp_cone, inverse_unwarp, draw_debug


def load_inferencer(model_path: str):
    """Load the anomalib inferencer from a model directory.

    Tries Torch first, then OpenVINO.

    Args:
        model_path: Path to the exported model directory (contains model.pt or model.xml).

    Returns:
        anomalib inferencer instance.
    """
    from anomalib.deploy import OpenVINOInferencer, TorchInferencer

    model_dir = Path(model_path)

    # Check for model files
    torch_model = model_dir / "model.pt"
    ov_model = model_dir / "model.xml"
    metadata = model_dir / "metadata.json"

    if torch_model.exists():
        print(f"Loading Torch model: {torch_model}")
        return TorchInferencer(
            path=str(torch_model),
            metadata=str(metadata) if metadata.exists() else None,
        )
    elif ov_model.exists():
        print(f"Loading OpenVINO model: {ov_model}")
        return OpenVINOInferencer(
            path=str(ov_model),
            metadata=str(metadata) if metadata.exists() else None,
        )
    else:
        print(f"ERROR: No model found in {model_dir}")
        print(f"  Expected: model.pt (Torch) or model.xml (OpenVINO)")
        sys.exit(1)


def predict_texture(inferencer, texture_bgr: np.ndarray) -> tuple:
    """Run PatchCore inference on an unwrapped texture strip.

    Args:
        inferencer: anomalib inferencer.
        texture_bgr: BGR texture strip image.

    Returns:
        (anomaly_score, anomaly_map) where anomaly_map is HxW float32.
    """
    texture_rgb = cv2.cvtColor(texture_bgr, cv2.COLOR_BGR2RGB)
    prediction = inferencer.predict(texture_rgb)

    score = float(prediction.pred_score)
    anomaly_map = prediction.anomaly_map  # HxW float

    return score, anomaly_map


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay anomaly heatmap onto an image.

    Args:
        image: BGR image.
        heatmap: Float anomaly map (0=normal, higher=anomaly).
        alpha: Blend factor.

    Returns:
        BGR image with heatmap overlay.
    """
    h, w = image.shape[:2]
    hm = cv2.resize(heatmap, (w, h))

    # Normalize to 0-1
    hm_min, hm_max = hm.min(), hm.max()
    if hm_max > hm_min:
        hm = (hm - hm_min) / (hm_max - hm_min)
    else:
        hm = np.zeros_like(hm)

    hm_uint8 = (np.clip(hm, 0, 1) * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)

    # Blend where anomaly is significant
    mask = hm > 0.3
    blended = image.copy()
    if mask.any():
        blended[mask] = cv2.addWeighted(image, 1 - alpha, hm_color, alpha, 0)[mask]

    return blended


def evaluate_single(
    inferencer,
    image_path: str,
    full_frame: bool,
    yolo_detector=None,
    angular_res: int = 1024,
    threshold: float = 0.5,
) -> dict:
    """Evaluate a single image and return results + visualizations.

    Args:
        inferencer: anomalib inferencer.
        image_path: Path to the image.
        full_frame: If True, run YOLO + unwarp first.
        yolo_detector: YOLODetector instance (required if full_frame=True).
        angular_res: Angular resolution for unwarp.
        threshold: Anomaly score threshold.

    Returns:
        Dict with keys: score, has_stain, texture, heatmap_on_texture,
        and optionally: cone_crop, heatmap_on_cone, debug_geometry.
    """
    image = cv2.imread(image_path)
    if image is None:
        return {"error": f"Cannot read {image_path}"}

    result = {}

    if full_frame:
        # YOLO detect → crop → unwarp
        detections = yolo_detector.detect(image)
        cone_det = yolo_detector.get_detection_by_class(detections, "yarn_cone")
        tube_det = yolo_detector.get_detection_by_class(detections, "yarn_tube")
        if cone_det is None:
            return {"error": "No yarn_cone detected"}
        if tube_det is None:
            return {"error": "No yarn_tube detected"}

        cone_crop = yolo_detector.extract_roi(image, cone_det)
        center, inner_r, outer_r = find_geometry(cone_det.bbox, tube_det.bbox)
        texture = unwarp_cone(cone_crop, center, inner_r, outer_r, angular_res)
        result["cone_crop"] = cone_crop
        result["geometry"] = (center, inner_r, outer_r)
        result["debug_geometry"] = draw_debug(cone_crop, center, inner_r, outer_r)
    else:
        texture = image

    # Run PatchCore inference
    score, anomaly_map = predict_texture(inferencer, texture)
    has_stain = score > threshold

    result["score"] = score
    result["has_stain"] = has_stain
    result["texture"] = texture
    result["anomaly_map"] = anomaly_map

    # Heatmap on the unwrapped texture
    result["heatmap_on_texture"] = overlay_heatmap(texture, anomaly_map)

    # If full frame: inverse warp heatmap back to cone view
    if full_frame and anomaly_map is not None:
        center, inner_r, outer_r = result["geometry"]
        h, w = cone_crop.shape[:2]

        # Resize anomaly map to match texture dimensions
        hm_resized = cv2.resize(anomaly_map, (texture.shape[1], texture.shape[0]))

        cone_heatmap = inverse_unwarp(
            hm_resized, (w, h), center, inner_r, outer_r, angular_res
        )
        result["heatmap_on_cone"] = overlay_heatmap(cone_crop, cone_heatmap)

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate PatchCore model on test images")
    parser.add_argument(
        "--model", required=True,
        help="Path to exported model directory (contains model.pt or model.xml)",
    )
    parser.add_argument("--image", help="Single image to evaluate")
    parser.add_argument("--input-dir", help="Directory of images to batch evaluate")
    parser.add_argument(
        "--full-frame", action="store_true",
        help="Images are full frames (run YOLO + unwarp first)",
    )
    parser.add_argument(
        "--weights", default="weights/visible_yolo.pt",
        help="YOLO weights (only needed with --full-frame)",
    )
    parser.add_argument("--conf", type=float, default=0.6, help="YOLO confidence")
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Anomaly score threshold for stain detection",
    )
    parser.add_argument(
        "--angular-res", type=int, default=1024,
        help="Angular resolution for unwarp",
    )
    parser.add_argument("--save", action="store_true", help="Save instead of display")
    args = parser.parse_args()

    if not args.image and not args.input_dir:
        print("ERROR: Specify --image or --input-dir")
        sys.exit(1)

    # Load model
    inferencer = load_inferencer(args.model)

    # Load YOLO if needed
    yolo_detector = None
    if args.full_frame:
        from inspection.yolo_detector import YOLODetector
        yolo_detector = YOLODetector(args.weights, conf_threshold=args.conf)

    # Collect images
    if args.image:
        image_paths = [args.image]
    else:
        input_dir = Path(args.input_dir)
        image_paths = sorted(
            str(p) for p in input_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
        )

    out_dir = Path("training/patchcore/output/evaluate")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEvaluating {len(image_paths)} image(s)...")
    print(f"Threshold: {args.threshold}")
    print()

    for img_path in image_paths:
        name = Path(img_path).stem
        print(f"  {Path(img_path).name}: ", end="", flush=True)

        result = evaluate_single(
            inferencer, img_path, args.full_frame,
            yolo_detector, args.angular_res, args.threshold,
        )

        if "error" in result:
            print(f"ERROR - {result['error']}")
            continue

        status = "STAIN" if result["has_stain"] else "OK"
        print(f"score={result['score']:.4f}  [{status}]")

        # Save / display
        cv2.imwrite(str(out_dir / f"{name}_texture_heatmap.jpg"), result["heatmap_on_texture"])

        if "heatmap_on_cone" in result:
            cv2.imwrite(str(out_dir / f"{name}_cone_heatmap.jpg"), result["heatmap_on_cone"])
        if "debug_geometry" in result:
            cv2.imwrite(str(out_dir / f"{name}_geometry.jpg"), result["debug_geometry"])

        if not args.save and len(image_paths) == 1:
            # Display for single image evaluation
            tex_hm = result["heatmap_on_texture"]
            scale = min(1200 / tex_hm.shape[1], 600 / tex_hm.shape[0], 1.0)
            if scale < 1.0:
                tex_hm = cv2.resize(tex_hm, None, fx=scale, fy=scale)
            cv2.imshow("Texture + Heatmap", tex_hm)

            if "heatmap_on_cone" in result:
                cone_hm = result["heatmap_on_cone"]
                scale = min(800 / cone_hm.shape[1], 600 / cone_hm.shape[0], 1.0)
                if scale < 1.0:
                    cone_hm = cv2.resize(cone_hm, None, fx=scale, fy=scale)
                cv2.imshow("Cone + Heatmap", cone_hm)

            print(f"\nSaved to: {out_dir}")
            print("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if args.save or len(image_paths) > 1:
        print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
