"""
Prepare UV unwrapped images for PatchCore training.

Takes raw UV cone images from Master/raw/*/UV/, runs UV YOLO to detect
cone + tube, extracts blue channel, polar unwraps, crops 15% from both
inner and outer edges, and saves for PatchCore training.

Usage:
    uv run python training/patchcore_uv/prepare_dataset.py
    uv run python training/patchcore_uv/prepare_dataset.py --input ../Master/raw --output training/patchcore_uv/dataset
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from inspection.yolo_detector import YOLODetector

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def polar_unwrap_blue(
    cone_crop: np.ndarray,
    center: tuple,
    inner_r: int,
    outer_r: int,
    angular_res: int = 720,
    inner_crop: float = 0.15,
    outer_crop: float = 0.15,
    use_clahe: bool = True,
) -> np.ndarray:
    """Polar unwrap using BLUE channel only (for UV fluorescence).

    Args:
        cone_crop: BGR cone crop from YOLO.
        center: Tube center in cone crop coordinates (cx, cy).
        inner_r: Inner radius (tube boundary).
        outer_r: Outer radius (cone edge).
        angular_res: Angular resolution (pixels around circumference).
        inner_crop: Fraction to crop from inner edge.
        outer_crop: Fraction to crop from outer edge.
        use_clahe: Apply CLAHE contrast enhancement.

    Returns:
        Unwrapped grayscale image.
    """
    # Extract BLUE channel (index 0 in BGR)
    blue = cone_crop[:, :, 0].copy()

    # Mask out the tube region with black circle
    h, w = blue.shape
    mask = np.ones((h, w), dtype=np.uint8) * 255
    cv2.circle(mask, center, inner_r, 0, -1)
    blue[mask == 0] = 0

    # Apply CLAHE for contrast enhancement
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        blue = clahe.apply(blue)

    # Calculate crop radii (15% from both sides)
    radial_range = outer_r - inner_r
    inner_r_crop = inner_r + int(radial_range * inner_crop)
    outer_r_crop = outer_r - int(radial_range * outer_crop)

    if outer_r_crop <= inner_r_crop:
        return None

    # Polar unwrap from center to full outer radius
    unwrapped = cv2.warpPolar(
        blue,
        dsize=(outer_r, angular_res),
        center=center,
        maxRadius=outer_r,
        flags=cv2.WARP_POLAR_LINEAR,
    )
    # Rotate so: rows = radius (inner at top), cols = angle
    unwrapped = cv2.rotate(unwrapped, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Crop to donut region (exclude tube and outer background)
    unwrapped = unwrapped[inner_r_crop:outer_r_crop, :]

    # Strip remaining black rows (tube region after unwrap)
    row_means = unwrapped.mean(axis=1)
    valid_rows = row_means > 5

    if valid_rows.sum() < 10:
        return unwrapped if unwrapped.size > 0 else None

    valid_indices = np.where(valid_rows)[0]
    return unwrapped[valid_indices[0]:valid_indices[-1] + 1, :]


def process_image(
    img_path: Path,
    output_path: Path,
    detector: YOLODetector,
    target_size: tuple = (720, 256),
    save_debug: bool = False,
    debug_dir: Path = None,
) -> bool:
    """Process a single UV image: YOLO detect → unwrap → save.

    Args:
        img_path: Input image path.
        output_path: Output image path.
        detector: UV YOLO detector.
        target_size: Target size (width, height) for PatchCore.
        save_debug: Save intermediate debug images.
        debug_dir: Directory for debug images.

    Returns:
        True if successful.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        logger.warning(f"Failed to read: {img_path}")
        return False

    # Run YOLO to detect cone + tube
    detections = detector.detect(img)
    cone_det = detector.get_detection_by_class(detections, "yarn_cone")
    tube_det = detector.get_detection_by_class(detections, "yarn_tube")

    if cone_det is None or tube_det is None:
        logger.warning(f"YOLO failed (cone={cone_det is not None}, tube={tube_det is not None}): {img_path.name}")
        return False

    # Extract cone crop
    cx1, cy1, cx2, cy2 = map(int, cone_det.bbox)
    h, w = img.shape[:2]
    cx1, cy1 = max(0, cx1), max(0, cy1)
    cx2, cy2 = min(w, cx2), min(h, cy2)
    cone_crop = img[cy1:cy2, cx1:cx2].copy()

    if cone_crop.size == 0:
        logger.warning(f"Empty cone crop: {img_path.name}")
        return False

    # Geometry from YOLO bboxes
    tx1, ty1, tx2, ty2 = map(int, tube_det.bbox)
    center_x = (tx1 + tx2) // 2 - cx1
    center_y = (ty1 + ty2) // 2 - cy1
    center = (center_x, center_y)
    inner_r = min(tx2 - tx1, ty2 - ty1) // 2
    outer_r = min(cx2 - cx1, cy2 - cy1) // 2

    if outer_r <= inner_r or outer_r <= 0:
        logger.warning(f"Invalid radii (inner={inner_r}, outer={outer_r}): {img_path.name}")
        return False

    # Polar unwrap with blue channel — 15% crop from both sides
    unwrapped = polar_unwrap_blue(
        cone_crop, center, inner_r, outer_r,
        inner_crop=0.15, outer_crop=0.15,
    )

    if unwrapped is None or unwrapped.size == 0:
        logger.warning(f"Unwrap failed: {img_path.name}")
        return False

    # Resize to consistent size for PatchCore
    trim_rows = 5
    unwrapped_resized = cv2.resize(
        unwrapped,
        (target_size[0], target_size[1] + trim_rows),
        interpolation=cv2.INTER_LINEAR,
    )
    unwrapped_resized = unwrapped_resized[:-trim_rows, :]

    # Save
    cv2.imwrite(str(output_path), unwrapped_resized)

    # Save debug images if requested
    if save_debug and debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / f"cone_{img_path.name}"), cone_crop)
        cv2.imwrite(str(debug_dir / f"blue_{img_path.name}"), cone_crop[:, :, 0])

    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare UV dataset for PatchCore")
    parser.add_argument(
        "--input",
        type=str,
        default="../Master/raw",
        help="Master/raw directory with material subfolders containing UV/",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training/patchcore_uv/dataset",
        help="Output directory for prepared dataset",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/uv_yolo.pt",
        help="UV YOLO weights path",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="YOLO confidence threshold",
    )
    parser.add_argument(
        "--target-width", type=int, default=720,
    )
    parser.add_argument(
        "--target-height", type=int, default=256,
    )
    parser.add_argument(
        "--save-debug", action="store_true",
        help="Save debug crops (cone + blue channel)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    target_size = (args.target_width, args.target_height)

    # Collect all UV images from Master/raw/*/UV/
    uv_images = []
    for uv_folder in sorted(input_dir.glob("*/UV")):
        material_id = uv_folder.parent.name
        for img_path in sorted(uv_folder.glob("*.png")):
            uv_images.append((material_id, img_path))
    for uv_folder in sorted(input_dir.glob("*/UV")):
        for img_path in sorted(uv_folder.glob("*.jpg")):
            material_id = uv_folder.parent.name
            uv_images.append((material_id, img_path))

    if not uv_images:
        logger.error(f"No UV images found in {input_dir}/*/UV/")
        return

    logger.info(f"Found {len(uv_images)} UV images from {input_dir}")

    # Initialize UV YOLO detector
    logger.info(f"Loading UV YOLO: {args.weights}")
    detector = YOLODetector(
        model_path=args.weights,
        conf_threshold=args.conf,
    )

    # Process all images into good/ (all assumed normal for training)
    good_dir = output_dir / "good"
    good_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug" if args.save_debug else None

    success = 0
    fail = 0
    for material_id, img_path in uv_images:
        # Name: {material}_{original_name} to avoid collisions
        out_name = f"{material_id}_{img_path.name}"
        out_path = good_dir / out_name

        if process_image(img_path, out_path, detector, target_size,
                         save_debug=args.save_debug, debug_dir=debug_dir):
            success += 1
        else:
            fail += 1

    logger.info(f"\nDataset prepared at {output_dir}/good/")
    logger.info(f"  Success: {success}/{len(uv_images)}")
    logger.info(f"  Failed: {fail}/{len(uv_images)}")
    logger.info(f"  Image size: {target_size[0]}x{target_size[1]}")
    logger.info(f"  Crop: 15%% inner + 15%% outer")


if __name__ == "__main__":
    main()
