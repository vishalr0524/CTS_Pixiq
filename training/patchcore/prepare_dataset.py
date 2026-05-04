"""
Prepare PatchCore training dataset: full frame → YOLO detect → annular crop → save.

Processes all images in a source directory through the YOLO detector to extract
yarn_cone donut crops (annular mask: black outside cone edge + black inside tube
hole) and saves them to an anomalib-compatible dataset structure.

The training images match what PatchCore sees at inference: the rectangular cone
crop with an annular mask applied. Only the yarn surface (donut between tube hole
and cone edge) is visible. Black corners and tube hole are zeroed out.

Output structure:
    dataset/
        good/       ← annular cone crops of normal cones (from --input)
        stain/      ← annular cone crops with defects  (from --stain, optional)

Usage:
    # Step 1: Process normal images (all materials at once)
    uv run python training/patchcore/prepare_dataset.py \\
        --input /path/to/Master/visible \\
        --output training/patchcore/dataset/good

    # Step 2 (optional): Process defect images for validation
    uv run python training/patchcore/prepare_dataset.py \\
        --input /path/to/defect/images \\
        --output training/patchcore/dataset/stain
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project src to path for inspection imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from inspection.yolo_detector import YOLODetector


def find_geometry(cone_bbox, tube_bbox):
    """Derive cone geometry from YOLO bboxes (same as polar_unwarp.py)."""
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
    """Create a donut-shaped binary mask.

    Args:
        shape: Image shape (H, W) or (H, W, C).
        center: (cx, cy) center of the annulus.
        inner_r: Inner radius (tube hole edge).
        outer_r: Outer radius (cone edge).

    Returns:
        Binary mask (H, W) with 255 inside the donut, 0 outside.
    """
    h, w = shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = (dist >= inner_r) & (dist <= outer_r)
    return mask.astype(np.uint8) * 255


def draw_debug(image, center, inner_r, outer_r):
    """Draw geometry overlay on a copy of the image."""
    vis = image.copy()
    cv2.circle(vis, center, int(inner_r), (0, 0, 255), 2)   # Inner (tube) - red
    cv2.circle(vis, center, int(outer_r), (0, 255, 0), 2)   # Outer (cone) - green
    cv2.circle(vis, center, 5, (255, 0, 0), -1)              # Center - blue
    return vis


def collect_images(input_dir):
    """Collect image files from input_dir or its subdirectories.

    Searches for images directly in input_dir, and also in
    input_dir/*/raw/ and input_dir/*/good/ subdirectories
    (Master/visible/MATERIAL/raw structure).
    """
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    image_paths = []

    # Direct images in input_dir
    direct = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if direct:
        image_paths.extend(direct)

    # Subdirectory structure: MATERIAL/raw/ or MATERIAL/good/
    for subdir in sorted(input_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for folder_name in ("raw", "good"):
            folder = subdir / folder_name
            if folder.exists():
                imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
                image_paths.extend(imgs)

    return sorted(set(image_paths))


def main():
    parser = argparse.ArgumentParser(
        description="Prepare annular (donut) cone crops for PatchCore training"
    )
    parser.add_argument(
        "--input", required=True,
        help="Directory with full-frame images. Searches directly and in */raw/, */good/ subdirs.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for annular cone crops (e.g. training/patchcore/dataset/good)",
    )
    parser.add_argument(
        "--weights", default="weights/visible_yolo.pt",
        help="YOLO model weights",
    )
    parser.add_argument(
        "--conf", type=float, default=0.6,
        help="YOLO confidence threshold",
    )
    parser.add_argument(
        "--save-debug", action="store_true",
        help="Also save debug images showing detected geometry circles",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_debug:
        debug_dir = output_dir.parent / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

    # Collect image files (direct + */raw/ + */good/ subdirs)
    image_paths = collect_images(input_dir)
    if not image_paths:
        print(f"ERROR: No images found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images in {input_dir}")
    print(f"Output: {output_dir}")

    # Load YOLO detector
    print(f"Loading YOLO: {args.weights}")
    detector = YOLODetector(args.weights, conf_threshold=args.conf)

    # Process each image
    success = 0
    skipped = 0

    for i, img_path in enumerate(image_paths):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [{i+1}/{len(image_paths)}] SKIP (cannot read): {img_path.name}")
            continue

        # YOLO detect — need both cone and tube
        detections = detector.detect(image)
        cone_det = detector.get_detection_by_class(detections, "yarn_cone")
        tube_det = detector.get_detection_by_class(detections, "yarn_tube")

        if cone_det is None or tube_det is None:
            skipped += 1
            missing = "yarn_cone" if cone_det is None else "yarn_tube"
            print(f"  [{i+1}/{len(image_paths)}] SKIP (no {missing}): {img_path.name}")
            continue

        # Extract rectangular cone crop
        cone_crop = detector.extract_roi(image, cone_det)

        # Compute geometry for annular mask
        center, inner_r, outer_r = find_geometry(cone_det.bbox, tube_det.bbox)

        # Apply annular mask (donut): black outside cone edge + black inside tube hole
        mask = create_annular_mask(cone_crop.shape, center, inner_r, outer_r)
        donut_crop = cone_crop.copy()
        donut_crop[mask == 0] = 0

        # Save annular cone crop
        # Use material folder name as prefix to avoid filename collisions
        material_name = img_path.parent.name
        if material_name in ("raw", "good"):
            material_name = img_path.parent.parent.name
        out_name = f"{material_name}_{img_path.stem}.png"
        cv2.imwrite(str(output_dir / out_name), donut_crop)
        success += 1

        if args.save_debug:
            debug_img = draw_debug(cone_crop, center, inner_r, outer_r)
            cv2.imwrite(str(debug_dir / out_name), debug_img)

        if (i + 1) % 50 == 0 or (i + 1) == len(image_paths):
            print(f"  [{i+1}/{len(image_paths)}] {success} saved, {skipped} skipped")

    print(f"\nDone: {success}/{len(image_paths)} images processed")
    print(f"  Saved to: {output_dir}")
    print(f"  Cone crops: annular donut (yarn surface only, black bg + tube hole)")
    if skipped:
        print(f"  Skipped (missing detection): {skipped}")


if __name__ == "__main__":
    main()
