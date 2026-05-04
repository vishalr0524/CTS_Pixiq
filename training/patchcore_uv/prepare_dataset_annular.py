"""
Prepare UV PatchCore dataset: raw UV frame → YOLO detect → annular crop → save.

Pipeline per image:
    1. YOLO detect → yarn_cone bbox + yarn_tube bbox
    2. Extract rectangular cone crop
    3. Apply annular (donut) mask: black outside cone edge + black inside tube hole
    4. Save masked crop → dataset/good/ or dataset/defect/

The annular crop is what PatchCore sees at both training and inference time.
At inference, anomaly score = max(heatmap[annular_mask > 0]) so background and
tube hole pixels never contribute to the score.

Output structure (anomalib MVTec-like, ready for train.py):
    dataset/
        good/       ← annular UV crops of normal cones (both date folders)
        defect/     ← annular UV crops of bad cones (UV Bad images)

Usage:
    # Step 1: process good images (both date folders)
    uv run python training/patchcore_uv/prepare_dataset_annular.py --good

    # Step 2: process defect images for validation
    uv run python training/patchcore_uv/prepare_dataset_annular.py --defect

    # Both in one shot
    uv run python training/patchcore_uv/prepare_dataset_annular.py --good --defect

    # With debug overlays
    uv run python training/patchcore_uv/prepare_dataset_annular.py --good --defect --save-debug
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project src to path for YOLO detector import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from inspection.yolo_detector import YOLODetector


# --- Paths (relative to project root) ---
CLEAN_GOOD_DIR = "training/patchcore_uv/dataset/raw_good"   # output of clean_dark_frames.py
UV_BAD_DIR     = "../uv_train_data_18_03_26/UV Bad images"
OUTPUT_GOOD    = "training/patchcore_uv/dataset/good"
OUTPUT_DEFECT  = "training/patchcore_uv/dataset/defect"
YOLO_WEIGHTS   = "weights/uv_yolo.pt"
YOLO_CONF      = 0.3   # UV YOLO — lower threshold than VL (UV images are dimmer)


def find_geometry(cone_bbox, tube_bbox):
    """Derive annular geometry from YOLO bboxes.

    Returns center in cone-crop coordinates, inner radius (tube edge),
    and outer radius (cone edge). Uses min(w, h) for inscribed circle.
    """
    cx1, cy1, cx2, cy2 = cone_bbox
    tx1, ty1, tx2, ty2 = tube_bbox

    # Tube center mapped into cone-crop coordinate space
    tube_cx = (tx1 + tx2) / 2 - cx1
    tube_cy = (ty1 + ty2) / 2 - cy1
    center = (int(tube_cx), int(tube_cy))

    # Max inscribed circle radii
    inner_r = float(min(tx2 - tx1, ty2 - ty1)) / 2
    outer_r = float(min(cx2 - cx1, cy2 - cy1)) / 2

    return center, inner_r, outer_r


def apply_outer_margin(outer_r: float, margin: float) -> float:
    """Shrink outer radius by margin fraction to exclude background leakage."""
    return outer_r * (1.0 - margin)


def create_annular_mask(shape, center, inner_r, outer_r):
    """Binary donut mask: 255 on yarn surface, 0 on tube hole and background."""
    h, w = shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = (dist >= inner_r) & (dist <= outer_r)
    return mask.astype(np.uint8) * 255


def draw_debug(image, center, inner_r, outer_r):
    """Draw geometry circles on image copy for visual verification."""
    vis = image.copy()
    cv2.circle(vis, center, int(inner_r), (0, 0, 255), 2)   # tube edge — red
    cv2.circle(vis, center, int(outer_r), (0, 255, 0), 2)   # cone edge — green
    cv2.circle(vis, center, 5, (255, 0, 0), -1)             # center — blue
    return vis


def process_images(
    image_paths: list[Path],
    output_dir: Path,
    detector: YOLODetector,
    debug_dir: Path | None = None,
    outer_margin: float = 0.0,
) -> tuple[int, int]:
    """Process a list of images → annular crops → output_dir.

    Returns (success_count, skipped_count).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    success = 0
    skipped = 0

    for i, img_path in enumerate(image_paths):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [{i+1}/{len(image_paths)}] SKIP (unreadable): {img_path.name}")
            skipped += 1
            continue

        # YOLO detect — need both cone and tube
        detections = detector.detect(image)
        cone_det = detector.get_detection_by_class(detections, "yarn_cone")
        tube_det = detector.get_detection_by_class(detections, "yarn_tube")

        if cone_det is None or tube_det is None:
            missing = "yarn_cone" if cone_det is None else "yarn_tube"
            print(f"  [{i+1}/{len(image_paths)}] SKIP (no {missing}): {img_path.name}")
            skipped += 1
            continue

        # Rectangular cone crop
        cone_crop = detector.extract_roi(image, cone_det)

        # Annular geometry
        center, inner_r, outer_r = find_geometry(cone_det.bbox, tube_det.bbox)
        outer_r = apply_outer_margin(outer_r, outer_margin)

        if outer_r <= inner_r or outer_r <= 0:
            print(f"  [{i+1}/{len(image_paths)}] SKIP (bad geometry inner={inner_r:.0f} outer={outer_r:.0f}): {img_path.name}")
            skipped += 1
            continue

        # Extract green channel only (yellow-green filter — fluorescence signal is in green)
        # Convert to 3-channel grayscale so PatchCore (RGB model) accepts it
        green_channel = cone_crop[:, :, 1]  # BGR index 1 = green
        green_3ch = cv2.merge([green_channel, green_channel, green_channel])

        # Apply donut mask — black outside cone + black inside tube
        mask = create_annular_mask(green_3ch.shape, center, inner_r, outer_r)
        donut_crop = green_3ch.copy()
        donut_crop[mask == 0] = 0

        # Save — stem only (filenames already prefixed by date in clean_dark_frames.py)
        out_path = output_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(out_path), donut_crop)
        success += 1

        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / f"{img_path.stem}.png"), draw_debug(cone_crop, center, inner_r, outer_r))

        if (i + 1) % 100 == 0 or (i + 1) == len(image_paths):
            print(f"  [{i+1}/{len(image_paths)}] {success} saved, {skipped} skipped")

    return success, skipped


def collect_images(directory: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts)


def main():
    parser = argparse.ArgumentParser(description="Prepare UV annular crops for PatchCore")
    parser.add_argument("--good", action="store_true", help="Process good images")
    parser.add_argument("--defect", action="store_true", help="Process UV bad images")
    parser.add_argument("--save-debug", action="store_true", help="Save debug geometry overlays")
    parser.add_argument("--weights", default=YOLO_WEIGHTS, help="UV YOLO weights path")
    parser.add_argument("--conf", type=float, default=YOLO_CONF, help="YOLO confidence threshold")
    parser.add_argument("--outer-margin", type=float, default=0.05,
                        help="Fraction to shrink outer radius to exclude background (default: 0.05)")
    args = parser.parse_args()

    if not args.good and not args.defect:
        parser.print_help()
        print("\nERROR: Specify --good and/or --defect")
        sys.exit(1)

    project_root = Path(__file__).resolve().parent.parent.parent

    weights_path = project_root / args.weights
    if not weights_path.exists():
        print(f"ERROR: YOLO weights not found: {weights_path}")
        sys.exit(1)

    print(f"Loading UV YOLO: {weights_path}")
    detector = YOLODetector(str(weights_path), conf_threshold=args.conf)

    if args.good:
        good_dir = project_root / CLEAN_GOOD_DIR
        if not good_dir.exists():
            print(f"ERROR: Clean good dir not found: {good_dir}")
            print("Run clean_dark_frames.py first.")
            sys.exit(1)

        images = collect_images(good_dir)
        print(f"\nProcessing {len(images)} clean good images...")

        debug_dir = project_root / "training/patchcore_uv/dataset/debug_good" if args.save_debug else None
        output_good = project_root / OUTPUT_GOOD
        success, skipped = process_images(images, output_good, detector, debug_dir, outer_margin=args.outer_margin)
        print(f"\nGood images: {success} saved, {skipped} skipped → {output_good}")

    if args.defect:
        bad_dir = project_root / UV_BAD_DIR
        if not bad_dir.exists():
            print(f"ERROR: UV Bad images dir not found: {bad_dir}")
            sys.exit(1)

        images = collect_images(bad_dir)
        # Remove hidden files (e.g. ' .png' — seen in the bad folder)
        images = [p for p in images if p.stem.strip()]
        print(f"\nProcessing {len(images)} defect images...")

        debug_dir = project_root / "training/patchcore_uv/dataset/debug_defect" if args.save_debug else None
        output_defect = project_root / OUTPUT_DEFECT
        success, skipped = process_images(images, output_defect, detector, debug_dir, outer_margin=args.outer_margin)
        print(f"\nDefect images: {success} saved, {skipped} skipped → {output_defect}")

    print("\nDone. Next step:")
    print("  uv run python training/patchcore_uv/split_and_train.py")


if __name__ == "__main__":
    main()
