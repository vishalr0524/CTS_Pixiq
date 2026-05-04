"""
Remove dark frames caused by UV light sync issues.

When the UV light hasn't fully fired, the image is mostly dark/black.
This script scans both good-image folders and copies only valid (bright enough)
images to a clean output directory.

Strategy:
    - Extract the blue channel (UV fluorescence channel)
    - Compute mean brightness of the blue channel
    - Reject images below a brightness threshold (default: 15.0)
    - Also reject images that are almost entirely black (> 90% of pixels near zero)

Usage:
    uv run python training/patchcore_uv/clean_dark_frames.py
    uv run python training/patchcore_uv/clean_dark_frames.py --threshold 15 --preview
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


# Source folders relative to project root
DEFAULT_INPUT_DIRS = [
    "uv_train_data_18_03_26/2026-03-17",
    "uv_train_data_18_03_26/2026-03-18",
]
DEFAULT_OUTPUT_DIR = "training/patchcore_uv/dataset/raw_good"

# Blue channel mean brightness below this → dark frame (light sync issue)
DEFAULT_BRIGHTNESS_THRESHOLD = 15.0

# If more than this fraction of blue-channel pixels are near-zero → reject
DEFAULT_DARK_PIXEL_FRACTION = 0.90


def is_dark_frame(
    img_path: Path,
    brightness_threshold: float,
    dark_pixel_fraction: float,
) -> tuple[bool, float, float]:
    """Check if a UV image is a dark frame.

    Args:
        img_path: Path to the image.
        brightness_threshold: Minimum acceptable mean blue-channel brightness.
        dark_pixel_fraction: Max fraction of near-zero pixels before rejection.

    Returns:
        (is_dark, mean_brightness, dark_fraction)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return True, 0.0, 1.0

    # Blue channel carries UV fluorescence signal (index 0 in BGR)
    blue = img[:, :, 0].astype(np.float32)

    mean_brightness = float(blue.mean())

    # Fraction of pixels with value <= 5 (effectively black)
    dark_fraction = float((blue <= 5).sum()) / blue.size

    # Reject if mean brightness too low OR too many dark pixels
    is_dark = (mean_brightness < brightness_threshold) or (dark_fraction > dark_pixel_fraction)

    return is_dark, mean_brightness, dark_fraction


def main():
    parser = argparse.ArgumentParser(description="Remove dark UV frames from training data")
    parser.add_argument(
        "--input-dirs", nargs="+", default=DEFAULT_INPUT_DIRS,
        help="Source image directories (relative to project root)",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_DIR,
        help="Output directory for clean images",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_BRIGHTNESS_THRESHOLD,
        help="Minimum mean blue-channel brightness (default: 15.0)",
    )
    parser.add_argument(
        "--dark-fraction", type=float, default=DEFAULT_DARK_PIXEL_FRACTION,
        help="Max fraction of near-zero pixels before rejection (default: 0.90)",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Dry run — print stats without copying files",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root (two levels up from this script)
    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = project_root / args.output

    if not args.preview:
        output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    rejected = 0
    rejected_list = []

    for input_rel in args.input_dirs:
        input_dir = project_root / input_rel
        if not input_dir.exists():
            print(f"WARNING: Input directory not found: {input_dir}")
            continue

        images = sorted(input_dir.glob("*.png")) + sorted(input_dir.glob("*.jpg"))
        print(f"\nScanning {input_dir.name}: {len(images)} images")

        folder_kept = 0
        folder_rejected = 0

        for img_path in images:
            total += 1
            dark, mean_b, dark_frac = is_dark_frame(
                img_path, args.threshold, args.dark_fraction
            )

            if dark:
                rejected += 1
                folder_rejected += 1
                rejected_list.append((img_path.name, mean_b, dark_frac))
                print(f"  REJECT  {img_path.name:<20}  blue_mean={mean_b:5.1f}  dark_frac={dark_frac:.2f}")
            else:
                kept += 1
                folder_kept += 1
                if not args.preview:
                    # Prefix with folder name to avoid filename collisions across days
                    out_name = f"{input_dir.name}_{img_path.name}"
                    shutil.copy2(img_path, output_dir / out_name)

        print(f"  -> kept {folder_kept}, rejected {folder_rejected}")

    print(f"\n{'='*50}")
    print(f"Total scanned : {total}")
    print(f"Kept (clean)  : {kept}")
    print(f"Rejected (dark): {rejected}  ({100*rejected/max(total,1):.1f}%)")
    if not args.preview:
        print(f"Output        : {output_dir}")
    else:
        print(f"[DRY RUN] No files copied")

    if rejected_list:
        print(f"\nRejected images:")
        for name, mean_b, dark_frac in sorted(rejected_list):
            print(f"  {name:<25}  blue_mean={mean_b:5.1f}  dark_frac={dark_frac:.2f}")


if __name__ == "__main__":
    main()
