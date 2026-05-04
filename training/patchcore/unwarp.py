"""
Polar unwrap of yarn cone surface to a flat texture strip.

The visible-light camera captures the cone from the top (front-facing).
The image shows a circular cone surface with a dark tube hole in the center.
This module uses YOLO-detected cone and tube bboxes to determine the geometry
(center, inner/outer radii) and applies a polar-to-rectangular transform to
produce a flat texture strip of the yarn surface — suitable for PatchCore
anomaly detection.

Algorithm (pure bbox arithmetic — no image processing for geometry):
    1. Tube bbox center (mapped into cone-crop coords) → cone center
    2. Half the max side of tube bbox → inner radius
    3. Half the cone crop width → outer radius
    4. cv2.warpPolar (linear) → rectangular strip
    5. Radial crop trims inner/outer edges to remove tube/background leakage

The cone bbox width is tight to the cone's horizontal extent (verified on
356 images). Any small background overshoot at the outer edge is handled
by the radial crop (default 5% from each edge). This avoids running Otsu
thresholding, contour analysis, or angular binning — important on Jetson
where YOLO already consumes significant compute.

Geometry:
    +------------------+         +--------------------------+
    |    dark bg       |         |                          |
    |  +----------+    |         | unwrapped yarn surface   |
    | |  yarn     |    |  --->   | (angle x radius)         |
    | |  (o) tube |    |         |                          |
    | |  surface  |    |         +--------------------------+
    |  +----------+    |
    +------------------+
    circular cone view           flat texture strip

Usage:
    center, inner_r, outer_r = find_geometry(cone_bbox, tube_bbox)
    texture = unwarp_cone(cone_crop, center, inner_r, outer_r)
"""

import cv2
import numpy as np


def find_geometry(
    cone_bbox: tuple,
    tube_bbox: tuple,
) -> tuple:
    """Derive cone geometry purely from YOLO bboxes.

    No image processing — just bbox arithmetic. The cone bbox width gives
    the outer diameter, the tube bbox gives center and inner radius.

    Args:
        cone_bbox: (x1, y1, x2, y2) of yarn_cone in full-frame pixels.
        tube_bbox: (x1, y1, x2, y2) of yarn_tube in full-frame pixels.

    Returns:
        (center, inner_r, outer_r) where center is (cx, cy) in cone-crop
        coordinates (i.e. relative to cone_bbox top-left corner).
    """
    cx1, cy1, cx2, cy2 = cone_bbox
    tx1, ty1, tx2, ty2 = tube_bbox

    # Tube center in cone-crop coordinates
    tube_cx = (tx1 + tx2) / 2 - cx1
    tube_cy = (ty1 + ty2) / 2 - cy1
    center = (int(tube_cx), int(tube_cy))

    # Inner radius = max inscribed circle in tube bbox
    tube_w = tx2 - tx1
    tube_h = ty2 - ty1
    inner_r = float(min(tube_w, tube_h)) / 2

    # Outer radius = max inscribed circle in cone bbox
    cone_w = cx2 - cx1
    cone_h = cy2 - cy1
    outer_r = float(min(cone_w, cone_h)) / 2

    return center, inner_r, outer_r


# ── core transform ───────────────────────────────────────────────────────


def unwarp_cone(
    image: np.ndarray,
    center: tuple,
    inner_r: float,
    outer_r: float,
    angular_res: int = 1024,
    radial_crop: float = 0.05,
) -> np.ndarray:
    """Polar-to-rectangular transform of the cone annulus.

    Maps the annular yarn surface (between inner_r and outer_r) to a
    rectangular texture strip using cv2.warpPolar, then trims a percentage
    from both the inner and outer edges to guarantee clean yarn-only content.

    Output layout:
        - Y-axis (rows): angle around the cone, 0 to 360 degrees
        - X-axis (cols): radial position, inner edge to outer edge

    Args:
        image: BGR cone image.
        center: (cx, cy) center of the cone.
        inner_r: Inner radius (tube hole boundary).
        outer_r: Outer radius (cone edge).
        angular_res: Number of angular steps (output height). Default 1024.
        radial_crop: Fraction to trim from each edge (default 0.05 = 5%).

    Returns:
        Flat texture strip (angular_res x radial_extent x 3), BGR.
    """
    radial_res = int(outer_r)

    unwrapped = cv2.warpPolar(
        image,
        dsize=(radial_res, angular_res),
        center=center,
        maxRadius=outer_r,
        flags=cv2.WARP_POLAR_LINEAR,
    )

    # Column range for the annulus
    col_inner = int(inner_r / outer_r * radial_res)
    col_outer = radial_res

    # Apply radial crop (trim both edges)
    annulus_width = col_outer - col_inner
    crop_px = int(annulus_width * radial_crop)
    col_inner += crop_px
    col_outer -= crop_px

    return unwrapped[:, col_inner:col_outer]


def inverse_unwarp(
    texture: np.ndarray,
    output_size: tuple,
    center: tuple,
    inner_r: float,
    outer_r: float,
    angular_res: int = 1024,
    radial_crop: float = 0.05,
) -> np.ndarray:
    """Map a rectangular texture (or heatmap) back to circular cone space.

    Inverse of unwarp_cone().  Used to project PatchCore anomaly heatmaps
    back onto the original cone view for visualization.

    Args:
        texture: Rectangular image/heatmap (angular_res x radial_extent).
        output_size: (width, height) of the output circular image.
        center: (cx, cy) center of the cone in the output.
        inner_r: Inner radius.
        outer_r: Outer radius.
        angular_res: Angular resolution used in the forward unwarp.
        radial_crop: Same crop value used in the forward unwarp.

    Returns:
        Image/heatmap in circular cone space.
    """
    radial_res = int(outer_r)
    col_inner = int(inner_r / outer_r * radial_res)
    annulus_width = radial_res - col_inner
    crop_px = int(annulus_width * radial_crop)
    col_start = col_inner + crop_px

    if texture.ndim == 2:
        full_polar = np.zeros((angular_res, radial_res), dtype=texture.dtype)
        full_polar[:, col_start : col_start + texture.shape[1]] = texture
    else:
        full_polar = np.zeros(
            (angular_res, radial_res, texture.shape[2]), dtype=texture.dtype
        )
        full_polar[:, col_start : col_start + texture.shape[1]] = texture

    result = cv2.warpPolar(
        full_polar,
        dsize=output_size,
        center=center,
        maxRadius=outer_r,
        flags=cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP,
    )
    return result


def draw_debug(
    image: np.ndarray, center: tuple, inner_r: float, outer_r: float
) -> np.ndarray:
    """Draw detected geometry circles on the image for debugging."""
    debug = image.copy()
    cv2.circle(debug, center, int(outer_r), (0, 255, 0), 2)
    cv2.circle(debug, center, int(inner_r), (0, 0, 255), 2)
    cv2.circle(debug, center, 3, (255, 0, 0), -1)
    cv2.putText(
        debug,
        f"R_out={outer_r:.0f} R_in={inner_r:.0f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
    )
    return debug


# ── standalone test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Test cone polar unwrap")
    parser.add_argument("image", help="Path to a full-frame image")
    parser.add_argument("--angular-res", type=int, default=1024)
    parser.add_argument("--radial-crop", type=float, default=0.05)
    parser.add_argument("--weights", default="weights/visible_yolo.pt")
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"ERROR: Cannot read {args.image}")
        sys.exit(1)
    print(f"Image: {args.image} ({img.shape[1]}x{img.shape[0]})")

    sys.path.insert(
        0, str(Path(__file__).resolve().parent.parent.parent / "src")
    )
    from inspection.yolo_detector import YOLODetector

    detector = YOLODetector(args.weights, conf_threshold=args.conf)
    dets = detector.detect(img)

    cone_det = detector.get_detection_by_class(dets, "yarn_cone")
    tube_det = detector.get_detection_by_class(dets, "yarn_tube")

    if cone_det is None:
        print("ERROR: No yarn_cone detected")
        sys.exit(1)
    if tube_det is None:
        print("ERROR: No yarn_tube detected")
        sys.exit(1)

    cone_crop = detector.extract_roi(img, cone_det)
    center, inner_r, outer_r = find_geometry(cone_det.bbox, tube_det.bbox)
    print(f"Center: {center}, Inner R: {inner_r:.1f}, Outer R: {outer_r:.1f}")

    texture = unwarp_cone(
        cone_crop, center, inner_r, outer_r, args.angular_res, args.radial_crop
    )
    print(f"Texture: {texture.shape[1]}x{texture.shape[0]}")

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    debug = draw_debug(cone_crop, center, inner_r, outer_r)
    cv2.imwrite(str(out_dir / "unwarp_debug.jpg"), debug)
    cv2.imwrite(str(out_dir / "unwarp_texture.jpg"), texture)
    print(f"Saved: {out_dir / 'unwarp_debug.jpg'}")
    print(f"Saved: {out_dir / 'unwarp_texture.jpg'}")

    if not args.save:
        scale = min(1200 / debug.shape[1], 800 / debug.shape[0], 1.0)
        debug_small = cv2.resize(debug, None, fx=scale, fy=scale) if scale < 1 else debug

        tex_scale = min(1200 / texture.shape[1], 400 / texture.shape[0], 1.0)
        tex_small = cv2.resize(texture, None, fx=tex_scale, fy=tex_scale) if tex_scale < 1 else texture

        cv2.imshow("Geometry Detection", debug_small)
        cv2.imshow("Unwrapped Texture", tex_small)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
