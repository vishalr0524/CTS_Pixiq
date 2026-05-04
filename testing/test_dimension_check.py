"""
Test 2: Dimension Check — measure cone outer diameter and verify against DB.

Place a visible-light camera image in testing/images/ and run:
    python testing/test_dimension_check.py testing/images/sample.jpg

    Optional: specify material ID to verify against:
    python testing/test_dimension_check.py testing/images/sample.jpg --material MAT-001

What this tests:
    - YOLO detection → yarn_cone bbox
    - Contour-based outer diameter measurement
    - Pixel-to-mm conversion using calibration constant K
    - Verification against database reference specs (bottom_diameter_mm)

Output:
    - Console: measured diameter, reference specs, match results
    - Window: image with dimension annotations
    - Saves: testing/output/dimension_result.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from inspection.yolo_detector import YOLODetector
from inspection.dimension_check import DimensionChecker
from inspection.database import MaterialDatabase


def main():
    parser = argparse.ArgumentParser(description="Test dimension verification")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--weights", default="weights/visible_yolo.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.6, help="YOLO confidence threshold")
    parser.add_argument("--ppm", type=float, default=5.0, help="Pixels per mm (calibration K)")
    parser.add_argument("--material", default="MAT-001", help="Material ID for DB lookup")
    parser.add_argument("--save", action="store_true", help="Save output instead of displaying")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        print(f"ERROR: Cannot read image: {args.image}")
        sys.exit(1)

    print(f"Image: {args.image} ({image.shape[1]}x{image.shape[0]})")

    # Step 1: YOLO detection
    print("\n--- YOLO Detection ---")
    detector = YOLODetector(args.weights, conf_threshold=args.conf)
    detections = detector.detect(image)

    cone_det = detector.get_detection_by_class(detections, "yarn_cone")
    if cone_det is None:
        print("ERROR: yarn_cone not detected. Cannot measure dimensions.")
        sys.exit(1)

    print(f"yarn_cone: bbox={cone_det.bbox}, conf={cone_det.confidence:.3f}")
    cone_crop = detector.extract_roi(image, cone_det)

    # Step 2: Dimension measurement
    print(f"\n--- Dimension Measurement (K={args.ppm} px/mm) ---")
    checker = DimensionChecker(pixels_per_mm=args.ppm)

    # Bbox-based (quick)
    dims_bbox = checker.measure(cone_det.bbox)
    print(f"Bbox method:    D={dims_bbox.diameter_mm}mm")

    # Contour-based (accurate)
    dims_contour = checker.measure_from_contour(cone_crop, cone_det.bbox)
    print(f"Contour method: D={dims_contour.diameter_mm}mm")

    # Step 3: Database verification
    print(f"\n--- Database Verification (material={args.material}) ---")
    db = MaterialDatabase(":memory:")
    db.init_mock_db()

    specs = db.get_material_specs(args.material)
    if specs is None:
        print(f"WARNING: Material '{args.material}' not in mock DB. Skipping verification.")
        db.close()
    else:
        print(f"Reference:      D={specs.bottom_diameter_mm}mm (tol={specs.tolerance_mm}mm)")

        result = checker.verify(dims_contour, specs)
        print(f"\nVerification:")
        print(f"  Diameter:    {'PASS' if result.diameter_match else 'FAIL'} ({dims_contour.diameter_mm} vs {specs.bottom_diameter_mm})")
        print(f"  Overall:     {'ALL MATCH' if result.all_match else 'MISMATCH'}")
        db.close()

    # Draw result
    output = image.copy()
    x1, y1, x2, y2 = cone_det.bbox
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 200, 0), 2)

    # Diameter arrows at mid-height
    mid_y = (y1 + y2) // 2
    cv2.arrowedLine(output, (x1, mid_y), (x2, mid_y), (0, 220, 220), 2, tipLength=0.03)
    cv2.arrowedLine(output, (x2, mid_y), (x1, mid_y), (0, 220, 220), 2, tipLength=0.03)
    cv2.putText(output, f"D: {dims_contour.diameter_mm}mm", (x1, mid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 2)

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "dimension_result.jpg"
    cv2.imwrite(str(out_path), output)
    print(f"\nSaved: {out_path}")

    if not args.save:
        cv2.imshow("Dimension Check", output)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
