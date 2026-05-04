"""
Test Dimension Check — verifies pixel→mm conversion and tolerance comparison.

Tests:
    1. Measure cone/tube diameters from YOLO bboxes
    2. Verify against recipe specs
    3. Check tolerance logic (within/outside range)
"""

import cv2
import pytest

from src.inspection.dimension_check import DimensionChecker
from src.inspection.data_types import MaterialSpecs


class TestDimensionCheck:
    """Test dimension measurement and verification logic."""

    def test_measure_cone_from_bbox(self, dimension_checker: DimensionChecker) -> None:
        """Cone diameter calculation: min(w, h) / pixels_per_mm."""
        # Simulate a 1286x1261 cone bbox (typical 4K image)
        bbox = (100, 100, 1386, 1361)
        dia = dimension_checker.measure_cone(bbox)

        # min(1286, 1261) / 4.3 = 293.3mm
        expected = round(min(1286, 1261) / 4.3, 1)
        assert dia == expected, f"Expected {expected}, got {dia}"

    def test_measure_tube_from_bbox(self, dimension_checker: DimensionChecker) -> None:
        """Tube diameter calculation from tube bbox."""
        # Simulate a 249x255 tube bbox
        bbox = (600, 600, 849, 855)
        dia = dimension_checker.measure_tube(bbox)

        expected = round(min(249, 255) / 4.3, 1)
        assert dia == expected, f"Expected {expected}, got {dia}"

    def test_verify_within_tolerance(self, dimension_checker: DimensionChecker) -> None:
        """Dimensions within tolerance should pass."""
        from src.inspection.data_types import Dimensions

        measured = Dimensions(cone_diameter_mm=295.0, tube_diameter_mm=54.0)
        specs = MaterialSpecs(
            material_id="2",
            height_mm=0.0,
            top_diameter_mm=0.0,
            bottom_diameter_mm=300.0,
            tube_diameter_mm=55.0,
            cone_tolerance_mm=30.0,
            tube_tolerance_mm=18.0,
        )
        result = dimension_checker.verify(measured, specs)
        assert result.cone_diameter_match is True
        assert result.tube_diameter_match is True
        assert result.all_match is True

    def test_verify_outside_tolerance(self, dimension_checker: DimensionChecker) -> None:
        """Dimensions outside tolerance should fail."""
        from src.inspection.data_types import Dimensions

        measured = Dimensions(cone_diameter_mm=250.0, tube_diameter_mm=54.0)
        specs = MaterialSpecs(
            material_id="2",
            height_mm=0.0,
            top_diameter_mm=0.0,
            bottom_diameter_mm=300.0,
            tube_diameter_mm=55.0,
            cone_tolerance_mm=30.0,
            tube_tolerance_mm=18.0,
        )
        result = dimension_checker.verify(measured, specs)
        # 250 vs 300 ± 30 → diff=50 > 30 → FAIL
        assert result.cone_diameter_match is False
        assert result.all_match is False

    def test_tolerance_fallback(self, dimension_checker: DimensionChecker) -> None:
        """When per-dimension tolerance is 0, fall back to tolerance_mm."""
        from src.inspection.data_types import Dimensions

        measured = Dimensions(cone_diameter_mm=299.0, tube_diameter_mm=0.0)
        specs = MaterialSpecs(
            material_id="test",
            height_mm=0.0,
            top_diameter_mm=0.0,
            bottom_diameter_mm=300.0,
            tolerance_mm=5.0,
            cone_tolerance_mm=0.0,  # should fallback to 5.0
        )
        result = dimension_checker.verify(measured, specs)
        assert result.cone_diameter_match is True  # 299 vs 300±5 → OK

    @pytest.mark.parametrize("image_idx", range(6))
    def test_dimension_on_real_images(
        self,
        vl_images: dict,
        yolo_detector,
        dimension_checker: DimensionChecker,
        recipe_store,
        results_dir,
        image_idx: int,
    ) -> None:
        """Run dimension check on real VL images, save results."""
        all_images = []
        for pattern, paths in vl_images.items():
            for p in paths:
                all_images.append((pattern, p))

        if image_idx >= len(all_images):
            pytest.skip(f"Only {len(all_images)} VL images available")

        pattern, image_path = all_images[image_idx]
        frame = cv2.imread(str(image_path))
        detections = yolo_detector.detect(frame)

        cone_det = yolo_detector.get_detection_by_class(detections, "yarn_cone")
        tube_det = yolo_detector.get_detection_by_class(detections, "yarn_tube")

        cone_bbox = cone_det.bbox if cone_det else None
        tube_bbox = tube_det.bbox if tube_det else None

        measured = dimension_checker.measure_both(cone_bbox, tube_bbox)

        # Save measurement data
        out_dir = results_dir / "dimension" / pattern
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / f"{image_path.stem}_dimensions.txt", "w") as f:
            f.write(f"Image: {image_path.name}\n")
            f.write(f"Cone diameter: {measured.cone_diameter_mm:.1f} mm\n")
            f.write(f"Tube diameter: {measured.tube_diameter_mm:.1f} mm\n")
            if cone_bbox:
                w = cone_bbox[2] - cone_bbox[0]
                h = cone_bbox[3] - cone_bbox[1]
                f.write(f"Cone bbox: {cone_bbox} ({w}x{h}px)\n")
            if tube_bbox:
                w = tube_bbox[2] - tube_bbox[0]
                h = tube_bbox[3] - tube_bbox[1]
                f.write(f"Tube bbox: {tube_bbox} ({w}x{h}px)\n")

        # Measurements should be non-zero if detected
        if cone_det:
            assert measured.cone_diameter_mm > 0
        if tube_det:
            assert measured.tube_diameter_mm > 0
