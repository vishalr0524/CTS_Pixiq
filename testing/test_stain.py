"""
Test Stain Detection — verifies PatchCore anomaly detection on cone crops.

Saves intermediate images:
    - Cone crop (rectangular YOLO bbox)
    - Annular mask visualization
    - PatchCore anomaly heatmap overlay
    - Masked heatmap (only yarn surface scored)

Handles both PatchCore mode (model available) and fallback mode (HSV threshold).
"""

import cv2
import numpy as np
import pytest

from src.inspection.stain_detector import StainDetector
from src.inspection.polar_unwarp import find_geometry


class TestStainDetection:
    """Test PatchCore stain detection on visible-light cone crops."""

    def test_detector_initializes(self, stain_detector: StainDetector) -> None:
        """StainDetector must initialize (even if model missing, fallback mode)."""
        assert stain_detector is not None

    def test_empty_image_returns_clean(self, stain_detector: StainDetector) -> None:
        """Empty image should return no stain, score 0."""
        result = stain_detector.detect(np.zeros((0, 0, 3), dtype=np.uint8))
        assert result.has_stain is False
        assert result.anomaly_score == 0.0

    def test_no_geometry_returns_clean(self, stain_detector: StainDetector) -> None:
        """Missing geometry should return no stain."""
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        result = stain_detector.detect(image)
        assert result.has_stain is False

    @pytest.mark.parametrize("image_idx", range(6))
    def test_stain_on_real_images(
        self,
        vl_images: dict,
        yolo_detector,
        stain_detector: StainDetector,
        results_dir,
        image_idx: int,
    ) -> None:
        """Run stain detection on real VL images, save all intermediates."""
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

        if cone_det is None or tube_det is None:
            pytest.skip(f"Both cone and tube needed, image {image_path.name}")

        # Extract cone crop and geometry
        cone_crop = yolo_detector.extract_roi(frame, cone_det)
        center, inner_r, outer_r = find_geometry(cone_det.bbox, tube_det.bbox)

        # Run stain detection
        result = stain_detector.detect(
            cone_crop, center=center, inner_r=inner_r, outer_r=outer_r
        )

        # --- Save intermediates ---
        out_dir = results_dir / "stain" / pattern
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem

        # Cone crop
        cv2.imwrite(str(out_dir / f"{stem}_cone_crop.jpg"), cone_crop)

        # Annular mask visualization
        h, w = cone_crop.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, center, int(outer_r), 255, -1)
        cv2.circle(mask, center, int(inner_r), 0, -1)
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(out_dir / f"{stem}_annular_mask.jpg"), mask_vis)

        # Heatmap overlay
        if result.heatmap is not None:
            heatmap_norm = np.clip(result.heatmap, 0, 1)
            heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            overlay = cv2.addWeighted(cone_crop, 0.6, heatmap_color, 0.4, 0)
            cv2.imwrite(str(out_dir / f"{stem}_heatmap_overlay.jpg"), overlay)

            # Masked heatmap (only valid region)
            masked_heatmap = heatmap_color.copy()
            masked_heatmap[mask == 0] = 0
            cv2.imwrite(str(out_dir / f"{stem}_heatmap_masked.jpg"), masked_heatmap)

        # Results text
        with open(out_dir / f"{stem}_stain_result.txt", "w") as f:
            f.write(f"Image: {image_path.name}\n")
            f.write(f"Score: {result.anomaly_score:.4f}\n")
            f.write(f"Threshold: {stain_detector.threshold}\n")
            f.write(f"Has stain: {result.has_stain}\n")
            f.write(f"Mode: {'PatchCore' if stain_detector._inferencer else 'Fallback HSV'}\n")
            f.write(f"Geometry: center={center}, inner_r={inner_r:.0f}, outer_r={outer_r:.0f}\n")

        # Score must be a valid float
        assert isinstance(result.anomaly_score, float)
        assert result.anomaly_score >= 0.0
