"""
Test UV Inspection — verifies thread mixup detection via row variance.

Saves intermediate images:
    - YOLO detections on UV frame
    - Unwrapped grayscale strip
    - Row-mean intensity profile plot (saved as text data)

Tests skipped if no UV images available in test_images/UV/.
"""

import cv2
import numpy as np
import pytest


class TestUVInspection:
    """Test UV thread mixup detection pipeline."""

    def test_uv_inspector_initializes(self, uv_inspector) -> None:
        """UV inspector should initialize if config is valid."""
        if uv_inspector is None:
            pytest.skip("UV inspector not initialized (no weights path)")
        assert uv_inspector is not None

    def test_no_uv_images_skips(self, uv_images: dict) -> None:
        """Informational: report UV image availability."""
        if not uv_images:
            pytest.skip("No UV test images in test_images/UV/")

    @pytest.mark.parametrize("category", ["good", "defect"])
    def test_uv_detection(
        self,
        uv_images: dict,
        uv_inspector,
        results_dir,
        category: str,
    ) -> None:
        """Run UV inspection on categorized images."""
        if uv_inspector is None:
            pytest.skip("UV inspector not initialized")

        if category not in uv_images:
            pytest.skip(f"No UV {category} images")

        out_dir = results_dir / "uv" / category
        out_dir.mkdir(parents=True, exist_ok=True)

        for image_path in uv_images[category]:
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue

            result = uv_inspector.process_frame(frame)

            # Save unwrapped strip
            if result.unwrapped_strip is not None:
                cv2.imwrite(
                    str(out_dir / f"{image_path.stem}_unwrapped.jpg"),
                    result.unwrapped_strip,
                )

                # Save row mean profile as text
                row_means = result.unwrapped_strip.mean(axis=1)
                with open(out_dir / f"{image_path.stem}_profile.txt", "w") as f:
                    f.write(f"Image: {image_path.name}\n")
                    f.write(f"Has mixup: {result.has_mixup}\n")
                    f.write(f"Detrend std: {result.detrend_std:.2f}\n")
                    f.write(f"Row CV: {result.row_cv:.4f}\n")
                    f.write(f"\nRow means ({len(row_means)} rows):\n")
                    for i, val in enumerate(row_means):
                        f.write(f"  {i:3d}: {val:.1f}\n")

            # Validate expected outcomes
            if category == "good":
                assert result.detrend_std < 15.0, (
                    f"Good UV image {image_path.name} has detrend_std={result.detrend_std:.2f} >= 15.0"
                )
            elif category == "defect":
                assert result.has_mixup, (
                    f"Defect UV image {image_path.name} not detected as mixup "
                    f"(detrend_std={result.detrend_std:.2f})"
                )
