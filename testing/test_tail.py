"""
Test Tail Inspection — verifies YOLO-based yarn tail detection.

Tests skipped if no Tail images available in test_images/Tail/.
"""

import cv2
import pytest


class TestTailInspection:
    """Test tail YOLO detection pipeline."""

    def test_tail_inspector_initializes(self, tail_inspector) -> None:
        """Tail inspector should initialize."""
        assert tail_inspector is not None

    def test_no_tail_images_skips(self, tail_images: dict) -> None:
        """Informational: report Tail image availability."""
        if not tail_images:
            pytest.skip("No Tail test images in test_images/Tail/")

    @pytest.mark.parametrize("category", ["good", "bad"])
    def test_tail_detection(
        self,
        tail_images: dict,
        tail_inspector,
        results_dir,
        category: str,
    ) -> None:
        """Run tail inspection on categorized images."""
        if tail_inspector is None or not tail_inspector._model_loaded:
            pytest.skip("Tail model not loaded")

        if category not in tail_images:
            pytest.skip(f"No Tail {category} images")

        out_dir = results_dir / "tail" / category
        out_dir.mkdir(parents=True, exist_ok=True)

        for image_path in tail_images[category]:
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue

            result = tail_inspector.process_frame(frame)

            # Save detection result
            with open(out_dir / f"{image_path.stem}_result.txt", "w") as f:
                f.write(f"Image: {image_path.name}\n")
                f.write(f"Tail detected: {result.tail_detected}\n")
                f.write(f"Confidence: {result.confidence:.3f}\n")
                f.write(f"Bbox: {result.bbox}\n")
                f.write(f"Model loaded: {result.model_loaded}\n")

            # Save annotated frame if tail found
            if result.tail_detected and result.bbox:
                annotated = frame.copy()
                x1, y1, x2, y2 = result.bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"tail {result.confidence:.2f}"
                cv2.putText(annotated, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imwrite(str(out_dir / f"{image_path.stem}_detection.jpg"), annotated)

            # Validate expected outcomes
            if category == "good":
                assert result.tail_detected, (
                    f"Good tail image {image_path.name} not detected"
                )
            elif category == "bad":
                assert not result.tail_detected, (
                    f"Bad tail image {image_path.name} falsely detected"
                )
