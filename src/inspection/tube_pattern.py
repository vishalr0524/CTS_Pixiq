"""
Tube Pattern Matcher — verifies tube label against expected template.

Two modes:
  - Verification (default): Distance to expected template vs per-class threshold.
  - Classification (legacy): NN finds nearest class among all templates.

Combined distance = (1 - fft_weight) * color_bhatt + fft_weight * fft_cosine
  where color_bhatt = 0.7 * LAB_bhatt + 0.3 * HSV_bhatt

  1. Color NN: Bhattacharyya distance on LAB a*b* histogram (dominant signal)
  2. FFT NN: Cosine distance on 1D FFT magnitude of intensity profile
     (shift-invariant spatial feature — discriminates same-color patterns)
  3. Combined: Weighted sum finds nearest class → pass/fail decision
  4. Distance gate: Reject if bhatt > threshold (catches untaught patterns)
  5. ResNet NN: Runs for monitoring/logging only — does NOT affect pass/fail

Color provides the dominant signal (99.85% production accuracy).
FFT adds shift-invariant spatial discrimination for same-color patterns
(e.g., VIOLET_TRIANGLE vs VIOLET_CHECKED: bhatt=0.190, fft_cosine=0.299).

Production performance (6,876 cycles):
  - Color NN alone: 99.85% accuracy (10 errors)
  - Color+FFT (w=0.7): 100% on master data, min margin 0.129 (+40% over color-only)
  - ResNet NN alone: 95.05% accuracy (340 errors, monitoring only)

Reference data is created by the teaching module from N tube images.
All class templates are loaded at startup for NN comparison.

Usage:
    matcher = TubePatternMatcher(template_dir="templates/tube")
    matcher.load_all_references()  # Load all class templates
    result = matcher.verify(tube_crop, material_id="MAT-001")
    if result.passed:
        print("Tube pattern OK")
"""

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

from .data_types import TubePatternResult

# Color matching pipeline
from .color_matching.find_radius import find_radius
from .color_matching.preprocess_pipeline import preprocess_cone_tip
from .color_matching.get_signature import get_statistical_signature
from .color_matching.bhattacharyya_distance import compute_bhattacharyya_distance
from .color_matching.entropy_2d import compute_2d_entropy
from .color_matching.hsv_histogram import compute_hs_histogram

logger = logging.getLogger(__name__)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two L2-normalized vectors. 0 = identical."""
    return 1.0 - float(np.dot(a, b))


class _ResNetFeatureExtractor:
    """ResNet50 feature extractor for tube pattern discrimination."""

    def __init__(self, device: str = "auto", inner_ratio: float = 0.30):
        """Initialize ResNet50 feature extractor.

        Args:
            device: PyTorch device ("auto", "cuda", "cpu").
            inner_ratio: Inner hole radius as fraction of outer radius.
                Used for annular masking to black out the tube's inner hole.
                Set to 0 to use circular mask (no inner hole).
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.inner_ratio = inner_ratio

        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.model.eval()
        self.model.to(self.device)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        logger.info(f"ResNet50 feature extractor initialized on {self.device} (inner_ratio={inner_ratio})")

    @torch.no_grad()
    def extract(self, bgr_image: np.ndarray, apply_mask: bool = True) -> np.ndarray:
        """Extract L2-normalized 2048-dim feature vector from BGR image.

        Args:
            bgr_image: BGR image (already masked or raw).
            apply_mask: If True, apply annular mask. Set to False if input
                is already masked (e.g., from extract_annular_roi).

        Returns:
            L2-normalized 2048-dim feature vector.
        """
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        if apply_mask:
            # Apply annular mask (outer circle minus inner hole)
            h, w = rgb.shape[:2]
            cx, cy = w // 2, h // 2
            outer_r = min(cx, cy) - 2
            inner_r = int(outer_r * self.inner_ratio)

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (cx, cy), outer_r, 255, -1)  # Fill outer
            if inner_r > 0:
                cv2.circle(mask, (cx, cy), inner_r, 0, -1)  # Cut out inner
            rgb = cv2.bitwise_and(rgb, rgb, mask=mask)

        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        feat = self.model(tensor).squeeze()
        feat = feat / (feat.norm() + 1e-8)
        return feat.cpu().numpy()


class TubePatternMatcher:
    """Tube label verification using Nearest Neighbor classification.

    Combined distance (Color + FFT) finds nearest class:
      - Color: Bhattacharyya distance on LAB a*b* histogram (dominant signal)
      - FFT: Cosine distance on 1D FFT magnitude (shift-invariant spatial)
      - combined = (1 - fft_weight) * bhatt + fft_weight * fft_cosine
      - Distance gate rejects if bhatt > threshold (untaught pattern safety net)
      - ResNet NN runs for monitoring/logging only

    All class templates must be loaded at startup via load_all_references().
    """

    def __init__(
        self,
        template_dir: str,
        bilateral_d: int = 9,
        bilateral_sigma_color: int = 75,
        bilateral_sigma_space: int = 75,
        inner_crop_pct: float = 0.10,
        outer_crop_pct: float = 0.10,
        inner_ratio: float = 0.80,
        max_entropy_delta: float = 0.5,
        max_bhatt_distance: float = 0.35,
        fft_weight: float = 0.3,
        verification_mode: bool = True,
        threshold_config: str = "",
        default_threshold: float = 0.25,
        device: str = "auto",
    ):
        """Initialize tube pattern matcher.

        Args:
            template_dir: Directory containing per-material .npz reference files.
            bilateral_d: Bilateral filter diameter for color preprocessing.
            bilateral_sigma_color: Bilateral filter sigma in color space.
            bilateral_sigma_space: Bilateral filter sigma in coordinate space.
            inner_crop_pct: Inner edge crop for polar unwarp sweet spot.
            outer_crop_pct: Outer edge crop for polar unwarp sweet spot.
            inner_ratio: Inner hole radius as fraction of outer radius (for masking).
                Used to black out the tube's center hole in ResNet features.
            max_entropy_delta: Maximum allowed entropy difference between live
                and matched template. Rejects matches where pattern structure
                differs (e.g. solid vs half-solid). 0 = disabled.
            max_bhatt_distance: Maximum Bhattacharyya distance to accept a match.
                If the nearest class distance exceeds this, the match is rejected
                even if the class name matches. Acts as safety net for untaught
                patterns or badly degraded images. 0 = disabled.
            fft_weight: Weight for FFT cosine distance in combined distance.
                combined = (1 - fft_weight) * bhatt + fft_weight * fft_cosine.
                0 = color-only (no FFT). Default 0.3 (70% color, 30% FFT).
            verification_mode: If True, compute distance only to expected template
                and compare against per-class threshold (no classification).
                If False, use NN classification (find nearest among all templates).
            threshold_config: Path to tube_verify_config.json with per-class
                thresholds. If empty, looks for it in template_dir parent.
            default_threshold: Default combined distance threshold when no
                per-class threshold is available. Used as fallback.
            device: PyTorch device for ResNet ("auto", "cuda", "cpu").
        """
        self.template_dir = Path(template_dir)
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.inner_crop_pct = inner_crop_pct
        self.outer_crop_pct = outer_crop_pct
        self.inner_ratio = inner_ratio
        self.max_entropy_delta = max_entropy_delta
        self.max_bhatt_distance = max_bhatt_distance
        self.fft_weight = fft_weight
        self.verification_mode = verification_mode
        self.default_threshold = default_threshold

        # Per-class thresholds for verification mode
        self._per_class_thresholds: dict[str, float] = {}
        self._load_thresholds(threshold_config)

        # ResNet50 feature extractor (with annular masking for inner hole)
        self._resnet = _ResNetFeatureExtractor(device=device, inner_ratio=inner_ratio)

        # All class templates for NN comparison
        self._templates: dict[str, dict] = {}
        self._active_classes: set[str] = set()  # empty = use all

        # Per-cycle distance caches (populated by find_nearest_*)
        self._last_color_distances: dict[str, float] = {}
        self._last_resnet_distances: dict[str, float] = {}

        # CSV for fusion classifier training data
        self._fusion_csv = Path(template_dir).parent / "fusion_training.csv"

        logger.info(f"TubePatternMatcher initialized | template_dir={template_dir} | fft_weight={fft_weight}")

    def _load_thresholds(self, config_path: str) -> None:
        """Load per-class verification thresholds from JSON config.

        Looks for tube_verify_config.json in:
        1. Explicit config_path if provided
        2. template_dir parent directory
        3. template_dir itself

        The JSON must have a "classes" key with per-class entries containing
        "threshold" values (combined distance thresholds for 0% false accept).
        """
        search_paths = []
        if config_path:
            search_paths.append(Path(config_path))
        search_paths.append(self.template_dir.parent / "tube_verify_config.json")
        search_paths.append(self.template_dir / "tube_verify_config.json")

        for p in search_paths:
            if p.exists():
                try:
                    with open(p) as f:
                        data = json.load(f)
                    classes = data.get("classes", {})
                    for cls_name, info in classes.items():
                        if "threshold" in info:
                            self._per_class_thresholds[cls_name] = float(info["threshold"])
                    logger.info(
                        "Loaded %d per-class thresholds from %s",
                        len(self._per_class_thresholds), p,
                    )
                    return
                except Exception as e:
                    logger.warning("Failed to load thresholds from %s: %s", p, e)

        if self.verification_mode:
            logger.warning(
                "Verification mode enabled but no threshold config found — "
                "using default_threshold=%.4f for all classes", self.default_threshold,
            )

    def set_active_classes(self, classes: set[str]) -> None:
        """Set active classes for matching. Only these templates will be used.

        Args:
            classes: Set of master_names to match against. Empty = use all.
        """
        self._active_classes = classes
        if classes:
            active = [c for c in classes if c in self._templates]
            logger.info("Active classes set: %s (%d/%d loaded)", classes, len(active), len(self._templates))
        else:
            logger.info("Active classes cleared — using all %d templates", len(self._templates))

    @property
    def _match_templates(self) -> dict[str, dict]:
        """Return templates filtered by active classes."""
        if not self._active_classes:
            return self._templates
        return {k: v for k, v in self._templates.items() if k in self._active_classes}

    def load_all_references(self) -> int:
        """Load all reference templates from template_dir.

        Must be called before verify() for NN classification to work.

        Returns:
            Number of templates loaded.
        """
        self._templates.clear()

        if not self.template_dir.exists():
            logger.warning(f"Template directory does not exist: {self.template_dir}")
            return 0

        for ref_path in self.template_dir.glob("*.npz"):
            material_id = ref_path.stem
            try:
                data = np.load(str(ref_path), allow_pickle=False)

                template = {}

                # Color histogram
                if "color_hist_mean" in data:
                    template["histogram"] = data["color_hist_mean"].astype(np.float32)
                elif "color_hist" in data:
                    template["histogram"] = data["color_hist"].astype(np.float32)
                else:
                    logger.warning(f"No color histogram in {ref_path}")
                    continue

                # Normalize histogram
                template["histogram"] = template["histogram"] / (template["histogram"].sum() + 1e-7)

                # Compute entropy from histogram for pattern structure matching
                template["entropy"] = compute_2d_entropy(template["histogram"])

                # HSV H-S histogram (optional — added for violet/white separation)
                if "hsv_hist_mean" in data:
                    template["hsv_histogram"] = data["hsv_hist_mean"].astype(np.float32)
                    template["hsv_histogram"] = template["hsv_histogram"] / (template["hsv_histogram"].sum() + 1e-7)
                elif "hsv_histogram" in data:
                    template["hsv_histogram"] = data["hsv_histogram"].astype(np.float32)
                    template["hsv_histogram"] = template["hsv_histogram"] / (template["hsv_histogram"].sum() + 1e-7)

                # ResNet features
                if "resnet_mean_feat" in data:
                    template["resnet_feat"] = data["resnet_mean_feat"].astype(np.float32)
                elif "resnet_feats" in data:
                    feats = data["resnet_feats"].astype(np.float32)
                    mean_feat = feats.mean(axis=0)
                    template["resnet_feat"] = mean_feat / (np.linalg.norm(mean_feat) + 1e-8)
                else:
                    logger.warning(f"No ResNet features in {ref_path}")
                    continue

                # FFT features (optional — backward compatible with old .npz)
                if "fft_mean_feat" in data:
                    template["fft_feat"] = data["fft_mean_feat"].astype(np.float32)
                elif "fft_feats" in data:
                    fft_feats = data["fft_feats"].astype(np.float32)
                    fft_mean = fft_feats.mean(axis=0)
                    template["fft_feat"] = (fft_mean / (np.linalg.norm(fft_mean) + 1e-8)).astype(np.float32)

                # Mean lightness for lightness-based disambiguation
                if "color_mean_L_mean" in data:
                    template["mean_L"] = float(data["color_mean_L_mean"])

                self._templates[material_id] = template

                # Per-pattern threshold: fall back to .npz color_threshold
                # if tube_verify_config.json has no entry for this pattern.
                # Priority: tube_verify_config.json > .npz color_threshold > global default
                if material_id not in self._per_class_thresholds:
                    if "color_threshold" in data:
                        self._per_class_thresholds[material_id] = float(data["color_threshold"])
                        logger.debug(
                            "Threshold for '%s': loaded from .npz (%.4f)",
                            material_id, float(data["color_threshold"]),
                        )

            except Exception as e:
                logger.error(f"Failed to load {ref_path}: {e}")
                continue

        n_fft = sum(1 for t in self._templates.values() if "fft_feat" in t)
        logger.info(f"Loaded {len(self._templates)} tube pattern templates ({n_fft} with FFT features)")
        return len(self._templates)

    def load_reference(self, material_id: str) -> Optional[dict]:
        """Get template for a specific material ID.

        Args:
            material_id: Material identifier.

        Returns:
            Template dict or None if not found.
        """
        if material_id in self._templates:
            return self._templates[material_id]

        # Try loading from file if not in memory
        ref_path = self.template_dir / f"{material_id}.npz"
        if ref_path.exists():
            self.load_all_references()
            return self._templates.get(material_id)

        return None

    def extract_resnet_features(self, bgr_image: np.ndarray, apply_mask: bool = True) -> np.ndarray:
        """Extract ResNet50 features from a tube crop.

        Args:
            bgr_image: BGR tube crop image.
            apply_mask: If True, apply annular mask to black out corners and
                inner hole. Set to False if image is already masked.

        Returns:
            L2-normalized 2048-dim feature vector.
        """
        return self._resnet.extract(bgr_image, apply_mask=apply_mask)

    def compute_color_signature(self, bgr_image: np.ndarray) -> Optional[dict]:
        """Extract color signature (LAB a*b* histogram + HSV H-S histogram)."""
        cropped_img, center, radius = find_radius(bgr_image)

        if cropped_img is None:
            logger.warning("Color signature: find_radius returned None")
            return None

        lab_patch = preprocess_cone_tip(
            cropped_img, center, radius,
            inner_crop_pct=self.inner_crop_pct,
            outer_crop_pct=self.outer_crop_pct,
            bilateral_d=self.bilateral_d,
            bilateral_sigma_color=self.bilateral_sigma_color,
            bilateral_sigma_space=self.bilateral_sigma_space,
        )

        sig = get_statistical_signature(lab_patch)

        # HSV H-S histogram on BGR polar patch (violet vs white separation)
        from .color_matching.bilateral_filter import apply_bilateral_filter
        from .color_matching.unrolled import unroll_cone_tip
        from .color_matching.crop_sweet_spot import crop_polar_sweet_spot

        filtered_bgr = apply_bilateral_filter(
            cropped_img, self.bilateral_d,
            self.bilateral_sigma_color, self.bilateral_sigma_space,
        )
        mask = (cropped_img > 0).any(axis=2)
        filtered_bgr[~mask] = 0
        bgr_polar = unroll_cone_tip(filtered_bgr, center, radius)
        bgr_patch = crop_polar_sweet_spot(bgr_polar, self.inner_crop_pct, self.outer_crop_pct)

        sig["hsv_histogram"] = compute_hs_histogram(bgr_patch)

        return sig

    def find_nearest_color(
        self, histogram: np.ndarray, hsv_histogram: np.ndarray = None,
        mean_L: Optional[float] = None,
    ) -> tuple[str, float]:
        """Find nearest class by Bhattacharyya distance (LAB + HSV + L* penalty).

        Combines LAB a*b* histogram distance with HSV H-S histogram distance,
        plus a mean-lightness penalty to disambiguate classes with similar
        a*b* chromaticity but different lightness (e.g. violet vs white).

        Args:
            histogram: LAB a*b* histogram (32x32).
            hsv_histogram: Optional HSV H-S histogram (32x32).
            mean_L: Optional mean L* value of the test image (0-255 scale).

        Returns:
            Tuple of (nearest_class, combined_distance).
        """
        histogram = histogram / (histogram.sum() + 1e-7)

        hsv_weight = 0.3  # blend: 70% LAB + 30% HSV
        lightness_weight = 0.50  # penalty weight for L* mismatch
        lightness_scale = 100.0  # practical L* range for normalization

        distances = {}
        for cls_name, template in self._match_templates.items():
            lab_dist = compute_bhattacharyya_distance(histogram, template["histogram"])

            if hsv_histogram is not None and "hsv_histogram" in template:
                hsv_dist = compute_bhattacharyya_distance(hsv_histogram, template["hsv_histogram"])
                combined = (1 - hsv_weight) * lab_dist + hsv_weight * hsv_dist
            else:
                combined = lab_dist

            # Lightness penalty: penalize distance when mean L* differs
            if mean_L is not None and "mean_L" in template:
                l_diff = abs(mean_L - template["mean_L"])
                l_penalty = lightness_weight * (l_diff / lightness_scale)
                combined += l_penalty

            distances[cls_name] = combined

        self._last_color_distances = distances
        if not distances:
            return "", float("inf")

        nearest_class = min(distances, key=distances.get)
        return nearest_class, distances[nearest_class]

    def find_nearest_resnet(self, feat: np.ndarray) -> tuple[str, float]:
        """Find nearest class by cosine distance.

        Args:
            feat: Test image ResNet features (2048-dim, L2-normalized).

        Returns:
            Tuple of (nearest_class, distance).
        """
        distances = {}
        for cls_name, template in self._match_templates.items():
            distances[cls_name] = _cosine_distance(feat, template["resnet_feat"])

        self._last_resnet_distances = distances
        if not distances:
            return "", float("inf")

        nearest_class = min(distances, key=distances.get)
        return nearest_class, distances[nearest_class]

    @staticmethod
    def linearize_ring(bgr_image: np.ndarray) -> Optional[np.ndarray]:
        """Polar-unwrap the annular ring into a clean rectangular strip.

        Finds ring geometry from the annular-masked crop, unwraps via
        cv2.warpPolar(), crops to just the ring band, removes black rows/cols.

        Args:
            bgr_image: Annular-masked tube crop (donut shape, black bg).

        Returns:
            Clean BGR strip (~670x25) or None if ring cannot be found.
        """
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        mask = (gray > 5).astype(np.uint8)
        coords = cv2.findNonZero(mask)
        if coords is None:
            return None

        x, y, w, h = cv2.boundingRect(coords)
        cx = x + w // 2
        cy = y + h // 2
        outer_r = max(w, h) // 2

        # Find inner radius (first non-zero ring from center outward)
        inner_r = 0
        for r in range(1, outer_r):
            for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                px = int(cx + r * np.cos(angle))
                py = int(cy + r * np.sin(angle))
                if 0 <= px < bgr_image.shape[1] and 0 <= py < bgr_image.shape[0]:
                    if mask[py, px] > 0:
                        inner_r = r
                        break
            if inner_r > 0:
                break

        if inner_r == 0 or inner_r >= outer_r:
            return None

        # Polar unwrap
        angular_res = int(2 * np.pi * (inner_r + outer_r) / 2)
        radial_res = outer_r + 5
        polar = cv2.warpPolar(
            bgr_image, dsize=(radial_res, angular_res),
            center=(cx, cy), maxRadius=radial_res,
            flags=cv2.WARP_POLAR_LINEAR,
        )
        strip = polar[:, inner_r:outer_r]

        # Remove black rows/columns
        gray_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        col_ok = np.where((gray_strip > 10).mean(axis=0) > 0.5)[0]
        row_ok = np.where((gray_strip > 10).mean(axis=1) > 0.5)[0]
        if len(col_ok) > 0 and len(row_ok) > 0:
            strip = strip[row_ok[0]:row_ok[-1] + 1, col_ok[0]:col_ok[-1] + 1]

        if strip.size == 0:
            return None

        return strip

    @staticmethod
    def extract_fft_intensity(strip_bgr: np.ndarray, n_coeffs: int = 64) -> np.ndarray:
        """Extract 1D FFT magnitude from mean intensity profile.

        Perfectly shift-invariant: rotation changes phase only, not magnitude.
        The strip's vertical axis (rows) corresponds to the angular direction
        around the ring, so the mean intensity profile captures periodic
        patterns (stripes, triangles, checks) along the ring.

        Args:
            strip_bgr: Clean linearized strip from linearize_ring().
            n_coeffs: Number of FFT coefficients to keep.

        Returns:
            L2-normalized FFT magnitude vector (n_coeffs,).
        """
        gray = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
        profile = gray.mean(axis=1)
        profile = profile - profile.mean()
        fft = np.fft.rfft(profile)
        magnitude = np.abs(fft)
        if len(magnitude) > n_coeffs:
            magnitude = magnitude[:n_coeffs]
        else:
            magnitude = np.pad(magnitude, (0, n_coeffs - len(magnitude)))
        return (magnitude / (np.linalg.norm(magnitude) + 1e-8)).astype(np.float32)

    def find_nearest_combined(self, fft_feat: Optional[np.ndarray]) -> tuple[str, float, float]:
        """Find nearest class by weighted Color + FFT distance.

        Must be called AFTER find_nearest_color() which populates
        _last_color_distances. Uses the cached Bhattacharyya distances
        to avoid recomputation.

        combined = (1 - fft_weight) * bhatt + fft_weight * fft_cosine

        Falls back to color-only for templates without FFT features,
        or if fft_feat is None (linearization failed).

        Args:
            fft_feat: L2-normalized FFT magnitude vector, or None.

        Returns:
            (nearest_class, combined_distance, fft_distance_to_nearest).
        """
        if not self._last_color_distances:
            return "", float("inf"), 0.0

        distances = {}
        fft_dists = {}
        for cls_name, bhatt in self._last_color_distances.items():
            template = self._templates.get(cls_name, {})

            if "fft_feat" in template and fft_feat is not None and self.fft_weight > 0:
                fft_d = _cosine_distance(fft_feat, template["fft_feat"])
                combined = (1.0 - self.fft_weight) * bhatt + self.fft_weight * fft_d
            else:
                combined = bhatt  # fallback: color-only
                fft_d = 0.0

            distances[cls_name] = combined
            fft_dists[cls_name] = fft_d

        nearest = min(distances, key=distances.get)
        return nearest, distances[nearest], fft_dists[nearest]

    def _verify_threshold(
        self, tube_crop: np.ndarray, material_id: str
    ) -> TubePatternResult:
        """Verification mode: compute distance to expected template only.

        Does NOT classify against all templates. Computes combined distance
        (Color + FFT) to the expected template and compares against the
        per-class threshold. Much simpler and more reliable than classification.

        Args:
            tube_crop: BGR tube crop with black background.
            material_id: Expected material ID from PLC.

        Returns:
            TubePatternResult with verification results.
        """
        template = self._templates[material_id]
        threshold = self._per_class_thresholds.get(
            material_id, self.default_threshold,
        )

        # --- 1. Color distance to expected template ---
        logger.debug("Tube verify (threshold): %dx%d crop, expected='%s'",
                     tube_crop.shape[1], tube_crop.shape[0], material_id)
        color_sig = self.compute_color_signature(tube_crop)

        color_distance = 1.0
        if color_sig is not None:
            lab_hist = color_sig["histogram"]
            lab_hist = lab_hist / (lab_hist.sum() + 1e-7)
            lab_dist = compute_bhattacharyya_distance(lab_hist, template["histogram"])

            hsv_hist = color_sig.get("hsv_histogram")
            if hsv_hist is not None and "hsv_histogram" in template:
                hsv_dist = compute_bhattacharyya_distance(hsv_hist, template["hsv_histogram"])
                color_distance = 0.7 * lab_dist + 0.3 * hsv_dist
            else:
                color_distance = lab_dist

            # Lightness penalty for disambiguation (e.g. violet vs white)
            sample_L = color_sig.get("mean_L")
            if sample_L is not None and "mean_L" in template:
                l_diff = abs(sample_L - template["mean_L"])
                l_penalty = 0.50 * (l_diff / 100.0)
                color_distance += l_penalty

            logger.debug("  Color: LAB=%.4f HSV=%.4f combined=%.4f",
                         lab_dist,
                         hsv_dist if hsv_hist is not None and "hsv_histogram" in template else 0.0,
                         color_distance)
        else:
            logger.warning("Could not extract color signature")

        # --- 2. FFT distance to expected template ---
        fft_feat = None
        fft_dist = 0.0
        strip = self.linearize_ring(tube_crop)
        if strip is not None:
            fft_feat = self.extract_fft_intensity(strip)
            if "fft_feat" in template:
                fft_dist = _cosine_distance(fft_feat, template["fft_feat"])
            logger.debug("  FFT: dist=%.4f", fft_dist)
        else:
            logger.warning("  FFT: could not linearize ring — color-only")

        # --- 3. Combined distance ---
        if fft_feat is not None and "fft_feat" in template and self.fft_weight > 0:
            combined_distance = (1.0 - self.fft_weight) * color_distance + self.fft_weight * fft_dist
        else:
            combined_distance = color_distance

        # --- 4. Threshold decision ---
        color_match = combined_distance <= threshold
        logger.info(
            "Tube verify '%s': combined=%.4f threshold=%.4f → %s",
            material_id, combined_distance, threshold,
            "PASS" if color_match else "FAIL",
        )

        # --- 5. ResNet (disabled — templates have mismatched dimensions) ---
        resnet_nearest, resnet_distance, resnet_match = "", 1.0, False

        result = TubePatternResult(
            color_nearest=material_id,
            color_distance=color_distance,
            color_match=color_match,
            resnet_nearest=resnet_nearest,
            resnet_distance=resnet_distance,
            resnet_match=resnet_match,
            expected_class=material_id,
            reference_loaded=True,
            combined_nearest=material_id,
            combined_distance=combined_distance,
            fft_distance=fft_dist,
        )

        # --- Fusion feature logging ---
        self._log_fusion_features(
            material_id, material_id, resnet_nearest,
            color_sig, result,
        )

        return result

    def _verify_classification(
        self, tube_crop: np.ndarray, material_id: str
    ) -> TubePatternResult:
        """Classification mode: find nearest among all templates (legacy).

        Combined (Color + FFT) distance finds nearest class:
        - Color: Bhattacharyya on LAB a*b* histogram (dominant signal)
        - FFT: Cosine on 1D FFT magnitude (shift-invariant spatial)
        - combined = (1 - fft_weight) * bhatt + fft_weight * fft_cosine
        - Reject if bhatt distance > max_bhatt_distance (distance gate)
        - Reject if entropy delta > max_entropy_delta (entropy gate)
        - ResNet runs for monitoring/logging only

        Args:
            tube_crop: BGR tube crop with black background.
            material_id: Expected material ID from PLC.

        Returns:
            TubePatternResult with NN classification results.
        """
        # --- 1. Color features ---
        logger.debug("Tube verify (classification): %dx%d crop", tube_crop.shape[1], tube_crop.shape[0])
        color_sig = self.compute_color_signature(tube_crop)
        if color_sig is None:
            logger.warning("Could not extract color signature")
            color_nearest, color_distance = "", 1.0
        else:
            color_nearest, color_distance = self.find_nearest_color(
                color_sig["histogram"], color_sig.get("hsv_histogram"),
                mean_L=color_sig.get("mean_L"),
            )
            logger.debug("  Color NN: nearest='%s' dist=%.4f (entropy=%.3f, mean_L=%.1f)",
                         color_nearest, color_distance,
                         color_sig.get("entropy", 0), color_sig.get("mean_L", 0))

        # --- 2. FFT features (shift-invariant spatial) ---
        fft_feat = None
        strip = self.linearize_ring(tube_crop)
        if strip is not None:
            fft_feat = self.extract_fft_intensity(strip)
            logger.debug("  FFT: strip=%dx%d, %d coeffs", strip.shape[1], strip.shape[0], len(fft_feat))
        else:
            logger.warning("  FFT: could not linearize ring — falling back to color-only")

        # --- 3. Combined nearest (Color + FFT weighted distance) ---
        combined_nearest, combined_distance, fft_dist = self.find_nearest_combined(fft_feat)
        logger.debug("  Combined NN: nearest='%s' dist=%.4f (fft_dist=%.4f, w=%.2f)",
                     combined_nearest, combined_distance, fft_dist, self.fft_weight)

        # --- 4. Entropy gate ---
        entropy_rejected = False
        if color_sig is not None and combined_nearest:
            live_entropy = color_sig.get("entropy", 0)
            template = self._templates.get(combined_nearest)
            if template and "entropy" in template:
                entropy_delta = abs(live_entropy - template["entropy"])
                logger.info(
                    "  Entropy: live=%.3f template(%s)=%.3f delta=%.3f",
                    live_entropy, combined_nearest, template["entropy"], entropy_delta,
                )
                if self.max_entropy_delta > 0 and entropy_delta > self.max_entropy_delta:
                    entropy_rejected = True
                    logger.info(
                        "  Entropy gate: REJECT — delta=%.3f > max=%.3f",
                        entropy_delta, self.max_entropy_delta,
                    )

        # --- 5. Distance gate (on pure Bhattacharyya distance) ---
        distance_rejected = False
        if self.max_bhatt_distance > 0 and color_distance > self.max_bhatt_distance:
            distance_rejected = True
            logger.info(
                "  Distance gate: REJECT — bhatt=%.4f > max=%.4f",
                color_distance, self.max_bhatt_distance,
            )

        # --- 6. ResNet (disabled — templates have mismatched dimensions) ---
        resnet_nearest, resnet_distance = "", 1.0

        # --- 7. Decision ---
        color_match = (combined_nearest == material_id
                       and not entropy_rejected
                       and not distance_rejected)
        resnet_match = resnet_nearest == material_id

        result = TubePatternResult(
            color_nearest=color_nearest,
            color_distance=color_distance,
            color_match=color_match,
            resnet_nearest=resnet_nearest,
            resnet_distance=resnet_distance,
            resnet_match=resnet_match,
            expected_class=material_id,
            reference_loaded=True,
            combined_nearest=combined_nearest,
            combined_distance=combined_distance,
            fft_distance=fft_dist,
        )

        # --- Fusion feature logging for classifier training ---
        self._log_fusion_features(
            material_id, color_nearest, resnet_nearest,
            color_sig, result,
        )

        # Log result — include distance to expected class for debugging
        status = "PASS" if result.passed else "FAIL"
        expected_color_d = self._last_color_distances.get(material_id, -1)
        expected_fft_d = 0.0
        tmpl = self._templates.get(material_id, {})
        if fft_feat is not None and "fft_feat" in tmpl and self.fft_weight > 0:
            expected_fft_d = _cosine_distance(fft_feat, tmpl["fft_feat"])
            expected_combined_d = (1.0 - self.fft_weight) * expected_color_d + self.fft_weight * expected_fft_d
        else:
            expected_combined_d = expected_color_d
        logger.info(
            f"Tube verify '{material_id}': "
            f"combined_nn={combined_nearest}(d={combined_distance:.3f}) "
            f"color_nn={color_nearest}(bhatt={color_distance:.3f}) "
            f"fft_dist={fft_dist:.3f} "
            f"expected_d={expected_combined_d:.3f}(color={expected_color_d:.3f},fft={expected_fft_d:.3f}) "
            f"→ {status}"
        )

        return result

    def verify(
        self, tube_crop: np.ndarray, material_id: str
    ) -> TubePatternResult:
        """Run tube pattern verification.

        Two modes controlled by self.verification_mode:
        - True (default): Threshold mode — compute distance to expected template
          only, compare against per-class threshold. No classification.
        - False: Classification mode (legacy) — find nearest among all templates,
          check if nearest matches expected.

        Args:
            tube_crop: BGR tube crop with black background.
            material_id: Expected material ID from PLC.

        Returns:
            TubePatternResult.
        """
        # Ensure templates are loaded
        if not self._templates:
            logger.warning("No templates loaded — call load_all_references() first")
            self.load_all_references()

        if not self._templates:
            logger.error("Cannot verify tube — no templates available")
            return TubePatternResult(
                color_nearest="",
                color_distance=1.0,
                color_match=False,
                resnet_nearest="",
                resnet_distance=1.0,
                resnet_match=False,
                expected_class=material_id,
                reference_loaded=False,
            )

        if material_id not in self._templates:
            logger.error(f"Unknown material_id '{material_id}' — not in templates")
            return TubePatternResult(
                color_nearest="",
                color_distance=1.0,
                color_match=False,
                resnet_nearest="",
                resnet_distance=1.0,
                resnet_match=False,
                expected_class=material_id,
                reference_loaded=False,
            )

        if self.verification_mode:
            return self._verify_threshold(tube_crop, material_id)
        else:
            return self._verify_classification(tube_crop, material_id)

    def _log_fusion_features(
        self,
        expected_class: str,
        color_nearest: str,
        resnet_nearest: str,
        color_sig: Optional[dict],
        result: TubePatternResult,
    ):
        """Compute and log all fusion features to CSV for classifier training.

        Logs 10 features + metadata per cycle:
            bhatt_to_expected, bhatt_to_nearest, bhatt_margin,
            cosine_to_expected, cosine_to_nearest, cosine_margin,
            live_entropy, expected_entropy, entropy_delta,
            classifiers_agree
        """
        try:
            # --- Bhattacharyya features ---
            bhatt_to_expected = self._last_color_distances.get(expected_class, 1.0)
            bhatt_to_nearest = result.color_distance
            # Margin: gap between nearest and 2nd nearest (higher = more confident)
            color_sorted = sorted(self._last_color_distances.values())
            bhatt_margin = (color_sorted[1] - color_sorted[0]) if len(color_sorted) >= 2 else 0.0

            # --- Cosine features ---
            cosine_to_expected = self._last_resnet_distances.get(expected_class, 1.0)
            cosine_to_nearest = result.resnet_distance
            resnet_sorted = sorted(self._last_resnet_distances.values())
            cosine_margin = (resnet_sorted[1] - resnet_sorted[0]) if len(resnet_sorted) >= 2 else 0.0

            # --- Entropy features ---
            live_entropy = color_sig.get("entropy", 0.0) if color_sig else 0.0
            expected_template = self._templates.get(expected_class, {})
            expected_entropy = expected_template.get("entropy", 0.0)
            entropy_delta = abs(live_entropy - expected_entropy)

            # --- Agreement ---
            classifiers_agree = 1 if color_nearest == resnet_nearest else 0

            # --- Write CSV ---
            write_header = not self._fusion_csv.exists()
            self._fusion_csv.parent.mkdir(parents=True, exist_ok=True)

            with open(self._fusion_csv, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "timestamp", "expected_class",
                        "bhatt_to_expected", "bhatt_to_nearest", "bhatt_margin",
                        "cosine_to_expected", "cosine_to_nearest", "cosine_margin",
                        "live_entropy", "expected_entropy", "entropy_delta",
                        "classifiers_agree",
                        "color_nearest", "resnet_nearest",
                        "color_match", "resnet_match", "result",
                    ])
                writer.writerow([
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    expected_class,
                    f"{bhatt_to_expected:.6f}", f"{bhatt_to_nearest:.6f}", f"{bhatt_margin:.6f}",
                    f"{cosine_to_expected:.6f}", f"{cosine_to_nearest:.6f}", f"{cosine_margin:.6f}",
                    f"{live_entropy:.4f}", f"{expected_entropy:.4f}", f"{entropy_delta:.4f}",
                    classifiers_agree,
                    color_nearest, resnet_nearest,
                    int(result.color_match), int(result.resnet_match), int(result.passed),
                ])

            logger.debug(
                "Fusion CSV: bhatt=[exp=%.4f nn=%.4f margin=%.4f] "
                "cosine=[exp=%.4f nn=%.4f margin=%.4f] "
                "entropy=[live=%.3f exp=%.3f delta=%.3f] agree=%d",
                bhatt_to_expected, bhatt_to_nearest, bhatt_margin,
                cosine_to_expected, cosine_to_nearest, cosine_margin,
                live_entropy, expected_entropy, entropy_delta, classifiers_agree,
            )
        except Exception as e:
            logger.warning("Failed to log fusion features: %s", e)

    def clear_cache(self, material_id: Optional[str] = None):
        """Clear cached templates.

        Args:
            material_id: Clear specific material. None clears all.
        """
        if material_id is None:
            self._templates.clear()
        else:
            self._templates.pop(material_id, None)
