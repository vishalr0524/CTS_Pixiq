"""
Tube Teacher — teaches tube patterns from full-frame camera images.

Takes N full-frame images (typically 5+), runs YOLO to extract the
yarn_tube ROI from each, then extracts:
  - Color signatures (histogram + entropy + mean_L)
  - FFT intensity features (64-dim, shift-invariant spatial)
  - ResNet50 features (2048-dim, monitoring only)

Verification uses combined Color + FFT distance (nearest neighbor).

YOLO extraction is always part of the pipeline — the input is always
a full camera frame, never a pre-cropped tube.

Usage:
    teacher = TubeTeacher(
        yolo_weights="weights/visible_yolo.pt",
        template_dir="data/templates/tube",
        db_path="data/db/materials.db",
    )
    result = teacher.teach(frames=[frame1, ..., frame5], material_id="MAT-001")
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Handle both relative and absolute imports for different run contexts
try:
    from ..inspection.tube_pattern import TubePatternMatcher, _cosine_distance
    from ..inspection.color_matching.bhattacharyya_distance import compute_bhattacharyya_distance
    from ..inspection.yolo_detector import YOLODetector
except ImportError:
    from inspection.tube_pattern import TubePatternMatcher, _cosine_distance
    from inspection.color_matching.bhattacharyya_distance import compute_bhattacharyya_distance
    from inspection.yolo_detector import YOLODetector

logger = logging.getLogger(__name__)

MIN_IMAGES = 2


class TubeTeacher:
    """Teaches tube patterns from full-frame images via YOLO + feature extraction.

    Pipeline per image:
        full frame → YOLO → extract yarn_tube ROI →
        color signature (polar strip) + ResNet50 features (raw crop)

    Storage:
        Features: {template_dir}/{material_id}.npz
    """

    def __init__(
        self,
        yolo_weights: str = "weights/visible_yolo.pt",
        yolo_conf: float = 0.6,
        template_dir: str = "data/templates/tube",
        bilateral_d: int = 9,
        bilateral_sigma_color: int = 75,
        bilateral_sigma_space: int = 75,
        device: str = "auto",
    ):
        """Initialize teacher with YOLO detector and feature extractor.

        Args:
            yolo_weights: Path to visible YOLO12 weights.
            yolo_conf: YOLO confidence threshold.
            template_dir: Directory to save .npz reference files.
            bilateral_d: Bilateral filter diameter.
            bilateral_sigma_color: Bilateral filter sigma in color space.
            bilateral_sigma_space: Bilateral filter sigma in coordinate space.
            device: PyTorch device for ResNet ("auto", "cuda", "cpu").
        """
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(parents=True, exist_ok=True)

        # YOLO detector for tube extraction
        self.detector = YOLODetector(
            model_path=yolo_weights,
            conf_threshold=yolo_conf,
        )

        # Feature extractor (reuses TubePatternMatcher for consistency)
        self.matcher = TubePatternMatcher(
            template_dir=str(self.template_dir),
            bilateral_d=bilateral_d,
            bilateral_sigma_color=bilateral_sigma_color,
            bilateral_sigma_space=bilateral_sigma_space,
            device=device,
        )

        logger.info(
            f"TubeTeacher initialized | yolo={yolo_weights} | "
            f"template_dir={template_dir}"
        )

    def _extract_rois(self, frame: np.ndarray) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Run YOLO on a full frame and extract cone + tube ROIs.

        Tube crop uses annular mask on tube_det bbox:
        - Outer corners zeroed (background outside the tube circle)
        - Inner hole zeroed (empty center of the tube)
        - inner_ratio = hole_dia / tube_outer_dia (from matcher config)

        Returns:
            (cone_crop, tube_masked) — either may be None if not detected.
        """
        detections = self.detector.detect(frame)
        cone_det = self.detector.get_detection_by_class(detections, "yarn_cone")
        tube_det = self.detector.get_detection_by_class(detections, "yarn_tube")
        cone_crop = self.detector.extract_roi(frame, cone_det) if cone_det else None
        tube_masked = (
            self.detector.extract_annular_roi(
                frame, tube_det, inner_ratio=self.matcher.inner_ratio
            )
            if tube_det else None
        )
        return cone_crop, tube_masked

    def _compute_p99_self_distance(
        self,
        color_hists: np.ndarray,
        hsv_hists: Optional[np.ndarray],
        mean_Ls: np.ndarray,
    ) -> float:
        """Compute p99 pairwise combined distance across all teaching samples.

        Mirrors the distance formula used in _verify_threshold exactly:
            color_dist = 0.7 * LAB_bhatt + 0.3 * HSV_bhatt  (if HSV available)
            l_penalty  = 0.50 * abs(mean_L[i] - mean_L[j]) / 100.0
            combined   = color_dist + l_penalty

        This ensures color_threshold is calibrated on the same distance that
        verify() uses at inspection time — no systematic bias.

        Args:
            color_hists: (N, H, W) LAB a*b* histograms from teaching samples.
            hsv_hists:   (N, H, W) HSV H-S histograms, or None if not available.
            mean_Ls:     (N,) mean L* values from teaching samples.

        Returns:
            p99 pairwise combined distance (scalar float).
        """
        n = len(color_hists)
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                lab_d = compute_bhattacharyya_distance(color_hists[i], color_hists[j])
                if hsv_hists is not None:
                    hsv_d = compute_bhattacharyya_distance(hsv_hists[i], hsv_hists[j])
                    color_d = 0.7 * lab_d + 0.3 * hsv_d
                else:
                    color_d = lab_d
                l_penalty = 0.50 * abs(float(mean_Ls[i]) - float(mean_Ls[j])) / 100.0
                distances.append(color_d + l_penalty)
        if not distances:
            return 0.0
        return float(np.percentile(distances, 99))

    def teach(
        self,
        frames: list[np.ndarray],
        material_id: str,
        save_crops_dir: str = None,
        pre_cropped: bool = False,
    ) -> dict:
        """Teach a tube pattern from images.

        Args:
            frames: List of BGR images. If pre_cropped=False (default), these
                    are full frames — YOLO runs to extract the tube annular crop.
                    If pre_cropped=True, these are already 256x256 annular crops
                    (e.g. saved by auto-capture) — YOLO is skipped entirely.
            material_id: Material identifier.
            save_crops_dir: If provided, save extracted crops here for verification.
            pre_cropped: If True, frames are pre-extracted annular crops —
                         skip YOLO detection. Reduces training compute significantly.

        Returns:
            Dict with teaching results.

        Raises:
            ValueError: If fewer than MIN_IMAGES valid crops are found.
        """
        n_input = len(frames)
        logger.info(f"Teaching '{material_id}' from {n_input} frames...")

        # --- YOLO extraction + feature extraction per frame ---
        color_sigs = []
        tube_crops = []
        tube_sizes = []
        resnet_feats = []
        fft_feats = []

        for i, frame in enumerate(frames):
            if pre_cropped:
                # Frame is already a 256x256 annular crop — skip YOLO entirely
                tube_crop = frame
                cone_crop = None
                if tube_crop is None or tube_crop.size == 0:
                    logger.warning(f"  Crop {i+1}/{n_input}: empty crop, skipping")
                    continue
            else:
                cone_crop, tube_crop = self._extract_rois(frame)
                if tube_crop is None:
                    logger.warning(f"  Frame {i+1}/{n_input}: no tube detected, skipping")
                    continue

            h, w = tube_crop.shape[:2]
            tube_sizes.append(f"{w}x{h}")
            cone_info = f"cone {cone_crop.shape[1]}x{cone_crop.shape[0]}" if cone_crop is not None else "pre-cropped"
            logger.info(f"  Frame {i+1}/{n_input}: tube ROI {w}x{h}, {cone_info}")

            sig = self.matcher.compute_color_signature(tube_crop)

            if sig is None:
                logger.warning(f"  Frame {i+1}/{n_input}: color extraction failed, skipping")
                continue

            feat = self.matcher.extract_resnet_features(tube_crop, apply_mask=False)

            # FFT features: linearize ring → 1D FFT magnitude
            strip = self.matcher.linearize_ring(tube_crop)
            if strip is not None:
                fft_feat = self.matcher.extract_fft_intensity(strip)
                fft_feats.append(fft_feat)
                logger.debug(f"  Frame {i+1}: FFT strip={strip.shape[1]}x{strip.shape[0]}")
            else:
                logger.warning(f"  Frame {i+1}/{n_input}: FFT linearization failed")

            color_sigs.append(sig)
            tube_crops.append(tube_crop)
            resnet_feats.append(feat)

        n = len(color_sigs)
        if n < MIN_IMAGES:
            raise ValueError(
                f"Need at least {MIN_IMAGES} tube detections, "
                f"got {n} out of {n_input} frames"
            )

        # --- Color: compute mean histogram ---
        color_hists = np.array(
            [s["histogram"] for s in color_sigs], dtype=np.float32
        )  # (N, 32, 32)
        color_entropies = np.array(
            [s["entropy"] for s in color_sigs], dtype=np.float32
        )  # (N,)
        color_mean_Ls = np.array(
            [s["mean_L"] for s in color_sigs], dtype=np.float32
        )  # (N,)

        color_hist_mean = color_hists.mean(axis=0).astype(np.float32)
        total = color_hist_mean.sum()
        if total > 0:
            color_hist_mean = color_hist_mean / total

        color_entropy_mean = float(color_entropies.mean())
        color_mean_L_mean = float(color_mean_Ls.mean())

        # --- HSV: compute mean H-S histogram ---
        hsv_hists = [s["hsv_histogram"] for s in color_sigs if "hsv_histogram" in s]
        hsv_hist_mean = None
        if hsv_hists:
            hsv_hists_arr = np.array(hsv_hists, dtype=np.float32)
            hsv_hist_mean = hsv_hists_arr.mean(axis=0).astype(np.float32)
            total = hsv_hist_mean.sum()
            if total > 0:
                hsv_hist_mean = hsv_hist_mean / total
            logger.info(f"  HSV H-S histograms: {len(hsv_hists)}/{n} frames")

        # --- ResNet: compute mean feature ---
        resnet_feats_arr = np.stack(resnet_feats).astype(np.float32)  # (N, 2048)
        resnet_mean_feat = resnet_feats_arr.mean(axis=0)
        resnet_mean_feat = resnet_mean_feat / (np.linalg.norm(resnet_mean_feat) + 1e-8)

        # --- FFT: compute mean feature ---
        fft_feats_arr = None
        fft_mean_feat = None
        if fft_feats:
            fft_feats_arr = np.stack(fft_feats).astype(np.float32)  # (M, 64)
            fft_mean_feat = fft_feats_arr.mean(axis=0)
            fft_mean_feat = (fft_mean_feat / (np.linalg.norm(fft_mean_feat) + 1e-8)).astype(np.float32)
            logger.info(f"  FFT features: {len(fft_feats)}/{n} frames linearized")
        else:
            logger.warning("  No FFT features extracted — .npz will have color+resnet only")

        # --- Save .npz (Color NN + FFT + ResNet NN features) ---
        npz_path = self.template_dir / f"{material_id}.npz"
        save_kwargs = {
            # Color features for Color NN
            "color_hists": color_hists,
            "color_entropies": color_entropies,
            "color_mean_Ls": color_mean_Ls,
            "color_hist_mean": color_hist_mean,
            "color_entropy_mean": np.array(color_entropy_mean),
            "color_mean_L_mean": np.array(color_mean_L_mean),
            "n_references": np.array(n),
            # ResNet features for ResNet NN (monitoring)
            "resnet_feats": resnet_feats_arr,
            "resnet_mean_feat": resnet_mean_feat,
        }

        # HSV H-S histogram
        if hsv_hist_mean is not None:
            save_kwargs["hsv_hist_mean"] = hsv_hist_mean

        # FFT features (shift-invariant spatial)
        if fft_feats_arr is not None:
            save_kwargs["fft_feats"] = fft_feats_arr
            save_kwargs["fft_mean_feat"] = fft_mean_feat

        # --- Per-pattern threshold: p99 pairwise self-distance * 1.5 ---
        hsv_hists_for_thresh = np.array(hsv_hists, dtype=np.float32) if hsv_hists else None
        p99_self_dist = self._compute_p99_self_distance(
            color_hists, hsv_hists_for_thresh, color_mean_Ls
        )
        color_threshold = round(p99_self_dist * 1.5, 4)
        save_kwargs["color_threshold"] = np.array(color_threshold, dtype=np.float32)
        save_kwargs["extend_count"] = np.array(0, dtype=np.int32)
        logger.info(
            "Per-pattern threshold for '%s': p99=%.4f threshold=%.4f",
            material_id, p99_self_dist, color_threshold,
        )

        # --- Atomic save: write to temp then replace ---
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".npz.tmp", dir=str(self.template_dir)
        )
        os.close(tmp_fd)
        try:
            np.savez(tmp_path, **save_kwargs)
            os.replace(tmp_path, str(npz_path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        # --- Save tube YOLO crops for frontend display / review ---
        if save_crops_dir:
            try:
                tube_dir = Path(save_crops_dir) / "tube"
                tube_dir.mkdir(parents=True, exist_ok=True)

                for idx, tube_img in enumerate(tube_crops):
                    cv2.imwrite(str(tube_dir / f"{idx+1:03d}.png"), tube_img)

                logger.info(f"Saved {len(tube_crops)} tube crops to {tube_dir}/")
            except OSError as e:
                logger.warning(f"Could not save tube crops (non-fatal): {e}")

        # Invalidate matcher cache (in case this is a re-teach)
        self.matcher.clear_cache(material_id)

        result = {
            "material_id": material_id,
            "n_frames": n_input,
            "n_tubes_detected": n,
            "tube_sizes": tube_sizes,
            "template_path": str(npz_path),
            "file_size_bytes": npz_path.stat().st_size,
            "color_hist_shape": list(color_hists.shape),
            "color_threshold": color_threshold,
            "resnet_n_features": n,
            "fft_n_features": len(fft_feats),
        }

        logger.info(
            f"Teaching complete for '{material_id}': "
            f"{n}/{n_input} tubes, saved to {npz_path}"
        )
        return result

    def get_reference_info(self, material_id: str) -> Optional[dict]:
        """Get metadata for a taught material from .npz file on disk."""
        npz_path = self.template_dir / f"{material_id}.npz"
        if not npz_path.exists():
            return None
        try:
            data = np.load(str(npz_path), allow_pickle=False)
            n_images = int(data["n_references"]) if "n_references" in data else 0
        except Exception:
            n_images = 0

        import os
        stat = npz_path.stat()
        from datetime import datetime, timezone
        created = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat()
        updated = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

        color_threshold = None
        extend_count = 0
        try:
            if "color_threshold" in data:
                color_threshold = float(data["color_threshold"])
            if "extend_count" in data:
                extend_count = int(data["extend_count"])
        except Exception:
            pass

        return {
            "material_id": material_id,
            "template_path": str(npz_path),
            "n_images": n_images,
            "color_threshold": color_threshold,
            "extend_count": extend_count,
            "created_at": created,
            "updated_at": updated,
        }


    def extend(
        self,
        frames: list,
        material_id: str,
    ) -> dict:
        """Append new teaching samples to an existing .npz reference.

        Loads existing features from .npz, extracts features from new frames
        using the same pipeline as teach(), appends, recomputes mean features
        and per-pattern threshold.

        Capped at MAX_EXTEND_COUNT (3) extensions before a full re-teach is
        required — prevents threshold drift from incremental bad samples.

        Args:
            frames: List of full-frame BGR images.
            material_id: Material ID — must already have a .npz reference.

        Returns:
            Dict with updated n_references, new color_threshold, delta.

        Raises:
            ValueError: If no existing reference, extend cap reached,
                        or too few new tubes detected.
        """
        MAX_EXTEND_COUNT = 3

        npz_path = self.template_dir / f"{material_id}.npz"
        if not npz_path.exists():
            raise ValueError(
                f"No reference found for '{material_id}' — run full teach first"
            )

        # --- Load existing .npz ---
        data = np.load(str(npz_path), allow_pickle=False)

        extend_count = int(data["extend_count"]) if "extend_count" in data else 0
        if extend_count >= MAX_EXTEND_COUNT:
            raise ValueError(
                f"Extend limit reached for '{material_id}' "
                f"({extend_count}/{MAX_EXTEND_COUNT}) — full re-teach required"
            )

        old_threshold = float(data["color_threshold"]) if "color_threshold" in data else None

        # Load existing feature arrays
        color_hists_old = data["color_hists"].astype(np.float32)
        color_entropies_old = data["color_entropies"].astype(np.float32)
        color_mean_Ls_old = data["color_mean_Ls"].astype(np.float32)
        resnet_feats_old = data["resnet_feats"].astype(np.float32)
        hsv_hists_old = data["hsv_hist_mean"] if "hsv_hist_mean" in data else None
        fft_feats_old = data["fft_feats"].astype(np.float32) if "fft_feats" in data else None

        logger.info(
            "Extending '%s': existing=%d samples, extend_count=%d/%d",
            material_id, len(color_hists_old), extend_count, MAX_EXTEND_COUNT,
        )

        # --- Extract features from new frames (same pipeline as teach()) ---
        n_input = len(frames)
        color_sigs = []
        resnet_feats_new = []
        fft_feats_new = []

        for i, frame in enumerate(frames):
            _, tube_crop = self._extract_rois(frame)
            if tube_crop is None:
                logger.warning("  Frame %d/%d: no tube detected, skipping", i + 1, n_input)
                continue

            sig = self.matcher.compute_color_signature(tube_crop)
            if sig is None:
                logger.warning("  Frame %d/%d: color extraction failed, skipping", i + 1, n_input)
                continue

            feat = self.matcher.extract_resnet_features(tube_crop, apply_mask=False)
            strip = self.matcher.linearize_ring(tube_crop)
            if strip is not None:
                fft_feats_new.append(self.matcher.extract_fft_intensity(strip))

            color_sigs.append(sig)
            resnet_feats_new.append(feat)

        n_new = len(color_sigs)
        if n_new < 1:
            raise ValueError(
                f"No tubes detected in new frames for '{material_id}'"
            )

        logger.info("  New samples detected: %d/%d frames", n_new, n_input)

        # --- Append new features to existing arrays ---
        color_hists_new = np.array([s["histogram"] for s in color_sigs], dtype=np.float32)
        color_entropies_new = np.array([s["entropy"] for s in color_sigs], dtype=np.float32)
        color_mean_Ls_new = np.array([s["mean_L"] for s in color_sigs], dtype=np.float32)
        resnet_feats_new_arr = np.stack(resnet_feats_new).astype(np.float32)

        color_hists_all = np.concatenate([color_hists_old, color_hists_new], axis=0)
        color_entropies_all = np.concatenate([color_entropies_old, color_entropies_new])
        color_mean_Ls_all = np.concatenate([color_mean_Ls_old, color_mean_Ls_new])
        resnet_feats_all = np.concatenate([resnet_feats_old, resnet_feats_new_arr], axis=0)

        # --- Recompute means ---
        color_hist_mean = color_hists_all.mean(axis=0).astype(np.float32)
        total = color_hist_mean.sum()
        if total > 0:
            color_hist_mean = color_hist_mean / total

        color_entropy_mean = float(color_entropies_all.mean())
        color_mean_L_mean = float(color_mean_Ls_all.mean())

        resnet_mean_feat = resnet_feats_all.mean(axis=0)
        resnet_mean_feat = (resnet_mean_feat / (np.linalg.norm(resnet_mean_feat) + 1e-8)).astype(np.float32)

        # HSV: reuse existing mean (no per-sample HSV stored in .npz)
        hsv_hist_mean = hsv_hists_old

        # FFT: append if available
        fft_feats_all = None
        fft_mean_feat = None
        if fft_feats_old is not None and fft_feats_new:
            fft_feats_new_arr = np.stack(fft_feats_new).astype(np.float32)
            fft_feats_all = np.concatenate([fft_feats_old, fft_feats_new_arr], axis=0)
            fft_mean_feat = fft_feats_all.mean(axis=0)
            fft_mean_feat = (fft_mean_feat / (np.linalg.norm(fft_mean_feat) + 1e-8)).astype(np.float32)
        elif fft_feats_old is not None:
            fft_feats_all = fft_feats_old
            fft_mean_feat = data["fft_mean_feat"].astype(np.float32) if "fft_mean_feat" in data else None

        # --- Recompute per-pattern threshold ---
        hsv_for_thresh = None
        # Build per-sample HSV array if possible (not stored individually, use mean as proxy)
        p99_self_dist = self._compute_p99_self_distance(
            color_hists_all, hsv_for_thresh, color_mean_Ls_all
        )
        new_threshold = round(p99_self_dist * 1.5, 4)

        # --- Build save kwargs ---
        save_kwargs = {
            "color_hists": color_hists_all,
            "color_entropies": color_entropies_all,
            "color_mean_Ls": color_mean_Ls_all,
            "color_hist_mean": color_hist_mean,
            "color_entropy_mean": np.array(color_entropy_mean),
            "color_mean_L_mean": np.array(color_mean_L_mean),
            "n_references": np.array(len(color_hists_all)),
            "resnet_feats": resnet_feats_all,
            "resnet_mean_feat": resnet_mean_feat,
            "color_threshold": np.array(new_threshold, dtype=np.float32),
            "extend_count": np.array(extend_count + 1, dtype=np.int32),
        }
        if hsv_hist_mean is not None:
            save_kwargs["hsv_hist_mean"] = hsv_hist_mean
        if fft_feats_all is not None:
            save_kwargs["fft_feats"] = fft_feats_all
        if fft_mean_feat is not None:
            save_kwargs["fft_mean_feat"] = fft_mean_feat

        # --- Atomic save ---
        import tempfile as _tempfile
        tmp_fd, tmp_path = _tempfile.mkstemp(
            suffix=".npz.tmp", dir=str(self.template_dir)
        )
        os.close(tmp_fd)
        try:
            np.savez(tmp_path, **save_kwargs)
            os.replace(tmp_path, str(npz_path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        self.matcher.clear_cache(material_id)

        threshold_delta = round(new_threshold - old_threshold, 4) if old_threshold is not None else None
        logger.info(
            "Extended '%s': %d → %d samples, threshold %.4f → %.4f (delta=%s), extend_count=%d/%d",
            material_id, len(color_hists_old), len(color_hists_all),
            old_threshold or 0, new_threshold, threshold_delta,
            extend_count + 1, MAX_EXTEND_COUNT,
        )

        return {
            "material_id": material_id,
            "n_references_before": len(color_hists_old),
            "n_references_after": len(color_hists_all),
            "n_new_samples": n_new,
            "old_threshold": old_threshold,
            "new_threshold": new_threshold,
            "threshold_delta": threshold_delta,
            "extend_count": extend_count + 1,
            "extends_remaining": MAX_EXTEND_COUNT - (extend_count + 1),
            "template_path": str(npz_path),
        }

    def delete_reference(self, material_id: str) -> bool:
        """Delete reference .npz file from disk."""
        npz_path = self.template_dir / f"{material_id}.npz"
        file_existed = npz_path.exists()
        if file_existed:
            npz_path.unlink()

        self.matcher.clear_cache(material_id)

        if file_existed:
            logger.info(f"Deleted tube reference for '{material_id}'")
            return True
        return False

    def list_references(self) -> list[dict]:
        """List all taught materials by globbing .npz files from template_dir."""
        results = []
        for npz_path in sorted(self.template_dir.glob("*.npz")):
            material_id = npz_path.stem
            try:
                data = np.load(str(npz_path), allow_pickle=False)
                n_images = int(data["n_references"]) if "n_references" in data else 0
            except Exception:
                n_images = 0

            stat = npz_path.stat()
            from datetime import datetime, timezone
            created = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat()
            updated = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

            color_threshold = None
            extend_count = 0
            try:
                if "color_threshold" in data:
                    color_threshold = float(data["color_threshold"])
                if "extend_count" in data:
                    extend_count = int(data["extend_count"])
            except Exception:
                pass

            results.append({
                "material_id": material_id,
                "template_path": str(npz_path),
                "n_images": n_images,
                "color_threshold": color_threshold,
                "extend_count": extend_count,
                "created_at": created,
                "updated_at": updated,
            })
        return results
