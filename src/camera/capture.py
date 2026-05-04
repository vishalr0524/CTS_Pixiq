"""
Capture sequence module — orchestrates sequential image capture from 3 cameras.

Each camera is triggered by its own proximity sensor (hardware trigger on Line0).
Captures are sequential: VL → Tail → UV, each blocking until its sensor fires.

IMPORTANT: Every camera operation is wrapped in try/except so that one camera
crashing (disconnect, SDK error) never affects the other cameras.
"""

import logging
import time

from .camera import Camera
from .data_types import CapturedImages


logger = logging.getLogger(__name__)


class CaptureSequence:
    """Captures one complete part across 3 cameras sequentially.

    Each call to capture_part() blocks on 3 hardware triggers in
    physical conveyor order:
        Sensor 1 → VL image   (Visible Light)
        Sensor 2 → Tail image (Yarn Tail)
        Sensor 3 → UV image   (Ultraviolet)

    The timing between captures is determined by the conveyor speed and
    station spacing, not by software.

    All camera operations are isolated — if one camera crashes or
    disconnects, the others continue working independently.
    """

    def __init__(self, cam_vl: Camera, cam_uv: Camera, cam_tail: Camera,
                 device_manager=None):
        self.cam_vl = cam_vl
        self.cam_uv = cam_uv
        self.cam_tail = cam_tail
        self._device_manager = device_manager  # needed for reconnect

    def capture_part(self) -> CapturedImages:
        """Capture one complete part — VL, Tail, and UV images.

        Uses capture_latest() — drains any stale frames and keeps the
        freshest. With cycle_start gating, PLC only releases one cone at
        a time, so the latest frame is always the correct one.

        Each camera uses its own configured timeout (from config.json).
        Errors are handled per camera — if one camera crashes, the others
        still capture. The inspection pipeline handles None frames gracefully.

        Returns:
            CapturedImages with BGR numpy arrays (or None per camera
            if that camera timed out or errored).
        """
        t_start = time.perf_counter()
        vl = None
        uv = None
        tail = None

        # Flush VL buffer right before capture. Between the cycle-start flush
        # and now, the PLC trigger poll ran — a stale frame from sensor bounce
        # or vibration could have arrived in the buffer during that time.
        try:
            self.cam_vl.flush_buffers()
        except Exception as e:
            logger.warning("VL flush_buffers failed: %s", e)

        # Camera 1: Visible Light (first station on conveyor)
        try:
            logger.debug("Capture: waiting for VL trigger (timeout=%dms)...", self.cam_vl.timeout)
            vl = self.cam_vl.capture_latest()
            t_vl = time.perf_counter()
            logger.info(f"VL captured in {t_vl - t_start:.3f}s")
            logger.debug("  VL frame: %dx%d dtype=%s", vl.shape[1], vl.shape[0], vl.dtype)
        except TimeoutError:
            logger.warning("VL camera timeout (%dms) — cone may have missed sensor", self.cam_vl.timeout)
        except Exception as e:
            logger.error("VL camera error: %s — camera may be disconnected", e)

        # Flush Tail buffer after VL capture. The cone is now between VL
        # and Tail stations — any frame in Tail buffer is stale (sensor
        # bounce, vibration). Clears it right before Tail capture.
        try:
            self.cam_tail.flush_buffers()
        except Exception as e:
            logger.warning("Tail flush_buffers failed: %s", e)

        # Camera 2: Yarn Tail (second station on conveyor)
        try:
            t_tail_start = time.perf_counter()
            logger.debug("Capture: waiting for Tail trigger (timeout=%dms)...", self.cam_tail.timeout)
            tail = self.cam_tail.capture_latest()
            t_tail = time.perf_counter()
            logger.info(f"Tail captured in {t_tail - t_tail_start:.3f}s")
            logger.debug("  Tail frame: %dx%d dtype=%s", tail.shape[1], tail.shape[0], tail.dtype)
        except TimeoutError:
            logger.warning("Tail camera timeout (%dms) — cone may have missed sensor", self.cam_tail.timeout)
        except Exception as e:
            logger.error("Tail camera error: %s — camera may be disconnected", e)

        # Flush UV buffer after Tail capture. Same rationale as Tail flush:
        # if the UV sensor fired while VL/Tail were capturing, that frame is
        # stale. Flush right before UV capture so capture_latest() blocks for
        # the real trigger or gets the freshest frame.
        try:
            self.cam_uv.flush_buffers()
        except Exception as e:
            logger.warning("UV flush_buffers failed: %s", e)

        # Camera 3: UV (third station on conveyor)
        try:
            t_uv_start = time.perf_counter()
            logger.debug("Capture: waiting for UV trigger (timeout=%dms)...", self.cam_uv.timeout)
            uv = self.cam_uv.capture_latest()
            t_uv = time.perf_counter()
            logger.info(f"UV captured in {t_uv - t_uv_start:.3f}s")
            logger.debug("  UV frame: %dx%d dtype=%s", uv.shape[1], uv.shape[0], uv.dtype)
        except TimeoutError:
            logger.warning("UV camera timeout (%dms) — cone may have missed sensor", self.cam_uv.timeout)
        except Exception as e:
            logger.error("UV camera error: %s — camera may be disconnected", e)

        t_end = time.perf_counter()
        captured = sum(1 for f in (vl, uv, tail) if f is not None)
        logger.info(f"Total capture: {t_end - t_start:.3f}s ({captured}/3 cameras)")

        return CapturedImages(vl=vl, uv=uv, tail=tail)

    def stop_acquisition(self):
        """Stop acquisition on all 3 cameras and flush buffers.

        Call on inspection stop. After this, proximity sensor triggers
        are ignored and no frames accumulate in the buffer.
        Per-camera errors are logged but don't stop the other cameras.
        """
        for cam in (self.cam_vl, self.cam_uv, self.cam_tail):
            try:
                cam.stop_acquisition()
            except Exception as e:
                logger.warning(f"Camera '{cam.name}': stop_acquisition failed: {e}")

    def start_acquisition(self):
        """Restart acquisition on all 3 cameras.

        Call on inspection start. Buffer is empty — first capture()
        will block until a real hardware trigger fires.

        If a camera fails to start, attempts reconnect once. If reconnect
        also fails, that camera is skipped — capture_part() will return
        None for it (handled gracefully by the inspection pipeline).
        """
        for cam in (self.cam_vl, self.cam_uv, self.cam_tail):
            try:
                cam.start_acquisition()
            except Exception as e:
                logger.error(
                    f"Camera '{cam.name}': start_acquisition failed: {e} — "
                    f"attempting reconnect"
                )
                if self._device_manager is not None:
                    try:
                        cam.reconnect(self._device_manager)
                    except Exception as re:
                        logger.error(
                            f"Camera '{cam.name}': reconnect failed: {re} — "
                            f"camera unavailable this session"
                        )
                else:
                    logger.warning(
                        f"Camera '{cam.name}': no device manager for reconnect — "
                        f"camera unavailable this session"
                    )

    def flush_buffers(self):
        """Flush stale frames from all camera buffers (non-blocking).

        Call before starting a capture loop to discard any frames that
        triggered while the system was idle or between cycles.
        """
        for cam in (self.cam_vl, self.cam_uv, self.cam_tail):
            try:
                cam.flush_buffers()
            except Exception as e:
                logger.warning(f"Camera '{cam.name}': flush_buffers failed: {e}")

    def flush_all(self, timeout_ms: int = 2000):
        """Consume and discard triggers from all 3 cameras.

        Used when a part is rejected (mat_no=0 or master not found) but
        is already on the conveyor — sensors WILL fire regardless, and
        we must consume those triggers to stay in sync.

        Args:
            timeout_ms: Max wait time per camera.
        """
        for cam in (self.cam_vl, self.cam_uv, self.cam_tail):
            try:
                cam.capture(timeout_ms)
            except TimeoutError:
                logger.warning(
                    f"Flush timeout on camera '{cam.name}' — "
                    f"conveyor may have stopped"
                )
            except Exception as e:
                logger.warning(
                    f"Camera '{cam.name}': flush_all failed: {e}"
                )

    def log_stream_statistics(self):
        """Log stream statistics for all 3 cameras. Call periodically or on stop."""
        for cam in (self.cam_vl, self.cam_uv, self.cam_tail):
            try:
                cam.log_stream_statistics()
            except Exception as e:
                logger.warning(f"Camera '{cam.name}': stream stats failed: {e}")

    def get_stream_statistics(self) -> dict:
        """Get stream statistics for all 3 cameras.

        Returns:
            Dict mapping camera name to stats dict.
        """
        result = {}
        for cam in (self.cam_vl, self.cam_uv, self.cam_tail):
            try:
                result[cam.name] = cam.get_stream_statistics()
            except Exception:
                result[cam.name] = {"camera": cam.name, "error": "unavailable"}
        return result

    def health_check(self) -> dict:
        """Check connection status of all 3 cameras.

        Returns:
            Dict mapping camera name to health status (bool).
        """
        result = {}
        for cam in (self.cam_vl, self.cam_uv, self.cam_tail):
            try:
                result[cam.name] = cam.health_check()
            except Exception:
                result[cam.name] = False
        return result
