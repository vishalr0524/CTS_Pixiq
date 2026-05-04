"""
Camera module — wraps Basler pypylon SDK for GigE cameras.

One Camera instance per physical camera. Instances are created once at app
startup and reused across start/stop cycles.

Supported cameras:
    - acA1920-40gc    (ace classic, UV station)
    - a2A1920-40gc    (ace 2, Tail station)
    - a2A2600-20gcPRO (ace 2 PRO, VL station)

Both are GigE Vision cameras identified by static IP address on the
factory floor network. Hardware trigger on Line1 (proximity sensor).

SDK library lifecycle is managed by pypylon internally (no explicit
Initialize/Close needed).
"""

import logging

import numpy as np
from pypylon import pylon, genicam


logger = logging.getLogger(__name__)

BUFFER_COUNT = 5


class Camera:
    """Wrapper for a single Basler GigE camera via pypylon.

    Lifecycle:
        1. __init__()   — stores config (name, ip, exposure). No SDK calls.
        2. connect()    — opens device, configures trigger, starts acquisition.
        3. capture()    — blocks until sensor trigger, returns BGR image.
        4. disconnect() — stops acquisition, releases handle.
        Steps 2-4 can repeat. Instance stays alive.
    """

    def __init__(
        self,
        name: str,
        exposure: int,
        timeout: int = 30000,
        ip: str | None = None,
        serial: str | None = None,
        trigger_debounce_us: int = 0,
    ):
        """Create a camera instance. No SDK calls — just stores config.

        Args:
            name: Human-readable name ("VL", "UV", "Tail").
            exposure: Exposure time in microseconds.
            timeout: Default capture timeout in milliseconds.
            ip: Static IP address (GigE cameras).
            serial: Serial number (alternative identification).
            trigger_debounce_us: Trigger debounce time in microseconds.
                Ignores rapid-fire triggers (sensor bounce/vibration) shorter
                than this interval. 0 = disabled. Typical: 200000 (200ms).
        """
        if not ip and not serial:
            raise ValueError(f"Camera '{name}': must provide either ip or serial")
        self.name = name
        self.ip = ip
        self.serial = serial
        self.exposure = exposure
        self.timeout = timeout
        self.trigger_debounce_us = trigger_debounce_us

        # pypylon handles — set on connect(), cleared on disconnect()
        self._camera: pylon.InstantCamera | None = None
        self._converter: pylon.ImageFormatConverter | None = None
        self._trigger_mode: str | None = None

        # Observability — software-side counters (camera-side via CounterSelector)
        self._frames_delivered: int = 0
        self._frames_skipped: int = 0
        self._last_block_id: int = 0
        self._block_id_gaps: int = 0
        self._last_timestamp: int = 0
        self._is_ace2: bool = False
        self._counter_available: bool = False

    @property
    def connected(self) -> bool:
        return self._camera is not None and self._camera.IsOpen()

    # ── Connection lifecycle ──────────────────────────────────────────

    def connect(self, devices=None):
        """Open the camera by IP or serial, configure trigger, start acquisition.

        Args:
            devices: Unused — kept for interface compatibility.
                pypylon discovers devices internally via TlFactory.

        Raises:
            RuntimeError: If the camera is not found on the network.
            genicam.GenericException: If SDK calls fail during setup.
        """
        if self.connected:
            logger.warning(f"Camera '{self.name}' already connected — disconnecting first")
            self.disconnect()

        tl_factory = pylon.TlFactory.GetInstance()

        # Find device by IP or serial
        di = pylon.DeviceInfo()
        if self.ip:
            di.SetIpAddress(self.ip)
        elif self.serial:
            di.SetSerialNumber(self.serial)

        try:
            self._camera = pylon.InstantCamera(tl_factory.CreateFirstDevice(di))
        except genicam.GenericException as e:
            ident = self.ip or self.serial
            raise RuntimeError(
                f"Camera '{self.name}' at {ident} not found — check network/power"
            ) from e

        self._camera.Open()

        ident = self.ip or self.serial
        model = self._camera.GetDeviceInfo().GetModelName()
        logger.info(f"Camera '{self.name}' opened: {model} [{ident}]")

        # Buffer pool
        self._camera.MaxNumBuffer.Value = BUFFER_COUNT

        # Trigger setup — hardware Line1, RisingEdge, FrameStart
        self._camera.TriggerSelector.Value = "FrameStart"
        self._camera.TriggerMode.Value = "On"
        self.set_trigger("hardware")

        if genicam.IsAvailable(self._camera.TriggerActivation):
            current = self._camera.TriggerActivation.Value
            self._camera.TriggerActivation.Value = "RisingEdge"
            logger.info(f"Camera '{self.name}': TriggerActivation {current} → RisingEdge")

        # Trigger debounce — filter sensor bounce/vibration
        if self.trigger_debounce_us > 0:
            try:
                # Select the trigger input line before setting debounce
                if genicam.IsAvailable(self._camera.LineSelector):
                    self._camera.LineSelector.Value = "Line1"
                self._camera.LineDebouncerTime.Value = float(self.trigger_debounce_us)
                logger.info(
                    f"Camera '{self.name}': line debounce set to {self.trigger_debounce_us}us "
                    f"({self.trigger_debounce_us / 1000:.0f}ms)"
                )
            except genicam.GenericException as e:
                logger.warning(
                    f"Camera '{self.name}': trigger debounce not supported ({e})"
                )

        # Detect ace 2 vs ace classic — ace 2 has CounterTriggerSource
        self._is_ace2 = genicam.IsAvailable(
            getattr(self._camera, 'CounterTriggerSource', None)
        ) if hasattr(self._camera, 'CounterTriggerSource') else False
        # Safer detection: ace 2 models start with "a2A"
        if model.startswith("a2A"):
            self._is_ace2 = True

        # Hardware trigger counter — count Line1 rising edges via Counter1.
        # Comparing this against delivered frames reveals missed triggers.
        self._counter_available = False
        try:
            self._camera.CounterSelector.Value = "Counter1"
            self._camera.CounterEventSource.Value = "FrameStart"
            if self._is_ace2:
                # ace 2 requires explicit trigger source config
                self._camera.CounterTriggerSource.Value = "Off"
            self._camera.CounterReset.Execute()
            self._counter_available = True
            logger.info(f"Camera '{self.name}': trigger counter (Counter1) enabled on FrameStart")
        except Exception as e:
            logger.debug(f"Camera '{self.name}': trigger counter not supported ({e})")

        # Second counter — count raw Line1 edges (trigger input, before debounce).
        # Comparing Counter2 (raw triggers) vs Counter1 (frames) shows debounce filtering.
        self._line_counter_available = False
        try:
            self._camera.CounterSelector.Value = "Counter2"
            self._camera.CounterEventSource.Value = "Line1"
            if self._is_ace2:
                self._camera.CounterTriggerSource.Value = "Off"
                self._camera.CounterEventActivation.Value = "RisingEdge"
            self._camera.CounterReset.Execute()
            self._line_counter_available = True
            logger.info(f"Camera '{self.name}': line counter (Counter2) enabled on Line1 edges")
        except Exception as e:
            logger.debug(f"Camera '{self.name}': line counter not supported ({e})")

        # Reset software-side observability counters
        self._frames_delivered = 0
        self._frames_skipped = 0
        self._last_block_id = 0
        self._block_id_gaps = 0
        self._last_timestamp = 0

        # Exposure
        # ace classic uses ExposureTimeAbs, ace 2 uses ExposureTime
        if genicam.IsAvailable(self._camera.ExposureTime):
            self._camera.ExposureTime.Value = float(self.exposure)
        elif genicam.IsAvailable(self._camera.ExposureTimeAbs):
            self._camera.ExposureTimeAbs.Value = float(self.exposure)

        # GigE transport — optimize packet size for throughput
        try:
            self._camera.GevSCPSPacketSize.Value = 8192
        except genicam.GenericException:
            logger.info(f"Camera '{self.name}': jumbo frame packet size not supported, using default")

        # Inter-packet delay — spread packets to avoid NIC overflow
        try:
            if genicam.IsAvailable(self._camera.GevSCPD):
                self._camera.GevSCPD.Value = 1000
        except genicam.GenericException:
            pass

        # Enable packet resend for reliability
        try:
            stream_grabber = self._camera.GetStreamGrabberNodeMap()
            stream_grabber["EnableResend"].Value = True
        except Exception:
            pass

        # Image format converter — Bayer/Mono → BGR for OpenCV
        self._converter = pylon.ImageFormatConverter()
        self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        # Start acquisition with LatestImageOnly — auto-discards stale frames
        self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        debounce_str = f", debounce={self.trigger_debounce_us}us" if self.trigger_debounce_us > 0 else ""
        logger.info(
            f"Camera '{self.name}' ready — trigger={self._trigger_mode}, "
            f"exposure={self.exposure}us, buffers={BUFFER_COUNT}{debounce_str}"
        )

    def disconnect(self):
        """Stop acquisition and release device handle.

        Safe to call even if not connected or partially initialized.
        """
        if self._camera is None:
            return

        name = self.name

        try:
            if self._camera.IsGrabbing():
                self._camera.StopGrabbing()
        except genicam.GenericException as e:
            logger.warning(f"Camera '{name}': StopGrabbing failed: {e}")

        try:
            if self._camera.IsOpen():
                self._camera.Close()
        except genicam.GenericException as e:
            logger.warning(f"Camera '{name}': Close failed: {e}")

        self._camera = None
        self._converter = None
        self._trigger_mode = None

        logger.info(f"Camera '{name}' disconnected")

    def stop_acquisition(self):
        """Stop acquisition. Device stays open.

        After this call, no frames will be captured until
        start_acquisition() is called.
        """
        if not self.connected or self._camera is None:
            return

        try:
            if self._camera.IsGrabbing():
                self._camera.StopGrabbing()
        except genicam.GenericException as e:
            logger.warning(f"Camera '{self.name}': StopGrabbing failed: {e}")

        logger.info(f"Camera '{self.name}': acquisition stopped")

    def start_acquisition(self):
        """Restart acquisition after stop_acquisition().

        Safe to call if acquisition is already running (no-op).
        """
        if not self.connected or self._camera is None:
            return

        if self._camera.IsGrabbing():
            logger.debug(f"Camera '{self.name}': acquisition already running, skipping start")
            return

        self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.reset_trigger_count()
        logger.info(f"Camera '{self.name}': acquisition restarted")

    def reconnect(self, device_manager=None) -> bool:
        """Disconnect and reconnect. Returns True on success.

        Args:
            device_manager: Unused — kept for interface compatibility.
        """
        logger.info(f"Camera '{self.name}': attempting reconnect...")
        try:
            self.disconnect()
        except Exception as e:
            logger.warning(f"Camera '{self.name}': disconnect during reconnect failed: {e}")
            self._camera = None
            self._converter = None
            self._trigger_mode = None
        try:
            self.connect()
            logger.info(f"Camera '{self.name}': reconnect successful")
            return True
        except Exception as e:
            logger.error(f"Camera '{self.name}': reconnect failed: {e}")
            return False

    # ── Configuration ─────────────────────────────────────────────────

    def set_trigger(self, mode: str):
        """Set trigger mode.

        Args:
            mode: 'hardware' for Line1 (proximity sensor) or 'software' for
                  software-triggered capture (calibration/live feed).
        """
        mode = mode.lower()
        if mode == "hardware":
            self._camera.TriggerSource.Value = "Line1"
        elif mode == "software":
            self._camera.TriggerSource.Value = "Software"
        else:
            raise ValueError(f"Unknown trigger mode: {mode}")
        self._trigger_mode = mode

    def set_exposure(self, microseconds: int):
        """Set exposure time in microseconds."""
        if genicam.IsAvailable(self._camera.ExposureTime):
            self._camera.ExposureTime.Value = float(microseconds)
        elif genicam.IsAvailable(self._camera.ExposureTimeAbs):
            self._camera.ExposureTimeAbs.Value = float(microseconds)
        self.exposure = microseconds

    # ── Capture ───────────────────────────────────────────────────────

    def capture(self, timeout_ms: int | None = None) -> np.ndarray:
        """Capture a single frame, return as BGR numpy array.

        In hardware trigger mode: blocks until the proximity sensor fires.
        In software trigger mode: sends trigger command, then waits.

        Args:
            timeout_ms: Max wait time in ms. Defaults to self.timeout.

        Returns:
            BGR image as numpy array (uint8).

        Raises:
            TimeoutError: If no frame arrives within timeout.
            RuntimeError: If camera is not connected.
        """
        if not self.connected:
            raise RuntimeError(f"Camera '{self.name}' is not connected")

        if timeout_ms is None:
            timeout_ms = self.timeout

        if self._trigger_mode == "software":
            if self._camera.WaitForFrameTriggerReady(1000, pylon.TimeoutHandling_Return):
                self._camera.ExecuteSoftwareTrigger()

        try:
            grab_result = self._camera.RetrieveResult(
                timeout_ms, pylon.TimeoutHandling_ThrowException
            )
        except genicam.TimeoutException:
            raise TimeoutError(
                f"Camera '{self.name}' capture timeout ({timeout_ms}ms) — "
                f"part may not have arrived at sensor"
            )
        except genicam.GenericException as e:
            raise RuntimeError(f"Camera '{self.name}' grab error: {e}") from e

        try:
            bgr = self._grab_to_bgr(grab_result)
            self._track_grab_result(grab_result)
            return bgr
        finally:
            grab_result.Release()

    def _grab_to_bgr(self, grab_result) -> np.ndarray:
        """Convert a grab result to a BGR numpy array."""
        if not grab_result.GrabSucceeded():
            raise RuntimeError(
                f"Camera '{self.name}' grab failed: "
                f"{grab_result.ErrorCode} — {grab_result.ErrorDescription}"
            )

        if not self._converter.ImageHasDestinationFormat(grab_result):
            image = self._converter.Convert(grab_result)
            return image.GetArray().copy()
        else:
            return grab_result.Array.copy()

    def capture_latest(self, timeout_ms: int | None = None) -> np.ndarray:
        """Capture the latest frame, discarding any stale buffered frames.

        With GrabStrategy_LatestImageOnly, pypylon automatically keeps only
        the most recent frame — so this is equivalent to capture().

        Args:
            timeout_ms: Max wait time in ms if buffer is empty.

        Returns:
            BGR image as numpy array (uint8) — the most recent frame.

        Raises:
            TimeoutError: If no frame arrives within timeout.
            RuntimeError: If camera is not connected.
        """
        return self.capture(timeout_ms)

    def flush_buffers(self):
        """Drain and discard any stale finished buffers.

        With GrabStrategy_LatestImageOnly, stale frames are auto-discarded.
        This method drains anything remaining in the output queue.
        """
        if not self.connected or self._camera is None:
            return
        flushed = 0
        while True:
            try:
                grab_result = self._camera.RetrieveResult(0, pylon.TimeoutHandling_Return)
                if grab_result and grab_result.GrabSucceeded():
                    grab_result.Release()
                    flushed += 1
                else:
                    if grab_result:
                        grab_result.Release()
                    break
            except Exception:
                break
        if flushed:
            logger.info(f"Camera '{self.name}': flushed {flushed} stale buffer(s)")

    # ── Observability ────────────────────────────────────────────────

    def _track_grab_result(self, grab_result):
        """Extract per-frame observability from grab result metadata.

        Tracks:
        - block_id gaps: camera assigned frame N but we received N+2 → 1 transport drop
        - skipped images: frames pypylon discarded (LatestImageOnly strategy)
        - delivered count: total frames successfully delivered to application
        - timestamp: camera-side exposure timestamp (ticks)
        """
        self._frames_delivered += 1

        try:
            block_id = grab_result.GetBlockID()
            if self._last_block_id > 0 and block_id > self._last_block_id + 1:
                gap = block_id - self._last_block_id - 1
                self._block_id_gaps += gap
                logger.warning(
                    f"Camera '{self.name}': BlockID gap — expected {self._last_block_id + 1}, "
                    f"got {block_id} ({gap} frame(s) lost in transport)"
                )
            self._last_block_id = block_id
        except Exception:
            pass

        try:
            skipped = grab_result.GetNumberOfSkippedImages()
            if skipped > 0:
                self._frames_skipped += skipped
                logger.debug(
                    f"Camera '{self.name}': {skipped} frame(s) skipped (LatestImageOnly)"
                )
        except Exception:
            pass

        try:
            self._last_timestamp = grab_result.GetTimeStamp()
        except Exception:
            pass

    def health_check(self) -> bool:
        """Check if the camera is still reachable.

        Returns:
            True if connected and device is open, False otherwise.
        """
        if not self.connected:
            return False
        try:
            return self._camera.IsOpen()
        except Exception:
            return False

    def get_temperature(self) -> float:
        """Read camera sensor temperature in °C. Returns -1.0 if unavailable.

        ace classic: TemperatureAbs node
        ace 2: DeviceTemperature node (with DeviceTemperatureSelector)
        """
        if not self.connected:
            return -1.0
        try:
            if self._is_ace2:
                if genicam.IsAvailable(self._camera.DeviceTemperatureSelector):
                    self._camera.DeviceTemperatureSelector.Value = "Sensor"
                return self._camera.DeviceTemperature.Value
            else:
                return self._camera.TemperatureAbs.Value
        except Exception:
            return -1.0

    def get_line_status(self) -> bool | None:
        """Read current state of trigger input line (Line1).

        Returns:
            True if line is high, False if low, None if unavailable.
        """
        if not self.connected:
            return None
        try:
            self._camera.LineSelector.Value = "Line1"
            return self._camera.LineStatus.Value
        except Exception:
            return None

    def get_trigger_count(self) -> int:
        """Read hardware frame counter (Counter1 — FrameStart events).

        This counts frames the camera actually produced. Compare against
        frames_delivered to detect transport-layer drops.

        Returns -1 if counter not available.
        """
        if not self.connected or not self._counter_available:
            return -1
        try:
            self._camera.CounterSelector.Value = "Counter1"
            return self._camera.CounterValue.Value
        except Exception:
            return -1

    def get_line_trigger_count(self) -> int:
        """Read raw trigger line counter (Counter2 — Line1 rising edges).

        This counts every rising edge on Line1, including edges
        suppressed by debounce. Compare against Counter1 to see how
        many spurious triggers the debouncer filtered.

        Returns -1 if counter not available.
        """
        if not self.connected or not self._line_counter_available:
            return -1
        try:
            self._camera.CounterSelector.Value = "Counter2"
            return self._camera.CounterValue.Value
        except Exception:
            return -1

    def reset_trigger_count(self):
        """Reset both hardware counters and software counters to zero."""
        if not self.connected:
            return
        if self._counter_available:
            try:
                self._camera.CounterSelector.Value = "Counter1"
                self._camera.CounterReset.Execute()
            except Exception:
                pass
        if self._line_counter_available:
            try:
                self._camera.CounterSelector.Value = "Counter2"
                self._camera.CounterReset.Execute()
            except Exception:
                pass
        self._frames_delivered = 0
        self._frames_skipped = 0
        self._last_block_id = 0
        self._block_id_gaps = 0

    def get_stream_statistics(self) -> dict:
        """Full observability snapshot — camera counters + transport stats + software stats.

        Returns a dict with:

        Camera-side (hardware counters):
            - frame_count: Counter1 — frames the camera produced (FrameStart)
            - line_trigger_count: Counter2 — raw Line1 edges (before debounce)
            - debounced: line_trigger_count - frame_count (triggers filtered by debounce)

        Transport-layer (stream grabber):
            - missed: frames camera sent but pylon never received (network drops)
            - failed: frames received but incomplete/corrupt
            - buffer_underruns: frames lost because no buffer was available
            - resend_requests: GigE packet resend requests (reliability indicator)
            - resend_packets: actual packets resent

        Application-side (software counters):
            - delivered: frames successfully converted to BGR and returned
            - skipped: frames pypylon discarded (LatestImageOnly strategy)
            - block_id_gaps: gaps in camera BlockID sequence (transport drops)
            - temperature_c: camera sensor temperature

        Leakage analysis:
            frame_count > delivered + skipped → frames lost in transport
            line_trigger_count > frame_count → debouncer filtered spurious triggers
            block_id_gaps > 0 → frames dropped between camera and application
        """
        frame_count = self.get_trigger_count()
        line_count = self.get_line_trigger_count()

        stats = {
            "camera": self.name,
            # Camera-side
            "frame_count": frame_count,
            "line_trigger_count": line_count,
            "debounced": (line_count - frame_count) if line_count >= 0 and frame_count >= 0 else -1,
            # Transport-layer
            "missed": -1,
            "failed": -1,
            "buffer_underruns": -1,
            "resend_requests": -1,
            "resend_packets": -1,
            # Application-side
            "delivered": self._frames_delivered,
            "skipped": self._frames_skipped,
            "block_id_gaps": self._block_id_gaps,
            "temperature_c": self.get_temperature(),
        }

        if not self.connected or self._camera is None:
            return stats

        try:
            sg = self._camera.GetStreamGrabberNodeMap()
            transport_nodes = {
                "missed": "Statistic_Total_Missed_Frame_Count",
                "failed": "Statistic_Failed_Frame_Count",
                "buffer_underruns": "Statistic_Buffer_Underrun_Count",
                "resend_requests": "Statistic_Resend_Request_Count",
                "resend_packets": "Statistic_Resend_Packet_Count",
            }
            for key, node_name in transport_nodes.items():
                try:
                    stats[key] = sg[node_name].Value
                except Exception:
                    continue
        except Exception:
            pass

        return stats

    def log_stream_statistics(self):
        """Log full observability snapshot at INFO level.

        Warns on any leakage: debounced triggers, transport drops,
        buffer underruns, or block ID gaps.
        """
        stats = self.get_stream_statistics()
        problems = []

        # Camera → transport leakage
        if stats["missed"] > 0:
            problems.append(f"missed={stats['missed']}")
        if stats["failed"] > 0:
            problems.append(f"failed={stats['failed']}")
        if stats["buffer_underruns"] > 0:
            problems.append(f"underrun={stats['buffer_underruns']}")
        if stats["block_id_gaps"] > 0:
            problems.append(f"block_id_gaps={stats['block_id_gaps']}")

        # Trigger leakage
        fc = stats["frame_count"]
        d = stats["delivered"]
        sk = stats["skipped"]
        if fc >= 0 and d >= 0:
            transport_loss = fc - (d + sk)
            if transport_loss > 0:
                problems.append(f"transport_loss={transport_loss}")

        if stats["debounced"] > 0:
            problems.append(f"debounced={stats['debounced']}")

        # Temperature warning
        temp = stats["temperature_c"]
        if temp > 70.0:
            problems.append(f"TEMP={temp:.1f}°C")

        status = " ISSUES: " + ", ".join(problems) if problems else " OK"
        logger.info(
            f"Camera '{self.name}' stats: "
            f"triggers(line={stats['line_trigger_count']} frame={stats['frame_count']} "
            f"debounced={stats['debounced']}) "
            f"delivery(delivered={d} skipped={sk} gaps={stats['block_id_gaps']}) "
            f"transport(missed={stats['missed']} failed={stats['failed']} "
            f"underrun={stats['buffer_underruns']} resend={stats['resend_requests']}) "
            f"temp={temp:.1f}°C{status}"
        )

    def get_incomplete_frame_count(self) -> int:
        """Get the count of incomplete/failed frames. Returns -1 if unavailable."""
        stats = self.get_stream_statistics()
        return stats.get("failed", -1)

    # ── Internal ──────────────────────────────────────────────────────

    def __repr__(self):
        status = "connected" if self.connected else "disconnected"
        ident = f"ip='{self.ip}'" if self.ip else f"serial='{self.serial}'"
        return f"Camera(name='{self.name}', {ident}, exposure={self.exposure}, {status})"
