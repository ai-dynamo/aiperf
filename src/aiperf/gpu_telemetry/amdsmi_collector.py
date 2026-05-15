# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AMDSMI-based GPU telemetry collector.

Collects GPU metrics from AMD ROCm GPUs (Instinct MI300X, MI355X, etc.) using
the amdsmi Python library shipped with ROCm. Emits AMD signals under
vendor-namespaced ``amd_*`` fields on TelemetryMetrics rather than aliasing
them onto NVML-shaped names, since the underlying signals do not always
measure the same physical quantity (e.g. amdsmi ``gfx_activity`` and NVML
``sm_utilization`` sample at different scopes).
"""

import asyncio
import contextlib
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import amdsmi
else:
    # ``amdsmi`` ships with ROCm and is not pip-installable, so it may be
    # absent on developer machines or CI runners without GPUs. The wheel can
    # also be installed without a working native libamd_smi.so (version
    # mismatch, missing amdgpu driver), in which case the import raises
    # OSError rather than ImportError. Defer either failure to collector
    # instantiation so module discovery (plugin enumeration,
    # tests/unit/test_imports.py) does not fail.
    try:
        import amdsmi
    except (ImportError, OSError):
        amdsmi = None  # type: ignore[assignment]

from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task, on_init, on_stop
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import (
    ErrorDetails,
    GpuMetadata,
    TelemetryMetrics,
    TelemetryRecord,
)
from aiperf.gpu_telemetry.constants import AMDSMI_SOURCE_IDENTIFIER
from aiperf.gpu_telemetry.protocols import TErrorCallback, TRecordCallback

__all__ = ["AMDSMITelemetryCollector"]


@dataclass(frozen=True)
class _AMDScalingFactors:
    """Unit conversion scaling factors for AMDSMI metrics.

    AMDSMI returns power in W (no scaling) and energy in counter ticks where
    one tick equals counter_resolution µJ. Memory bytes are converted to GB.
    """

    energy_uj_to_mj = 1e-12  # ticks * counter_resolution(µJ) -> MJ
    bytes_to_gb = 1e-9


@dataclass(slots=True)
class _AMDGpuDeviceState:
    """Per-GPU state for AMDSMI telemetry collection.

    Args:
        handle: AMDSMI processor handle (opaque pointer)
        metadata: GPU metadata
    """

    handle: Any
    metadata: GpuMetadata


def _numeric(value: Any) -> float | None:
    """Coerce an AMDSMI return value to float, treating sentinels as None.

    AMDSMI commonly returns the literal string ``'N/A'`` for unsupported sensors
    (e.g. ``average_socket_power`` on MI300X/MI355X, ``mm_activity`` on Instinct
    parts) instead of raising an exception. Any non-numeric value, including
    ``'N/A'``, becomes ``None`` so downstream pydantic validation is not given
    a string where it expects a float.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _amdsmi_returns_celsius() -> bool:
    """Whether the installed amdsmi binding returns temperatures in Celsius.

    The AMDSMI C API documents temperatures as millidegrees Celsius, but the
    Python binding started normalizing to Celsius in the ``26.x`` series
    (ROCm ~6.3+). Older bindings return millidegrees. Gate on the major
    version of ``amdsmi.__version__``.

    If the version is missing or unparsable, assume modern (>= 26): every
    currently-deployed binding we have validated against (Hotaisle MI300X
    amdsmi 26.0.2, AAC1 MI355X amdsmi 26.2.1) is in that range, and over-
    dividing a Celsius value would produce obviously-wrong sub-degree
    readings that would be caught immediately.
    """
    if amdsmi is None:
        return True
    version = getattr(amdsmi, "__version__", "")
    try:
        return int(version.split(".", 1)[0]) >= 26
    except (ValueError, AttributeError, IndexError):
        return True


def _is_throttled(value: Any) -> bool:
    """Decide whether an AMDSMI throttle_status value indicates active throttling.

    AMDSMI returns this field as ``bool`` on some platforms, ``int`` (bitfield)
    on others, and the literal string ``'N/A'`` when the sensor is unsupported.
    Treat any truthy bool or any non-zero int as throttled; treat strings and
    None as not throttled. ``_numeric`` cannot be reused here because it
    intentionally maps ``bool`` to ``None``.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return False


class AMDSMITelemetryCollector(AIPerfLifecycleMixin):
    """Collects GPU telemetry from AMD ROCm GPUs via the amdsmi library.

    Direct collector that uses AMD's amdsmi Python bindings to gather GPU
    metrics locally. Functionally equivalent to the PyNVML collector for
    purposes of the GPUTelemetryManager, but targets AMD Instinct GPUs
    (gfx942, gfx950, etc.) running on ROCm.

    Features:
        - Direct AMDSMI access (no HTTP exporter required)
        - Automatic GPU discovery and enumeration
        - Same TelemetryRecord output format as DCGM/PyNVML collectors
        - Callback-based record delivery
        - Tolerant of partial sensor support: fields that return 'N/A' or
          raise AmdSmiLibraryException are silently dropped from the record

    Requirements:
        - amdsmi Python package (ships with ROCm at
          /opt/rocm/share/amd_smi/amdsmi-*.whl)
        - ROCm driver loaded with at least one supported AMD GPU

    Args:
        collection_interval: Interval in seconds between metric collections
        record_callback: Async callback invoked with collected records.
            Signature: async (records: list[TelemetryRecord], collector_id: str) -> None
        error_callback: Async callback invoked on collection errors.
            Signature: async (error: ErrorDetails, collector_id: str) -> None
        collector_id: Unique identifier for this collector instance
    """

    @classmethod
    def validate_environment(cls) -> None:
        """Verify that amdsmi bindings are importable before config succeeds."""
        if amdsmi is None:
            raise RuntimeError(
                "amdsmi package not installed. The amdsmi Python bindings ship "
                "with ROCm; install from /opt/rocm/share/amd_smi/amdsmi-*.whl "
                "or your distro's amd-smi-lib package."
            )

    def __init__(
        self,
        collection_interval: float = Environment.GPU.COLLECTION_INTERVAL,
        record_callback: TRecordCallback | None = None,
        error_callback: TErrorCallback | None = None,
        collector_id: str = "amdsmi_collector",
    ) -> None:
        super().__init__(id=collector_id)
        self._collection_interval = collection_interval
        self._record_callback = record_callback
        self._error_callback = error_callback

        self._gpus: list[_AMDGpuDeviceState] = []
        self._initialized = False
        self._lock = threading.Lock()

    @property
    def endpoint_url(self) -> str:
        """Source identifier for this collector ('amdsmi://localhost')."""
        return AMDSMI_SOURCE_IDENTIFIER

    @property
    def collection_interval(self) -> float:
        """Collection interval in seconds."""
        return self._collection_interval

    async def is_url_reachable(self) -> bool:
        """Check if AMDSMI is available and at least one GPU is visible.

        Returns:
            True if amdsmi can initialize and enumerates >=1 processor handle.
        """
        if amdsmi is None:
            return False
        if self._initialized:
            return len(self._gpus) > 0
        try:
            return await asyncio.to_thread(self._probe_devices)
        except Exception:  # noqa: BLE001 - reachability probe must never raise
            return False

    def _probe_devices(self) -> bool:
        """Synchronous probe: init amdsmi, count GPUs, shut down."""
        amdsmi.amdsmi_init()
        try:
            return len(amdsmi.amdsmi_get_processor_handles()) > 0
        finally:
            with contextlib.suppress(amdsmi.AmdSmiException):
                amdsmi.amdsmi_shut_down()

    @on_init
    async def _initialize_amdsmi(self) -> None:
        """Initialize amdsmi and enumerate available GPUs.

        Raises:
            RuntimeError: If the amdsmi Python bindings are not installed,
                amdsmi cannot initialize, or no GPUs are present.
        """
        if amdsmi is None:
            raise RuntimeError(
                "amdsmi Python bindings not installed. The amdsmi package ships "
                "with ROCm; install from /opt/rocm/share/amd_smi/amdsmi-*.whl "
                "or your distro's amd-smi-lib package."
            )
        try:
            amdsmi.amdsmi_init()
        except amdsmi.AmdSmiException as e:
            raise RuntimeError(f"Failed to initialize amdsmi: {e}") from e

        self._initialized = True

        try:
            handles = amdsmi.amdsmi_get_processor_handles()
        except amdsmi.AmdSmiException as e:
            self._shutdown_sync()
            raise RuntimeError(f"Failed to enumerate AMD GPUs: {e}") from e

        self._gpus = [
            gpu
            for gpu in (self._build_gpu_state(idx, h) for idx, h in enumerate(handles))
            if gpu is not None
        ]

        if not self._gpus:
            self._shutdown_sync()
            raise RuntimeError("No AMD GPUs detected via amdsmi")

        self.info(f"AMDSMI initialized with {len(self._gpus)} GPU(s)")

    def _build_gpu_state(self, index: int, handle: Any) -> _AMDGpuDeviceState | None:
        """Build per-GPU state with static metadata."""
        try:
            uuid = amdsmi.amdsmi_get_gpu_device_uuid(handle)
        except amdsmi.AmdSmiException:
            uuid = f"GPU-unknown-{index}"

        try:
            board = amdsmi.amdsmi_get_gpu_board_info(handle)
            name = board.get("product_name") or "Unknown AMD GPU"
        except amdsmi.AmdSmiException:
            name = "Unknown AMD GPU"

        try:
            bdf = amdsmi.amdsmi_get_gpu_device_bdf(handle)
            pci_bus_id = bdf if isinstance(bdf, str) else None
        except amdsmi.AmdSmiException:
            pci_bus_id = None

        return _AMDGpuDeviceState(
            handle=handle,
            metadata=GpuMetadata(
                gpu_index=index,
                gpu_uuid=uuid,
                gpu_model_name=name,
                pci_bus_id=pci_bus_id,
                device=f"amd{index}",
                hostname="localhost",
            ),
        )

    def _shutdown_sync(self) -> None:
        """Thread-safe synchronous shutdown of amdsmi state."""
        with self._lock:
            if not self._initialized:
                return
            try:
                amdsmi.amdsmi_shut_down()
            except Exception as e:  # noqa: BLE001 - shutdown is best-effort
                self.warning(f"Error during amdsmi shutdown: {e!r}")
            finally:
                self._initialized = False
                self._gpus = []

    @on_stop
    async def _shutdown_amdsmi(self) -> None:
        """Shut down amdsmi (thread-safe; waits for in-flight collection)."""
        await asyncio.to_thread(self._shutdown_sync)
        self.debug("AMDSMI shutdown complete")

    @background_task(immediate=True, interval=lambda self: self.collection_interval)
    async def _collect_metrics_loop(self) -> None:
        """Periodic collection task that runs while the collector is RUNNING."""
        await self._collect_and_process_metrics()

    async def collect_and_process_metrics(self) -> None:
        """Public alias for one-shot scrape.

        ``GPUTelemetryManager`` calls this name during baseline and final-state
        capture (``manager.py`` :func:`_handle_profile_complete_command`).
        """
        await self._collect_and_process_metrics()

    async def _collect_and_process_metrics(self) -> None:
        """Collect metrics and dispatch via record/error callbacks."""
        try:
            records = await asyncio.to_thread(self._collect_gpu_metrics)
            if records and self._record_callback:
                await self._record_callback(records, self.id)
        except Exception as e:  # noqa: BLE001 - fault-tolerant telemetry
            if self._error_callback:
                try:
                    await self._error_callback(ErrorDetails.from_exception(e), self.id)
                except Exception as cb_err:  # noqa: BLE001 - callback failure must not propagate
                    self.error(f"Failed to send error via callback: {cb_err}")
            else:
                self.error(f"Metrics collection error: {e}")

    def _collect_gpu_metrics(self) -> list[TelemetryRecord]:
        """Collect one record per GPU using AMDSMI APIs.

        Thread-safe against concurrent shutdown via ``_lock``.
        """
        with self._lock:
            if not self._initialized or not self._gpus:
                return []

            now_ns = time.time_ns()
            ExcType = amdsmi.AmdSmiException
            records: list[TelemetryRecord] = []

            for gpu in self._gpus:
                metrics = self._snapshot_gpu(gpu, ExcType)
                if metrics.model_fields_set:
                    records.append(
                        TelemetryRecord(
                            timestamp_ns=now_ns,
                            dcgm_url=AMDSMI_SOURCE_IDENTIFIER,
                            **gpu.metadata.model_dump(),
                            telemetry_data=metrics,
                        )
                    )

            return records

    def _snapshot_gpu(
        self, gpu: _AMDGpuDeviceState, ExcType: type[Exception]
    ) -> TelemetryMetrics:
        """Capture all supported metrics for one GPU into a TelemetryMetrics."""
        td = TelemetryMetrics()
        handle = gpu.handle
        self._collect_power(handle, td, ExcType)
        self._collect_energy(handle, td, ExcType)
        self._collect_activity(handle, td, ExcType)
        self._collect_memory(handle, td, ExcType)
        self._collect_temperature(handle, td, ExcType)
        self._collect_ecc(handle, td, ExcType)
        self._collect_throttle(handle, td, ExcType)
        return td

    @staticmethod
    def _collect_power(
        handle: Any, td: TelemetryMetrics, ExcType: type[Exception]
    ) -> None:
        """Power in W. ``current_socket_power`` is already in W; no scaling."""
        with contextlib.suppress(ExcType):
            power = amdsmi.amdsmi_get_power_info(handle)
            value = _numeric(power.get("current_socket_power"))
            if value is None:
                value = _numeric(power.get("average_socket_power"))
            if value is not None:
                td.amd_power = value

    @staticmethod
    def _collect_energy(
        handle: Any, td: TelemetryMetrics, ExcType: type[Exception]
    ) -> None:
        """Energy: ``accumulator(ticks) * counter_resolution(µJ)`` -> MJ.

        AMDSMI renamed the field from ``power`` to ``energy_accumulator``
        somewhere around the 6.2 timeframe. Fall back to the older name so
        we keep working on ROCm 6.x.
        """
        with contextlib.suppress(ExcType):
            energy = amdsmi.amdsmi_get_energy_count(handle)
            acc = _numeric(energy.get("energy_accumulator"))
            if acc is None:
                acc = _numeric(energy.get("power"))  # ROCm 6.x naming
            res = _numeric(energy.get("counter_resolution"))
            if acc is not None and res is not None:
                td.amd_energy_consumption = (
                    acc * res * _AMDScalingFactors.energy_uj_to_mj
                )

    @staticmethod
    def _collect_activity(
        handle: Any, td: TelemetryMetrics, ExcType: type[Exception]
    ) -> None:
        """gfx/umc/mm activity. ``mm_activity`` is N/A on Instinct GPUs."""
        with contextlib.suppress(ExcType):
            activity = amdsmi.amdsmi_get_gpu_activity(handle)
            gfx = _numeric(activity.get("gfx_activity"))
            umc = _numeric(activity.get("umc_activity"))
            mm = _numeric(activity.get("mm_activity"))
            if gfx is not None:
                td.amd_gfx_activity = gfx
            if umc is not None:
                td.amd_umc_activity = umc
            if mm is not None:
                td.amd_mm_activity = mm

    @staticmethod
    def _collect_memory(
        handle: Any, td: TelemetryMetrics, ExcType: type[Exception]
    ) -> None:
        """VRAM used (bytes -> GB)."""
        with contextlib.suppress(ExcType):
            vram_used = _numeric(
                amdsmi.amdsmi_get_gpu_memory_usage(handle, amdsmi.AmdSmiMemoryType.VRAM)
            )
            if vram_used is not None:
                td.amd_memory_used = vram_used * _AMDScalingFactors.bytes_to_gb

    @staticmethod
    def _collect_temperature(
        handle: Any, td: TelemetryMetrics, ExcType: type[Exception]
    ) -> None:
        """Temperature: prefer JUNCTION, fall back to HOTSPOT (EDGE unsupported on Instinct).

        Unit conversion: divide by 1000 if ``amdsmi.__version__ < 26`` OR the
        raw value is implausibly high (> 200 °C). The version gate covers
        bindings whose Python API returns millidegrees; the >200 sanity
        check hedges against newer bindings whose docs still describe
        millideg semantics (amdsmi 26.2.2 docs notwithstanding our
        empirical Celsius observations on 26.0.2/26.2.1).
        """
        for kind in ("JUNCTION", "HOTSPOT"):
            try:
                temp = amdsmi.amdsmi_get_temp_metric(
                    handle,
                    getattr(amdsmi.AmdSmiTemperatureType, kind),
                    amdsmi.AmdSmiTemperatureMetric.CURRENT,
                )
            except ExcType:
                continue
            value = _numeric(temp)
            if value is None:
                continue
            if not _amdsmi_returns_celsius() or value > 200:
                value = value / 1000.0
            td.amd_temperature = value
            return

    @staticmethod
    def _collect_ecc(
        handle: Any, td: TelemetryMetrics, ExcType: type[Exception]
    ) -> None:
        """ECC: cumulative uncorrectable error count from ``uncorrectable_count``."""
        with contextlib.suppress(ExcType):
            ecc = amdsmi.amdsmi_get_gpu_total_ecc_count(handle)
            uc = _numeric(ecc.get("uncorrectable_count"))
            if uc is not None:
                td.amd_ecc_uncorrectable = uc

    @staticmethod
    def _collect_throttle(
        handle: Any, td: TelemetryMetrics, ExcType: type[Exception]
    ) -> None:
        """Throttle status snapshot from ``throttle_status``.

        AMDSMI exposes a boolean (or bitfield) throttle_status, not a duration
        counter. We surface the raw signal as a 0.0/1.0 gauge rather than
        synthesizing a duration client-side. ``throttle_status`` may be
        returned as ``bool``, ``int``, or the literal ``'N/A'`` string depending
        on AMDSMI version and platform support.

        Leave ``amd_throttle_status`` unset (rather than 0.0) when neither
        signal carries a real numeric/boolean value, so an unsupported sensor
        is not silently mislabeled as "not throttled".
        """
        with contextlib.suppress(ExcType):
            m = amdsmi.amdsmi_get_gpu_metrics_info(handle)
            primary = m.get("throttle_status")
            indep = m.get("indep_throttle_status")
            if not any(isinstance(v, (bool, int, float)) for v in (primary, indep)):
                return
            throttled = _is_throttled(primary) or _is_throttled(indep)
            td.amd_throttle_status = 1.0 if throttled else 0.0
