# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AMDSMI-based GPU telemetry collector.

Collects GPU metrics from AMD ROCm GPUs (Instinct MI300X, MI355X, etc.) using
the amdsmi Python library shipped with ROCm. Mirrors the behavioral contract of
the PyNVML collector so that downstream accumulation, export, and dashboard
code remains hardware-agnostic.
"""

import asyncio
import contextlib
import threading
import time
from dataclasses import dataclass
from typing import Any

import amdsmi

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
class ScalingFactors:
    """Unit conversion scaling factors for AMDSMI metrics.

    AMDSMI returns power in W (no scaling) and energy in counter ticks where
    one tick equals counter_resolution µJ. Memory bytes are converted to GB.
    """

    energy_uj_to_mj = 1e-12  # ticks * counter_resolution(µJ) -> MJ
    bytes_to_gb = 1e-9


@dataclass(slots=True)
class GpuDeviceState:
    """Per-GPU state for AMDSMI telemetry collection.

    Args:
        handle: AMDSMI processor handle (opaque pointer)
        metadata: GPU metadata
        throttle_accum_us: Accumulated throttle duration in microseconds, computed
            client-side because AMDSMI only exposes a boolean throttle_status
            rather than a duration counter like NVML's violationTime.
        last_collect_ns: Timestamp of last successful collection for throttle
            duration computation.
    """

    handle: Any
    metadata: GpuMetadata
    throttle_accum_us: float = 0.0
    last_collect_ns: int = 0


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

        self._gpus: list[GpuDeviceState] = []
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
        if self._initialized:
            return len(self._gpus) > 0
        try:
            return await asyncio.to_thread(self._probe_devices)
        except Exception:
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
            RuntimeError: If amdsmi cannot initialize or no GPUs are present.
        """
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

        self.info(f"AMDSMI initialized with {len(self._gpus)} GPU(s)")

    def _build_gpu_state(self, index: int, handle: Any) -> GpuDeviceState | None:
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

        return GpuDeviceState(
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
            except Exception as e:
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
        except Exception as e:
            if self._error_callback:
                try:
                    await self._error_callback(ErrorDetails.from_exception(e), self.id)
                except Exception as cb_err:
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
                metrics = self._snapshot_gpu(gpu, now_ns, ExcType)
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
        self, gpu: GpuDeviceState, now_ns: int, ExcType: type[Exception]
    ) -> TelemetryMetrics:
        """Capture all supported metrics for one GPU into a TelemetryMetrics."""
        handle = gpu.handle
        td = TelemetryMetrics()

        # Power (W). current_socket_power is already in W; no scaling.
        with contextlib.suppress(ExcType):
            power = amdsmi.amdsmi_get_power_info(handle)
            value = _numeric(power.get("current_socket_power"))
            if value is None:
                value = _numeric(power.get("average_socket_power"))
            if value is not None:
                td.gpu_power_usage = value

        # Energy: accumulator(ticks) * counter_resolution(µJ) -> MJ
        with contextlib.suppress(ExcType):
            energy = amdsmi.amdsmi_get_energy_count(handle)
            acc = _numeric(energy.get("energy_accumulator"))
            res = _numeric(energy.get("counter_resolution"))
            if acc is not None and res is not None:
                td.energy_consumption = acc * res * ScalingFactors.energy_uj_to_mj

        # Activity (gfx/umc/mm). mm_activity is N/A on Instinct GPUs.
        with contextlib.suppress(ExcType):
            activity = amdsmi.amdsmi_get_gpu_activity(handle)
            gfx = _numeric(activity.get("gfx_activity"))
            umc = _numeric(activity.get("umc_activity"))
            mm = _numeric(activity.get("mm_activity"))
            if gfx is not None:
                td.gpu_utilization = gfx
                td.sm_utilization = gfx  # AMD has no separate SM-level metric
            if umc is not None:
                td.mem_utilization = umc
            if mm is not None:
                td.encoder_utilization = mm
                td.decoder_utilization = mm

        # VRAM used (bytes -> GB)
        with contextlib.suppress(ExcType):
            vram_used = _numeric(
                amdsmi.amdsmi_get_gpu_memory_usage(handle, amdsmi.AmdSmiMemoryType.VRAM)
            )
            if vram_used is not None:
                td.gpu_memory_used = vram_used * ScalingFactors.bytes_to_gb

        # Temperature: prefer JUNCTION (works on MI300X/MI355X), fall back to
        # HOTSPOT. EDGE is unsupported on Instinct GPUs.
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
            if value is not None:
                td.gpu_temperature = value
                break

        # ECC: uncorrectable error count maps closest to xid_errors semantics.
        with contextlib.suppress(ExcType):
            ecc = amdsmi.amdsmi_get_gpu_total_ecc_count(handle)
            uc = _numeric(ecc.get("uncorrectable_count"))
            if uc is not None:
                td.xid_errors = uc

        # Throttle duration: AMDSMI exposes only a boolean throttle_status, so
        # accumulate microseconds spent throttled between scrapes client-side.
        # Both throttle_status and indep_throttle_status often return the literal
        # 'N/A' string when unsupported; coerce to int before truthiness check
        # so 'N/A' does not register as "throttled".
        with contextlib.suppress(ExcType):
            m = amdsmi.amdsmi_get_gpu_metrics_info(handle)
            ts = _numeric(m.get("throttle_status"))
            its = _numeric(m.get("indep_throttle_status"))
            throttled = bool((ts or 0) or (its or 0))
            if throttled and gpu.last_collect_ns:
                gpu.throttle_accum_us += (now_ns - gpu.last_collect_ns) / 1_000.0
            td.power_violation = gpu.throttle_accum_us

        gpu.last_collect_ns = now_ns
        return td
