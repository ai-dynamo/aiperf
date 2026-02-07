# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyNVML-based GPU telemetry collector.

Collects GPU metrics directly using the pynvml Python library, providing an
alternative to DCGM HTTP endpoints for local GPU monitoring.
"""

import asyncio
import contextlib
import threading
import time
from dataclasses import dataclass

import pynvml

from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task, on_init, on_stop
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import (
    ErrorDetails,
    GpuMetadata,
    TelemetryMetrics,
    TelemetryRecord,
)
from aiperf.gpu_telemetry.constants import (
    PYNVML_SOURCE_IDENTIFIER,
)
from aiperf.gpu_telemetry.protocols import TErrorCallback, TRecordCallback

__all__ = ["PyNVMLTelemetryCollector"]


@dataclass(frozen=True)
class ScalingFactors:
    """Unit conversion scaling factors for NVML metrics."""

    gpu_power_usage = 1e-3  # mW -> W
    energy_consumption = 1e-9  # mJ -> MJ
    gpu_memory_used = 1e-9  # bytes -> GB
    power_violation = 1e-3  # ns -> µs


@dataclass(slots=True)
class GpuDeviceState:
    """Per-GPU state for NVML telemetry collection.

    Args:
        handle: NVML device handle
        metadata: GPU metadata
        gpm_samples: GPM samples (prev, curr) if GPM supported, else None
    """

    handle: object
    metadata: GpuMetadata
    gpm_samples: tuple[object, object] | None = None


class PyNVMLTelemetryCollector(AIPerfLifecycleMixin):
    """Collects GPU telemetry metrics using the pynvml Python library.

    Direct collector that uses NVIDIA's pynvml library to gather GPU metrics
    locally without requiring a DCGM HTTP endpoint. Useful for environments
    where DCGM is not deployed or for simple local GPU monitoring.

    Features:
        - Direct NVML API access via pynvml
        - Automatic GPU discovery and enumeration
        - Same TelemetryRecord output format as DCGM collector
        - Callback-based record delivery

    Requirements:
        - pynvml package installed: `pip install nvidia-ml-py`
        - NVIDIA driver installed with NVML support

    Args:
        collection_interval: Interval in seconds between metric collections (default: from Environment)
        record_callback: Optional async callback to receive collected records.
            Signature: async (records: list[TelemetryRecord], collector_id: str) -> None
        error_callback: Optional async callback to receive collection errors.
            Signature: async (error: ErrorDetails, collector_id: str) -> None
        collector_id: Unique identifier for this collector instance

    Raises:
        RuntimeError: If pynvml package is not installed
    """

    def __init__(
        self,
        collection_interval: float = Environment.GPU.COLLECTION_INTERVAL,
        record_callback: TRecordCallback | None = None,
        error_callback: TErrorCallback | None = None,
        collector_id: str = "pynvml_collector",
    ) -> None:
        super().__init__(id=collector_id)
        self._collection_interval = collection_interval
        self._record_callback = record_callback
        self._error_callback = error_callback

        # Per-GPU state (populated on init)
        self._gpus: list[GpuDeviceState] = []

        # NVML initialization state and thread safety
        self._nvml_initialized = False
        self._nvml_lock = threading.Lock()

    @property
    def endpoint_url(self) -> str:
        """Get the source identifier for this collector.

        Returns:
            'pynvml://localhost' to identify records from pynvml collection.
        """
        return PYNVML_SOURCE_IDENTIFIER

    @property
    def collection_interval(self) -> float:
        """Get the collection interval in seconds."""
        return self._collection_interval

    async def is_url_reachable(self) -> bool:
        """Check if NVML is available and can be initialized.

        Tests NVML availability by attempting initialization if not already done.
        This allows pre-flight checks before starting collection.

        Returns:
            True if NVML is available and can access at least one GPU.
        """
        # If already initialized, just check if we have GPUs
        if self._nvml_initialized:
            return len(self._gpus) > 0

        try:
            return await asyncio.to_thread(self._probe_nvml_devices)
        except Exception:
            return False

    def _probe_nvml_devices(self) -> bool:
        """Probe NVML to check if GPUs are available.

        Synchronous helper that performs blocking NVML calls to check availability.
        Called via asyncio.to_thread to avoid blocking the event loop.

        Returns:
            True if NVML can be initialized and at least one GPU is available.
        """
        pynvml.nvmlInit()
        try:
            count = pynvml.nvmlDeviceGetCount()
            return count > 0
        finally:
            pynvml.nvmlShutdown()

    @on_init
    async def _initialize_nvml(self) -> None:
        """Initialize NVML and discover GPUs.

        Called automatically during initialization phase.
        Initializes the NVML library and enumerates available GPUs.

        Raises:
            RuntimeError: If NVML initialization or GPU discovery fails.
        """
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as e:
            raise RuntimeError(f"Failed to initialize NVML: {e}") from e

        self._nvml_initialized = True

        try:
            device_count = pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError as e:
            # Cleanup NVML if device enumeration fails
            self._shutdown_nvml_sync()
            raise RuntimeError(f"Failed to get GPU device count: {e}") from e

        self._gpus = []

        for i in range(device_count):
            gpu = self._create_gpu_for_device_index(i)
            if gpu:
                self._gpus.append(gpu)

        gpm_count = sum(1 for gpu in self._gpus if gpu.gpm_samples)
        self.info(
            f"PyNVML initialized with {len(self._gpus)} GPU(s) "
            f"({gpm_count} with GPM support)"
        )

    def _create_gpu_for_device_index(self, index: int) -> GpuDeviceState | None:
        """Initialize a GPU for telemetry collection."""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except pynvml.NVMLError as e:
            self.warning(f"Failed to get handle for GPU {index}: {e}")
            return None

        # Gather static metadata for this GPU
        try:
            uuid = pynvml.nvmlDeviceGetUUID(handle)
        except pynvml.NVMLError:
            uuid = f"GPU-unknown-{index}"

        try:
            name = pynvml.nvmlDeviceGetName(handle)
            # pynvml may return bytes in some versions
            if isinstance(name, bytes):
                name = name.decode("utf-8")
        except pynvml.NVMLError:
            name = "Unknown GPU"

        try:
            pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
            pci_bus_id = pci_info.busId
            if isinstance(pci_bus_id, bytes):
                pci_bus_id = pci_bus_id.decode("utf-8")
        except pynvml.NVMLError:
            pci_bus_id = None

        # Create GPU state with metadata
        # gpu_index in metadata reflects original NVML index for display
        gpu = GpuDeviceState(
            handle=handle,
            metadata=GpuMetadata(
                gpu_index=index,
                gpu_uuid=uuid,
                gpu_model_name=name,
                pci_bus_id=pci_bus_id,
                device=f"nvidia{index}",
                hostname="localhost",
            ),
        )

        # Check GPM support and allocate samples for efficient SM utilization
        self._init_gpm_for_device(gpu)
        return gpu

    def _init_gpm_for_device(self, gpu: GpuDeviceState) -> None:
        """Initialize GPM (GPU Performance Metrics) for efficient SM utilization."""
        try:
            if not pynvml.nvmlGpmQueryDeviceSupport(gpu.handle).isSupportedDevice:
                return
            sample1 = pynvml.nvmlGpmSampleAlloc()
            sample2 = pynvml.nvmlGpmSampleAlloc()
            # Take initial sample so delta computation works on first collection
            pynvml.nvmlGpmSampleGet(gpu.handle, sample1)
            gpu.gpm_samples = (sample1, sample2)
            self.debug(lambda: f"GPM enabled for GPU {gpu.metadata.gpu_index}")
        except pynvml.NVMLError:
            # GPM unavailable, will use process API fallback
            self.debug(lambda: f"GPM not supported for GPU {gpu.metadata.gpu_index}")

    def _free_gpm_samples(self) -> None:
        """Free all allocated GPM sample buffers."""
        for gpu in self._gpus:
            if gpu.gpm_samples:
                for sample in gpu.gpm_samples:
                    with contextlib.suppress(pynvml.NVMLError):
                        pynvml.nvmlGpmSampleFree(sample)
                gpu.gpm_samples = None

    def _get_sm_utilization_gpm(self, gpu: GpuDeviceState) -> float | None:
        """Get SM utilization using GPM API (device-level, more efficient)."""
        prev_sample, curr_sample = gpu.gpm_samples  # type: ignore[misc]
        try:
            pynvml.nvmlGpmSampleGet(gpu.handle, curr_sample)
            metrics_get = pynvml.c_nvmlGpmMetricsGet_t()
            metrics_get.version = pynvml.NVML_GPM_METRICS_GET_VERSION
            metrics_get.sample1 = prev_sample
            metrics_get.sample2 = curr_sample
            metrics_get.numMetrics = 1
            metrics_get.metrics[0].metricId = pynvml.NVML_GPM_METRIC_SM_UTIL
            pynvml.nvmlGpmMetricsGet(metrics_get)
            sm_util = metrics_get.metrics[0].value
        except pynvml.NVMLError:
            sm_util = None
        gpu.gpm_samples = (curr_sample, prev_sample)  # Swap for next iteration
        return sm_util

    def _shutdown_nvml_sync(self) -> None:
        """Synchronous NVML shutdown helper.

        Thread-safe shutdown that clears all state. Can be called from
        any context (init cleanup or stop phase).
        """
        with self._nvml_lock:
            if not self._nvml_initialized:
                return

            # Free GPM samples before NVML shutdown
            self._free_gpm_samples()

            try:
                pynvml.nvmlShutdown()
            except Exception as e:
                self.warning(f"Error during NVML shutdown: {e!r}")
            finally:
                # Always clear state regardless of shutdown success
                self._nvml_initialized = False
                self._gpus = []

    @on_stop
    async def _shutdown_nvml(self) -> None:
        """Shutdown NVML library.

        Called automatically during shutdown phase.
        Thread-safe - waits for any in-progress collection to complete.
        """
        await asyncio.to_thread(self._shutdown_nvml_sync)
        self.debug("PyNVML shutdown complete")

    @background_task(immediate=True, interval=lambda self: self.collection_interval)
    async def _collect_metrics_loop(self) -> None:
        """Background task for collecting metrics at regular intervals.

        Runs continuously during collector's RUNNING state, triggering a metrics
        collection every collection_interval seconds.
        """
        await self._collect_and_process_metrics()

    async def _collect_and_process_metrics(self) -> None:
        """Collect metrics from all GPUs and send via callback.

        Gathers current metrics from all discovered GPUs using NVML APIs,
        converts them to TelemetryRecord objects, and delivers via callback.
        Uses asyncio.to_thread() to avoid blocking the event loop with NVML calls.
        """
        try:
            records = await asyncio.to_thread(self._collect_gpu_metrics)
            if records and self._record_callback:
                await self._record_callback(records, self.id)
        except Exception as e:
            if self._error_callback:
                try:
                    await self._error_callback(ErrorDetails.from_exception(e), self.id)
                except Exception as callback_error:
                    self.error(f"Failed to send error via callback: {callback_error}")
            else:
                self.error(f"Metrics collection error: {e}")

    def _collect_gpu_metrics(self) -> list[TelemetryRecord]:
        """Collect metrics from all GPUs using NVML APIs.

        Thread-safe - acquires lock to prevent collection during shutdown.

        Returns:
            List of TelemetryRecord objects, one per GPU.
        """
        with self._nvml_lock:
            if not self._nvml_initialized or not self._gpus:
                return []

            current_timestamp = time.time_ns()
            records = []
            NVMLError = pynvml.NVMLError

            for gpu in self._gpus:
                handle = gpu.handle
                telemetry_data = TelemetryMetrics()

                # Power usage (milliwatts -> watts)
                with contextlib.suppress(NVMLError):
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    telemetry_data.gpu_power_usage = (
                        power_mw * ScalingFactors.gpu_power_usage
                    )

                # Total energy consumption (millijoules -> megajoules)
                with contextlib.suppress(NVMLError):
                    energy_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                    telemetry_data.energy_consumption = (
                        energy_mj * ScalingFactors.energy_consumption
                    )

                # GPU and memory utilization (percent)
                with contextlib.suppress(NVMLError):
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    telemetry_data.gpu_utilization = float(util.gpu)
                    telemetry_data.mem_utilization = float(util.memory)

                # Memory used (bytes -> gigabytes)
                with contextlib.suppress(NVMLError):
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    telemetry_data.gpu_memory_used = (
                        mem_info.used * ScalingFactors.gpu_memory_used
                    )

                # Temperature (Celsius)
                with contextlib.suppress(NVMLError):
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                    telemetry_data.gpu_temperature = float(temp)

                # Video decoder utilization (percent)
                with contextlib.suppress(NVMLError):
                    dec_util, _ = pynvml.nvmlDeviceGetDecoderUtilization(handle)
                    telemetry_data.decoder_utilization = float(dec_util)

                # Video encoder utilization (percent)
                with contextlib.suppress(NVMLError):
                    enc_util, _ = pynvml.nvmlDeviceGetEncoderUtilization(handle)
                    telemetry_data.encoder_utilization = float(enc_util)

                # JPEG decoder utilization (percent)
                with contextlib.suppress(NVMLError):
                    jpg_util, _ = pynvml.nvmlDeviceGetJpgUtilization(handle)
                    telemetry_data.jpg_utilization = float(jpg_util)

                # SM utilization: prefer GPM (device-level) over process enumeration
                sm_util: float | None = None
                if gpu.gpm_samples:
                    sm_util = self._get_sm_utilization_gpm(gpu)

                # Fallback to process-level API if GPM unavailable or returned None
                if sm_util is None:
                    with contextlib.suppress(NVMLError):
                        process_utils = pynvml.nvmlDeviceGetProcessesUtilizationInfo(
                            handle, 0
                        )
                        sm_util = (
                            sum(p.smUtil for p in process_utils)
                            if process_utils
                            else 0.0
                        )

                if sm_util is not None:
                    telemetry_data.sm_utilization = min(float(sm_util), 100.0)

                # Power violation / throttling duration (nanoseconds -> microseconds)
                with contextlib.suppress(NVMLError):
                    violation = pynvml.nvmlDeviceGetViolationStatus(
                        handle, pynvml.NVML_PERF_POLICY_POWER
                    )
                    telemetry_data.power_violation = (
                        violation.violationTime * ScalingFactors.power_violation
                    )

                # Create record if any metrics were collected
                if telemetry_data.model_fields_set:
                    record = TelemetryRecord(
                        timestamp_ns=current_timestamp,
                        dcgm_url=PYNVML_SOURCE_IDENTIFIER,
                        **gpu.metadata.model_dump(),
                        telemetry_data=telemetry_data,
                    )
                    records.append(record)

            return records
