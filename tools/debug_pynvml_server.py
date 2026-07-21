#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone PyNVML debug HTTP server.

Run with:
    uv run python tools/debug_pynvml_server.py --host 127.0.0.1 --port 8765

Then sample with:
    curl -X POST http://127.0.0.1:8765/start
    curl -X POST http://127.0.0.1:8765/sample
    curl -X POST http://127.0.0.1:8765/stop
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import Callable

import pynvml
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

LOGGER = logging.getLogger("debug_pynvml_server")


class GpuSample(BaseModel):
    """Instantaneous native PyNVML metrics for one GPU."""

    timestamp_ns: int = Field(description="Nanosecond timestamp for this sample")
    gpu_index: int = Field(description="NVML GPU index")
    gpu_uuid: str = Field(description="GPU UUID")
    gpu_model_name: str = Field(description="GPU model name")
    pci_bus_id: str | None = Field(default=None, description="PCI bus identifier")
    power_w: float | None = Field(default=None, description="Instantaneous power in W")
    total_energy_j: float | None = Field(
        default=None, description="Cumulative GPU energy counter in J"
    )
    energy_delta_j: float | None = Field(
        default=None, description="Energy consumed since /start in J"
    )
    gpu_utilization_pct: float | None = Field(
        default=None, description="GPU utilization percentage"
    )
    memory_utilization_pct: float | None = Field(
        default=None, description="Memory utilization percentage"
    )
    memory_used_gb: float | None = Field(default=None, description="GPU memory used")
    sm_utilization_pct: float | None = Field(
        default=None, description="Streaming multiprocessor utilization percentage"
    )
    temperature_c: float | None = Field(
        default=None, description="GPU temperature in Celsius"
    )
    power_violation_us: float | None = Field(
        default=None, description="Power throttling violation duration in microseconds"
    )


class Snapshot(BaseModel):
    """Point-in-time native PyNVML debug snapshot."""

    running: bool = Field(description="Whether NVML sampling is active")
    started_ns: int | None = Field(
        default=None, description="Nanosecond timestamp when sampling started"
    )
    elapsed_sec: float | None = Field(
        default=None, description="Seconds elapsed since /start"
    )
    gpu_count: int = Field(description="Number of GPUs included in the sample")
    total_power_w: float | None = Field(
        default=None, description="Sum of instantaneous GPU power across sampled GPUs"
    )
    total_energy_delta_j: float | None = Field(
        default=None, description="Sum of GPU energy deltas since /start"
    )
    samples: list[GpuSample] = Field(
        default_factory=list, description="Per-GPU native PyNVML samples"
    )


class Status(BaseModel):
    """Current native PyNVML debug server state."""

    running: bool = Field(description="Whether NVML sampling is active")
    started_ns: int | None = Field(
        default=None, description="Nanosecond timestamp when sampling started"
    )
    gpu_count: int = Field(description="Number of GPUs in the most recent sample")
    last_sample: Snapshot | None = Field(
        default=None, description="Most recent native PyNVML snapshot"
    )


class NativePynvmlSampler:
    """Direct PyNVML sampler with start/sample/stop lifecycle."""

    def __init__(self) -> None:
        self._gpu_handles: list[tuple[object, GpuStaticInfo]] = []
        self._nvml_initialized = False
        self._nvml_lock = threading.Lock()
        self._started_ns: int | None = None
        self._baseline_energy_j: dict[str, float] = {}
        self._last_sample: Snapshot | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> Status:
        """Initialize NVML and capture the energy baseline."""
        async with self._lock:
            if self._nvml_initialized:
                return self.status()

            try:
                await asyncio.to_thread(self._initialize_nvml_sync)
                self._started_ns = time.time_ns()
                baseline_samples = await self._read_samples()
                self._baseline_energy_j = {
                    sample.gpu_uuid: sample.total_energy_j
                    for sample in baseline_samples
                    if sample.total_energy_j is not None
                }
                samples = [
                    sample.model_copy(update={"energy_delta_j": 0.0})
                    if sample.total_energy_j is not None
                    else sample
                    for sample in baseline_samples
                ]
                self._last_sample = self._snapshot_from_samples(samples)
                self._log_snapshot(self._last_sample, "start")
                return self.status()
            except Exception as exc:  # report NVML startup failures over HTTP
                await asyncio.to_thread(self._shutdown_nvml_sync)
                self._started_ns = None
                self._baseline_energy_j = {}
                raise HTTPException(
                    status_code=503,
                    detail=f"Native PyNVML sampler failed to start: {exc}",
                ) from exc

    async def sample(self) -> Snapshot:
        """Capture and log one instantaneous native PyNVML sample."""
        async with self._lock:
            if not self._nvml_initialized:
                raise HTTPException(
                    status_code=409,
                    detail="Native PyNVML sampler is not running; call /start first.",
                )

            self._last_sample = self._snapshot_from_samples(
                samples=await self._read_samples()
            )
            self._log_snapshot(self._last_sample, "sample")
            return self._last_sample

    async def stop(self) -> Status:
        """Capture one final sample and shut down NVML."""
        async with self._lock:
            if not self._nvml_initialized:
                return self.status()

            try:
                self._last_sample = self._snapshot_from_samples(
                    samples=await self._read_samples()
                )
                self._log_snapshot(self._last_sample, "stop")
            finally:
                await asyncio.to_thread(self._shutdown_nvml_sync)
                self._started_ns = None
                self._baseline_energy_j = {}
                if self._last_sample is not None:
                    self._last_sample = self._last_sample.model_copy(
                        update={"running": False}
                    )

            return self.status()

    def status(self) -> Status:
        """Return current sampler state."""
        return Status(
            running=self._nvml_initialized,
            started_ns=self._started_ns,
            gpu_count=(
                len(self._last_sample.samples) if self._last_sample is not None else 0
            ),
            last_sample=self._last_sample,
        )

    def _initialize_nvml_sync(self) -> None:
        """Initialize NVML and cache native handles."""
        with self._nvml_lock:
            pynvml.nvmlInit()
            self._nvml_initialized = True
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                self._gpu_handles = [
                    gpu
                    for index in range(device_count)
                    if (gpu := self._create_gpu_handle(index)) is not None
                ]
                if not self._gpu_handles:
                    raise RuntimeError("NVML initialized but no GPUs were available")
            except Exception:  # clean up partially initialized NVML state
                self._shutdown_nvml_unlocked()
                raise

    def _create_gpu_handle(self, index: int) -> tuple[object, GpuStaticInfo] | None:
        """Create one native NVML handle and static metadata pair."""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except pynvml.NVMLError as exc:
            LOGGER.warning("Failed to get GPU %s handle: %s", index, exc)
            return None

        return (
            handle,
            GpuStaticInfo(
                gpu_index=index,
                gpu_uuid=_decode_nvml_string(
                    _read_nvml_value(
                        lambda: pynvml.nvmlDeviceGetUUID(handle),
                        default=f"GPU-unknown-{index}",
                    )
                ),
                gpu_model_name=_decode_nvml_string(
                    _read_nvml_value(
                        lambda: pynvml.nvmlDeviceGetName(handle),
                        default="Unknown GPU",
                    )
                ),
                pci_bus_id=_read_pci_bus_id(handle),
            ),
        )

    def _shutdown_nvml_sync(self) -> None:
        """Shutdown NVML and clear native handles."""
        with self._nvml_lock:
            self._shutdown_nvml_unlocked()

    def _shutdown_nvml_unlocked(self) -> None:
        """Shutdown NVML while the native NVML lock is already held."""
        if not self._nvml_initialized:
            return

        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError as exc:
            LOGGER.warning("Error during NVML shutdown: %s", exc)
        finally:
            self._nvml_initialized = False
            self._gpu_handles = []

    async def _read_samples(self) -> list[GpuSample]:
        """Read one sample from each cached native GPU handle."""
        baseline_energy_j = dict(self._baseline_energy_j)
        return await asyncio.to_thread(self._read_samples_sync, baseline_energy_j)

    def _read_samples_sync(
        self, baseline_energy_j: dict[str, float]
    ) -> list[GpuSample]:
        """Synchronous native PyNVML sample collection."""
        with self._nvml_lock:
            if not self._nvml_initialized:
                return []

            timestamp_ns = time.time_ns()
            return [
                self._read_gpu_sample(
                    timestamp_ns=timestamp_ns,
                    handle=handle,
                    static_info=static_info,
                    baseline_energy_j=baseline_energy_j,
                )
                for handle, static_info in self._gpu_handles
            ]

    def _read_gpu_sample(
        self,
        timestamp_ns: int,
        handle: object,
        static_info: GpuStaticInfo,
        baseline_energy_j: dict[str, float],
    ) -> GpuSample:
        """Read direct PyNVML values for one GPU."""
        total_energy_j = _read_total_energy_j(handle)
        baseline = baseline_energy_j.get(static_info.gpu_uuid)
        energy_delta_j = (
            total_energy_j - baseline
            if total_energy_j is not None and baseline is not None
            else None
        )
        gpu_utilization_pct, memory_utilization_pct = _read_utilization(handle)

        return GpuSample(
            timestamp_ns=timestamp_ns,
            gpu_index=static_info.gpu_index,
            gpu_uuid=static_info.gpu_uuid,
            gpu_model_name=static_info.gpu_model_name,
            pci_bus_id=static_info.pci_bus_id,
            power_w=_read_power_w(handle),
            total_energy_j=total_energy_j,
            energy_delta_j=energy_delta_j,
            gpu_utilization_pct=gpu_utilization_pct,
            memory_utilization_pct=memory_utilization_pct,
            memory_used_gb=_read_memory_used_gb(handle),
            sm_utilization_pct=_read_sm_utilization_pct(handle),
            temperature_c=_read_temperature_c(handle),
            power_violation_us=_read_power_violation_us(handle),
        )

    def _snapshot_from_samples(self, samples: list[GpuSample]) -> Snapshot:
        """Build an aggregate snapshot from per-GPU samples."""
        power_values = [
            sample.power_w for sample in samples if sample.power_w is not None
        ]
        delta_values = [
            sample.energy_delta_j
            for sample in samples
            if sample.energy_delta_j is not None
        ]
        elapsed_sec = (
            (time.time_ns() - self._started_ns) / 1_000_000_000
            if self._started_ns is not None
            else None
        )
        return Snapshot(
            running=self._nvml_initialized,
            started_ns=self._started_ns,
            elapsed_sec=elapsed_sec,
            gpu_count=len(samples),
            total_power_w=sum(power_values) if power_values else None,
            total_energy_delta_j=sum(delta_values) if delta_values else None,
            samples=samples,
        )

    def _log_snapshot(self, snapshot: Snapshot, action: str) -> None:
        """Log compact per-GPU native PyNVML readings."""
        LOGGER.info(
            "%s: gpus=%d total_power=%s energy_delta=%s",
            action,
            snapshot.gpu_count,
            _format_metric(snapshot.total_power_w, "W"),
            _format_metric(snapshot.total_energy_delta_j, "J"),
        )
        for sample in snapshot.samples:
            LOGGER.info(
                "gpu%d: power=%s energy_counter=%s energy_delta=%s "
                "gpu_util=%s sm_util=%s mem_util=%s mem_used=%s temp=%s",
                sample.gpu_index,
                _format_metric(sample.power_w, "W"),
                _format_metric(sample.total_energy_j, "J"),
                _format_metric(sample.energy_delta_j, "J"),
                _format_metric(sample.gpu_utilization_pct, "%"),
                _format_metric(sample.sm_utilization_pct, "%"),
                _format_metric(sample.memory_utilization_pct, "%"),
                _format_metric(sample.memory_used_gb, "GB"),
                _format_metric(sample.temperature_c, "C"),
            )


class GpuStaticInfo(BaseModel):
    """Static GPU metadata cached from NVML initialization."""

    gpu_index: int = Field(description="NVML GPU index")
    gpu_uuid: str = Field(description="GPU UUID")
    gpu_model_name: str = Field(description="GPU model name")
    pci_bus_id: str | None = Field(default=None, description="PCI bus identifier")


sampler = NativePynvmlSampler()
app = FastAPI(
    title="PyNVML Debug Server",
    description="Standalone local GPU power/utilization sampler using native PyNVML.",
)


@app.get("/", response_model=Status)
async def get_root_status() -> Status:
    """Return current sampler state."""
    return sampler.status()


@app.get("/status", response_model=Status)
async def get_status() -> Status:
    """Return current sampler state."""
    return sampler.status()


@app.post("/start", response_model=Status)
async def start_sampler() -> Status:
    """Start native PyNVML sampling and capture an energy baseline."""
    return await sampler.start()


@app.post("/sample", response_model=Snapshot)
async def sample() -> Snapshot:
    """Capture and log one instantaneous native PyNVML sample."""
    return await sampler.sample()


@app.post("/stop", response_model=Status)
async def stop_sampler() -> Status:
    """Capture one final native PyNVML sample and shut down NVML."""
    return await sampler.stop()


def _read_power_w(handle: object) -> float | None:
    """Read instantaneous power in watts."""
    value = _read_nvml_value(lambda: pynvml.nvmlDeviceGetPowerUsage(handle))
    return value * 1e-3 if value is not None else None


def _read_total_energy_j(handle: object) -> float | None:
    """Read cumulative energy in joules."""
    value = _read_nvml_value(lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(handle))
    return value * 1e-3 if value is not None else None


def _read_utilization(handle: object) -> tuple[float | None, float | None]:
    """Read GPU and memory utilization percentages."""
    util = _read_nvml_value(lambda: pynvml.nvmlDeviceGetUtilizationRates(handle))
    if util is None:
        return None, None
    return float(util.gpu), float(util.memory)


def _read_memory_used_gb(handle: object) -> float | None:
    """Read used GPU memory in GB."""
    mem_info = _read_nvml_value(lambda: pynvml.nvmlDeviceGetMemoryInfo(handle))
    return mem_info.used * 1e-9 if mem_info is not None else None


def _read_temperature_c(handle: object) -> float | None:
    """Read GPU temperature in Celsius."""
    value = _read_nvml_value(
        lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    )
    return float(value) if value is not None else None


def _read_sm_utilization_pct(handle: object) -> float | None:
    """Read SM utilization via the process utilization API."""
    get_processes = getattr(pynvml, "nvmlDeviceGetProcessesUtilizationInfo", None)
    if get_processes is None:
        return None
    with contextlib.suppress(pynvml.NVMLError):
        processes = get_processes(handle, 0)
        sm_util = sum(process.smUtil for process in processes) if processes else 0.0
        return min(float(sm_util), 100.0)
    return None


def _read_power_violation_us(handle: object) -> float | None:
    """Read power throttling violation duration in microseconds."""
    value = _read_nvml_value(
        lambda: pynvml.nvmlDeviceGetViolationStatus(
            handle, pynvml.NVML_PERF_POLICY_POWER
        )
    )
    return value.violationTime * 1e-3 if value is not None else None


def _read_pci_bus_id(handle: object) -> str | None:
    """Read PCI bus id for a GPU handle."""
    pci_info = _read_nvml_value(lambda: pynvml.nvmlDeviceGetPciInfo(handle))
    if pci_info is None:
        return None
    return _decode_nvml_string(pci_info.busId)


def _read_nvml_value(
    call: Callable[[], object], default: object | None = None
) -> object | None:
    """Read an NVML value, returning default when the API is unavailable."""
    try:
        return call()
    except (pynvml.NVMLError, AttributeError):
        return default


def _decode_nvml_string(value: object) -> str:
    """Decode string-like values returned by different pynvml versions."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _format_metric(value: float | None, unit: str) -> str:
    """Format optional numeric metrics for compact logs."""
    if value is None:
        return "n/a"
    return f"{value:.2f}{unit}"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", default=8765, type=int, help="Port to bind")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=("critical", "error", "warning", "info", "debug"),
        help="Uvicorn log level",
    )
    return parser.parse_args()


def main() -> None:
    """Run the standalone debug server."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
