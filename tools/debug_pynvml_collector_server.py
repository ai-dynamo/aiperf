#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone PyNVML collector debug HTTP server.

This server samples through AIPerf's PyNVMLTelemetryCollector without registering
an AIPerf API router or starting the normal AIPerf service graph.

Run with:
    uv run python tools/debug_pynvml_collector_server.py --host 127.0.0.1 --port 8766

Then sample with:
    curl -X POST http://127.0.0.1:8766/start
    curl -X POST http://127.0.0.1:8766/sample
    curl -X POST http://127.0.0.1:8766/stop
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from aiperf.common.enums import LifecycleState
from aiperf.common.models import ErrorDetails, TelemetryRecord
from aiperf.gpu_telemetry.pynvml_collector import PyNVMLTelemetryCollector

LOGGER = logging.getLogger("debug_pynvml_collector_server")


class GpuSample(BaseModel):
    """Instantaneous AIPerf collector metrics for one GPU."""

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
    """Point-in-time PyNVML collector debug snapshot."""

    running: bool = Field(description="Whether collector sampling is active")
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
        default_factory=list, description="Per-GPU collector samples"
    )


class Status(BaseModel):
    """Current PyNVML collector debug server state."""

    running: bool = Field(description="Whether collector sampling is active")
    started_ns: int | None = Field(
        default=None, description="Nanosecond timestamp when sampling started"
    )
    gpu_count: int = Field(description="Number of GPUs in the most recent sample")
    last_sample: Snapshot | None = Field(
        default=None, description="Most recent collector-backed snapshot"
    )


class CollectorPynvmlSampler:
    """Sampler that drives AIPerf's PyNVMLTelemetryCollector on demand."""

    def __init__(self) -> None:
        self._collector: PyNVMLTelemetryCollector | None = None
        self._started_ns: int | None = None
        self._baseline_energy_j: dict[str, float] = {}
        self._last_sample: Snapshot | None = None
        self._latest_records: list[TelemetryRecord] = []
        self._lock = asyncio.Lock()

    async def start(self) -> Status:
        """Initialize the AIPerf collector and capture an energy baseline."""
        async with self._lock:
            if self._collector is not None:
                return self.status()

            collector = PyNVMLTelemetryCollector(
                record_callback=self._on_records,
                error_callback=self._on_error,
                collector_id="pynvml_debug_collector",
            )
            try:
                await collector.initialize()
                self._collector = collector
                self._started_ns = time.time_ns()
                records = await self._collect_once()
                self._baseline_energy_j = {
                    record.gpu_uuid: total_energy_j
                    for record in records
                    if (total_energy_j := _total_energy_j(record)) is not None
                }
                self._last_sample = self._snapshot_from_records(records)
                self._log_snapshot(self._last_sample, "start")
                return self.status()
            except asyncio.CancelledError as exc:
                if collector.state != LifecycleState.FAILED:
                    raise
                await self._cleanup_failed_start(collector)
                raise HTTPException(
                    status_code=503,
                    detail=f"PyNVML collector sampler failed to start: {exc}",
                ) from exc
            except Exception as exc:  # noqa: BLE001 - report collector startup failures over HTTP
                await self._cleanup_failed_start(collector)
                raise HTTPException(
                    status_code=503,
                    detail=f"PyNVML collector sampler failed to start: {exc}",
                ) from exc

    async def sample(self) -> Snapshot:
        """Capture and log one instantaneous collector-backed sample."""
        async with self._lock:
            if self._collector is None:
                raise HTTPException(
                    status_code=409,
                    detail="PyNVML collector sampler is not running; call /start first.",
                )

            self._last_sample = self._snapshot_from_records(await self._collect_once())
            self._log_snapshot(self._last_sample, "sample")
            return self._last_sample

    async def stop(self) -> Status:
        """Capture one final sample and stop the AIPerf collector."""
        async with self._lock:
            collector = self._collector
            if collector is None:
                return self.status()

            try:
                self._last_sample = self._snapshot_from_records(
                    await self._collect_once()
                )
                self._log_snapshot(self._last_sample, "stop")
            finally:
                await collector.stop()
                self._collector = None
                self._started_ns = None
                self._baseline_energy_j = {}
                self._latest_records = []
                if self._last_sample is not None:
                    self._last_sample = self._last_sample.model_copy(
                        update={"running": False}
                    )

            return self.status()

    def status(self) -> Status:
        """Return current sampler state."""
        return Status(
            running=self._collector is not None,
            started_ns=self._started_ns,
            gpu_count=(
                len(self._last_sample.samples) if self._last_sample is not None else 0
            ),
            last_sample=self._last_sample,
        )

    async def _collect_once(self) -> list[TelemetryRecord]:
        """Run one collector pass and return callback-delivered records."""
        if self._collector is None:
            return []

        self._latest_records = []
        await self._collector.collect_and_process_metrics()
        return list(self._latest_records)

    async def _on_records(
        self, records: list[TelemetryRecord], collector_id: str
    ) -> None:
        """Capture records produced by the AIPerf collector callback."""
        self._latest_records = list(records)
        LOGGER.debug("collector %s returned %d record(s)", collector_id, len(records))

    async def _on_error(self, error: ErrorDetails, collector_id: str) -> None:
        """Log collector errors from callback delivery."""
        LOGGER.warning("collector %s error: %s", collector_id, error.message)

    async def _cleanup_failed_start(self, collector: PyNVMLTelemetryCollector) -> None:
        """Reset local state after a failed collector start."""
        self._collector = None
        self._started_ns = None
        self._baseline_energy_j = {}
        self._latest_records = []
        if collector.state != LifecycleState.STOPPED:
            await collector.stop()

    def _snapshot_from_records(self, records: list[TelemetryRecord]) -> Snapshot:
        """Convert AIPerf telemetry records into a debug snapshot."""
        samples = [self._sample_from_record(record) for record in records]
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
            running=self._collector is not None,
            started_ns=self._started_ns,
            elapsed_sec=elapsed_sec,
            gpu_count=len(samples),
            total_power_w=sum(power_values) if power_values else None,
            total_energy_delta_j=sum(delta_values) if delta_values else None,
            samples=samples,
        )

    def _sample_from_record(self, record: TelemetryRecord) -> GpuSample:
        """Convert one AIPerf telemetry record into a debug GPU sample."""
        metrics = record.telemetry_data
        total_energy_j = _total_energy_j(record)
        baseline_energy_j = self._baseline_energy_j.get(record.gpu_uuid)
        energy_delta_j = (
            total_energy_j - baseline_energy_j
            if total_energy_j is not None and baseline_energy_j is not None
            else None
        )
        return GpuSample(
            timestamp_ns=record.timestamp_ns,
            gpu_index=record.gpu_index,
            gpu_uuid=record.gpu_uuid,
            gpu_model_name=record.gpu_model_name,
            pci_bus_id=record.pci_bus_id,
            power_w=metrics.gpu_power_usage,
            total_energy_j=total_energy_j,
            energy_delta_j=energy_delta_j,
            gpu_utilization_pct=metrics.gpu_utilization,
            memory_utilization_pct=metrics.mem_utilization,
            memory_used_gb=metrics.gpu_memory_used,
            sm_utilization_pct=metrics.sm_utilization,
            temperature_c=metrics.gpu_temperature,
            power_violation_us=metrics.power_violation,
        )

    def _log_snapshot(self, snapshot: Snapshot, action: str) -> None:
        """Log compact per-GPU collector readings."""
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


sampler = CollectorPynvmlSampler()
app = FastAPI(
    title="PyNVML Collector Debug Server",
    description="Standalone local GPU sampler using AIPerf's PyNVMLTelemetryCollector.",
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
    """Start collector sampling and capture an energy baseline."""
    return await sampler.start()


@app.post("/sample", response_model=Snapshot)
async def sample() -> Snapshot:
    """Capture and log one instantaneous collector-backed sample."""
    return await sampler.sample()


@app.post("/stop", response_model=Status)
async def stop_sampler() -> Status:
    """Capture one final sample and stop the collector."""
    return await sampler.stop()


def _total_energy_j(record: TelemetryRecord) -> float | None:
    """Return the AIPerf collector cumulative energy counter in joules."""
    if record.telemetry_data.energy_consumption is None:
        return None
    return record.telemetry_data.energy_consumption * 1_000_000


def _format_metric(value: float | None, unit: str) -> str:
    """Format optional numeric metrics for compact logs."""
    if value is None:
        return "n/a"
    return f"{value:.2f}{unit}"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", default=8766, type=int, help="Port to bind")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=("critical", "error", "warning", "info", "debug"),
        help="Uvicorn log level",
    )
    return parser.parse_args()


def main() -> None:
    """Run the standalone collector-backed debug server."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
