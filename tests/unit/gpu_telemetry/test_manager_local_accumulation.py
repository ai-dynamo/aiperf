# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local accumulation tests for GPUTelemetryManager.

These tests exercise the in-process fan-out path (record callback -> local
processors) and the PROFILE_COMPLETE -> ProcessTelemetryResultMessage publish
path without touching ZMQ. The side-channel wire path was removed in the
records-manager decoupling refactor.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from aiperf.common.control_structs import Command
from aiperf.common.enums import CommandType
from aiperf.common.messages import ProcessTelemetryResultMessage
from aiperf.common.models import TelemetryRecord
from aiperf.common.models.export_models import (
    EndpointData,
    TelemetryExportData,
    TelemetrySummary,
)
from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.gpu_telemetry.manager import GPUTelemetryManager

_BASE = dict(
    models=["test-model"],
    endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
    datasets={
        "default": {
            "type": "synthetic",
            "entries": 100,
            "prompts": {"isl": 128, "osl": 64},
        }
    },
    phases={"default": {"type": "concurrency", "requests": 10, "concurrency": 1}},
)


def _make_run() -> BenchmarkRun:
    return BenchmarkRun(
        benchmark_id="test",
        cfg=AIPerfConfig(**_BASE),
        artifact_dir=Path("/tmp/test"),
    )


def _make_record(
    gpu_index: int = 0, timestamp_ns: int = 1_000_000_000
) -> TelemetryRecord:
    return TelemetryRecord(
        timestamp_ns=timestamp_ns,
        dcgm_url="http://localhost:9400/metrics",
        gpu_index=gpu_index,
        gpu_uuid=f"GPU-{gpu_index:08x}",
        gpu_model_name="Test GPU",
        telemetry_data={"gpu_power_usage": 100.0},
    )


class _FakeTelemetryProcessor:
    """Minimal processor stub that records every incoming record."""

    def __init__(self, raise_on_process: bool = False) -> None:
        self.processed: list[TelemetryRecord] = []
        self._raise_on_process = raise_on_process

    async def process_telemetry_record(self, record: TelemetryRecord) -> None:
        if self._raise_on_process:
            raise ValueError("simulated processor failure")
        self.processed.append(record)


class _FakeTelemetryAccumulator(_FakeTelemetryProcessor):
    """Fake accumulator that returns a known TelemetryExportData."""

    def __init__(self, export_data: TelemetryExportData | None) -> None:
        super().__init__()
        self._export_data = export_data
        self.export_calls: list[dict] = []

    def export_results(
        self,
        start_ns: int = 0,
        end_ns: int | None = None,
        error_summary=None,
    ) -> TelemetryExportData | None:
        self.export_calls.append(
            {"start_ns": start_ns, "end_ns": end_ns, "error_summary": error_summary}
        )
        return self._export_data


def _make_export_data() -> TelemetryExportData:
    """Build a minimal TelemetryExportData with required summary fields."""
    now = datetime.now(timezone.utc)
    return TelemetryExportData(
        summary=TelemetrySummary(start_time=now, end_time=now),
        endpoints={"http://localhost:9400/metrics": EndpointData(gpus={})},
    )


@pytest.fixture
def manager() -> GPUTelemetryManager:
    """Build a GPUTelemetryManager with its BaseComponentService fully constructed,
    then replace processor/accumulator state so tests control fan-out and export."""
    mgr = GPUTelemetryManager(run=_make_run())
    mgr._processors = []
    mgr._accumulator = None
    mgr._result_published = False
    # Replace pub_client with an async-capable mock so publish() is observable.
    mgr.pub_client = AsyncMock()
    return mgr


@pytest.mark.asyncio
async def test_telemetry_manager_fans_out_records_to_processors(
    manager: GPUTelemetryManager,
) -> None:
    """Every record is delivered to every loaded processor."""
    proc = _FakeTelemetryProcessor()
    manager._processors = [proc]

    rec1 = _make_record(gpu_index=0)
    rec2 = _make_record(gpu_index=1)
    await manager._on_telemetry_records([rec1, rec2], collector_id="col-1")

    assert proc.processed == [rec1, rec2]


@pytest.mark.asyncio
async def test_telemetry_manager_publishes_result_on_profile_complete(
    manager: GPUTelemetryManager,
) -> None:
    """PROFILE_COMPLETE publishes a single ProcessTelemetryResultMessage wrapping
    the accumulator's export. start_ns from payload is forwarded."""
    export = _make_export_data()
    accumulator = _FakeTelemetryAccumulator(export_data=export)
    manager._accumulator = accumulator
    manager._processors = [accumulator]

    payload = orjson.dumps({"start_ns": 123_456_789}).decode()
    await manager._handle_profile_complete_command(
        Command(cid="test", cmd=CommandType.PROFILE_COMPLETE, payload=payload)
    )

    assert manager.pub_client.publish.await_count == 1
    published = manager.pub_client.publish.await_args[0][0]
    assert isinstance(published, ProcessTelemetryResultMessage)
    assert published.telemetry_result.results is export
    assert accumulator.export_calls[0]["start_ns"] == 123_456_789


@pytest.mark.asyncio
async def test_telemetry_manager_publishes_null_result_when_no_accumulator(
    manager: GPUTelemetryManager,
) -> None:
    """With no accumulator loaded, PROFILE_COMPLETE still publishes exactly one
    message, with results=None."""
    manager._accumulator = None

    await manager._handle_profile_complete_command(
        Command(cid="test", cmd=CommandType.PROFILE_COMPLETE)
    )

    assert manager.pub_client.publish.await_count == 1
    published = manager.pub_client.publish.await_args[0][0]
    assert isinstance(published, ProcessTelemetryResultMessage)
    assert published.telemetry_result.results is None


@pytest.mark.asyncio
async def test_telemetry_manager_idempotent_publish(
    manager: GPUTelemetryManager,
) -> None:
    """Calling PROFILE_COMPLETE twice must publish exactly one
    ProcessTelemetryResultMessage thanks to the _result_published latch."""
    export = _make_export_data()
    accumulator = _FakeTelemetryAccumulator(export_data=export)
    manager._accumulator = accumulator

    cmd = Command(cid="test", cmd=CommandType.PROFILE_COMPLETE)
    await manager._handle_profile_complete_command(cmd)
    await manager._handle_profile_complete_command(cmd)

    assert manager.pub_client.publish.await_count == 1


@pytest.mark.asyncio
async def test_telemetry_manager_tracks_errors_from_failed_processor(
    manager: GPUTelemetryManager,
) -> None:
    """If a processor raises while processing a record, the error is captured
    in the manager's error_state (it must not propagate)."""
    failing = _FakeTelemetryProcessor(raise_on_process=True)
    manager._processors = [failing]
    # Silence the exception() log call from the manager.
    manager.exception = MagicMock()

    await manager._on_telemetry_records([_make_record()], collector_id="col-1")

    assert sum(manager._error_state.error_counts.values()) == 1
