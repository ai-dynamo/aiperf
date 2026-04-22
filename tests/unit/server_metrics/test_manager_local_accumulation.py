# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local accumulation tests for ServerMetricsManager.

These tests exercise the in-process fan-out path (record callback -> local
processors) and the PROFILE_COMPLETE -> ProcessServerMetricsResultMessage
publish path without touching ZMQ. The side-channel wire path was removed
in the records-manager decoupling refactor.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from aiperf.common.control_structs import Command
from aiperf.common.enums import CommandType
from aiperf.common.messages import ProcessServerMetricsResultMessage
from aiperf.common.models.server_metrics_models import (
    ServerMetricsRecord,
    ServerMetricsResults,
)
from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.server_metrics.manager import ServerMetricsManager

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


def _make_record(timestamp_ns: int = 1_000_000_000) -> ServerMetricsRecord:
    return ServerMetricsRecord(
        endpoint_url="http://localhost:8081/metrics",
        timestamp_ns=timestamp_ns,
        endpoint_latency_ns=5_000_000,
        metrics={},
    )


def _make_results(start_ns: int, end_ns: int) -> ServerMetricsResults:
    return ServerMetricsResults(start_ns=start_ns, end_ns=end_ns)


class _FakeServerMetricsProcessor:
    """Minimal processor stub that records every incoming record."""

    def __init__(self, raise_on_process: bool = False) -> None:
        self.processed: list[ServerMetricsRecord] = []
        self._raise_on_process = raise_on_process

    async def process_server_metrics_record(self, record: ServerMetricsRecord) -> None:
        if self._raise_on_process:
            raise ValueError("simulated processor failure")
        self.processed.append(record)

    async def summarize(self):
        return []


class _FakeServerMetricsAccumulator(_FakeServerMetricsProcessor):
    """Fake accumulator returning a known ServerMetricsResults."""

    def __init__(self, results: ServerMetricsResults | None) -> None:
        super().__init__()
        self._results = results
        self.export_calls: list[dict] = []

    async def export_results(
        self,
        start_ns: int,
        end_ns: int,
        time_filter=None,
        error_summary=None,
    ) -> ServerMetricsResults | None:
        self.export_calls.append(
            {
                "start_ns": start_ns,
                "end_ns": end_ns,
                "time_filter": time_filter,
                "error_summary": error_summary,
            }
        )
        return self._results


@pytest.fixture
def manager() -> ServerMetricsManager:
    """Build a ServerMetricsManager, then strip processor/accumulator state so
    tests control fan-out and export directly."""
    mgr = ServerMetricsManager(run=_make_run())
    mgr._processors = []
    mgr._accumulator = None
    mgr._collectors = {}
    mgr._result_published = False
    mgr.pub_client = AsyncMock()
    return mgr


@pytest.mark.asyncio
async def test_server_metrics_manager_fans_out_records_to_processors(
    manager: ServerMetricsManager,
) -> None:
    """Every record is delivered to every loaded processor."""
    proc = _FakeServerMetricsProcessor()
    manager._processors = [proc]

    rec1 = _make_record(timestamp_ns=1_000_000_000)
    rec2 = _make_record(timestamp_ns=2_000_000_000)
    await manager._on_server_metrics_records([rec1, rec2], collector_id="col-1")

    assert proc.processed == [rec1, rec2]


@pytest.mark.asyncio
async def test_server_metrics_manager_publishes_result_on_profile_complete(
    manager: ServerMetricsManager,
) -> None:
    """PROFILE_COMPLETE with {start_ns, end_ns} publishes one
    ProcessServerMetricsResultMessage wrapping the accumulator's export."""
    expected = _make_results(start_ns=111, end_ns=222)
    accumulator = _FakeServerMetricsAccumulator(results=expected)
    manager._accumulator = accumulator
    manager._processors = [accumulator]

    payload = orjson.dumps({"start_ns": 111, "end_ns": 222}).decode()
    await manager._handle_profile_complete_command(
        Command(cid="test", cmd=CommandType.PROFILE_COMPLETE, payload=payload)
    )

    assert manager.pub_client.publish.await_count == 1
    published = manager.pub_client.publish.await_args[0][0]
    assert isinstance(published, ProcessServerMetricsResultMessage)
    assert published.server_metrics_result.results is expected
    assert accumulator.export_calls[0]["start_ns"] == 111
    assert accumulator.export_calls[0]["end_ns"] == 222


@pytest.mark.asyncio
async def test_server_metrics_manager_publishes_null_result_when_no_accumulator(
    manager: ServerMetricsManager,
) -> None:
    """With no accumulator, PROFILE_COMPLETE still publishes exactly one message
    with results=None."""
    manager._accumulator = None

    await manager._handle_profile_complete_command(
        Command(cid="test", cmd=CommandType.PROFILE_COMPLETE)
    )

    assert manager.pub_client.publish.await_count == 1
    published = manager.pub_client.publish.await_args[0][0]
    assert isinstance(published, ProcessServerMetricsResultMessage)
    assert published.server_metrics_result.results is None


@pytest.mark.asyncio
async def test_server_metrics_manager_idempotent_publish(
    manager: ServerMetricsManager,
) -> None:
    """Calling PROFILE_COMPLETE twice must publish exactly one
    ProcessServerMetricsResultMessage thanks to the _result_published latch."""
    expected = _make_results(start_ns=111, end_ns=222)
    accumulator = _FakeServerMetricsAccumulator(results=expected)
    manager._accumulator = accumulator

    cmd = Command(cid="test", cmd=CommandType.PROFILE_COMPLETE)
    await manager._handle_profile_complete_command(cmd)
    await manager._handle_profile_complete_command(cmd)

    assert manager.pub_client.publish.await_count == 1


@pytest.mark.asyncio
async def test_server_metrics_manager_tracks_errors_from_failed_processor(
    manager: ServerMetricsManager,
) -> None:
    """Processor exceptions during fan-out are captured in error_state, not
    propagated out of the callback."""
    failing = _FakeServerMetricsProcessor(raise_on_process=True)
    manager._processors = [failing]
    manager.exception = MagicMock()

    await manager._on_server_metrics_records([_make_record()], collector_id="col-1")

    assert sum(manager._error_state.error_counts.values()) == 1


@pytest.mark.asyncio
async def test_server_metrics_manager_falls_back_to_zero_start_when_end_before_start(
    manager: ServerMetricsManager,
) -> None:
    """Task 3 Fix A: if the payload's end_ns is earlier than start_ns (fallback
    path clock-skew / stale payload), the manager emits a warning and rewrites
    start_ns=0 so the accumulator still produces a full-history export."""
    expected = _make_results(start_ns=0, end_ns=500)
    accumulator = _FakeServerMetricsAccumulator(results=expected)
    manager._accumulator = accumulator
    manager.warning = MagicMock()

    # end_ns earlier than start_ns triggers the time-window guard.
    payload = orjson.dumps({"start_ns": 1000, "end_ns": 500}).decode()
    await manager._handle_profile_complete_command(
        Command(cid="test", cmd=CommandType.PROFILE_COMPLETE, payload=payload)
    )

    assert accumulator.export_calls, "accumulator must have been invoked"
    call = accumulator.export_calls[0]
    assert call["start_ns"] == 0
    assert call["end_ns"] == 500
    manager.warning.assert_called_once()
    assert "Invalid time window" in manager.warning.call_args[0][0]
