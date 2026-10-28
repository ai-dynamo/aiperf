# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import errno
import mmap
import os
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from pytest import param

from aiperf.common.accumulator_protocols import ExportContext
from aiperf.common.enums import CreditPhase, PrometheusMetricType
from aiperf.common.environment import Environment
from aiperf.common.models import (
    ErrorDetails,
    MetricFamily,
    MetricSample,
    ServerMetricsRecord,
    TimeRangeFilter,
)
from aiperf.config import BenchmarkRun
from aiperf.server_metrics import array_store
from aiperf.server_metrics.accumulator import ServerMetricsAccumulator
from aiperf.server_metrics.array_store import (
    ServerMetricsArrayStore,
    ServerMetricsStorageError,
    run_history_io,
)
from aiperf.server_metrics.data_collector import ServerMetricsDataCollector
from aiperf.server_metrics.manager import ServerMetricsManager
from aiperf.server_metrics.storage import (
    HistogramTimeSeries,
    ScalarTimeSeries,
    ServerMetricsTimeSeries,
)


def metric_record(index: int, name: str = "gauge") -> ServerMetricsRecord:
    """Produce exact scalar and histogram histories with a warmup boundary."""
    return ServerMetricsRecord(
        endpoint_url="http://server:8000/metrics",
        timestamp_ns=1_000_000_000 + index * 1_000_000,
        endpoint_latency_ns=1_000 + index,
        is_duplicate=index % 3 == 0,
        benchmark_phase=CreditPhase.WARMUP if index < 100 else CreditPhase.PROFILING,
        metrics={
            name: MetricFamily(
                type=PrometheusMetricType.GAUGE,
                description="Gauge",
                samples=[MetricSample(value=float(index))],
            ),
            "counter": MetricFamily(
                type=PrometheusMetricType.COUNTER,
                description="Counter",
                samples=[MetricSample(value=float(index * 2))],
            ),
            "histogram": MetricFamily(
                type=PrometheusMetricType.HISTOGRAM,
                description="Histogram",
                samples=[
                    MetricSample(
                        buckets={"1": index, "+Inf": index * 2},
                        sum=float(index * 3),
                        count=float(index * 2),
                    )
                ],
            ),
        },
    )


@pytest.mark.asyncio
async def test_disk_histories_match_exact_in_memory_exports(
    benchmark_run: BenchmarkRun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spilling preserves samples, out-of-order inserts, phases, and statistics."""
    benchmark_run.cfg.artifacts.dir = tmp_path
    monkeypatch.setattr(Environment.SERVER_METRICS, "STORAGE_FLUSH_BYTES", 4096)
    disk = ServerMetricsAccumulator(run=benchmark_run)
    reference = ServerMetricsAccumulator(run=benchmark_run)
    reference._array_store = MagicMock(spec=ServerMetricsArrayStore)
    reference._array_store.initialized = True
    reference._array_store.dirty_bytes = 0
    reference._array_store.flush_bytes = 1_000_000
    indices = list(range(600))
    indices[101], indices[102] = indices[102], indices[101]
    try:
        for index in indices:
            record = metric_record(index)
            await disk.process_record(record)
            await reference.process_record(record)
        context = ExportContext(
            start_ns=1_100_000_000,
            end_ns=1_599_000_000,
            warmup_start_ns=1_000_000_000,
            warmup_end_ns=1_099_000_000,
        )
        actual = await disk.export_results(context)
        expected = await reference.export_results(context)
        assert actual.model_dump(mode="json") == expected.model_dump(mode="json")
        endpoint = next(iter(disk.get_hierarchy_for_export().endpoints.values()))
        for entry in endpoint.metrics.values():
            assert len(entry.data) == 600
            assert isinstance(entry.data._timestamps.base, mmap.mmap)
        assert isinstance(endpoint._fetch_latencies_ns._data.base, mmap.mmap)
        assert isinstance(disk._timestamps_ns._data.base, mmap.mmap)
        assert disk._array_store.dirty_bytes < 4096
    finally:
        await disk._close_storage()


@pytest.mark.asyncio
async def test_mapping_growth_and_cache_release_preserve_existing_views(
    tmp_path: Path,
) -> None:
    """Growing the backing file cannot invalidate arrays or discard dirty samples."""
    store = ServerMetricsArrayStore(tmp_path, max_bytes=1024 * 1024, flush_bytes=4096)
    await run_history_io(store.open)
    try:
        first = await run_history_io(store.allocate, 512, np.int64)
        first[:] = np.arange(512)
        second = await run_history_io(store.allocate, 32768, np.float64)
        second[:] = 42.5
        await run_history_io(store.flush)
        np.testing.assert_array_equal(first, np.arange(512))
        np.testing.assert_array_equal(second, np.full(32768, 42.5))
        with pytest.raises(ServerMetricsStorageError, match="disk budget"):
            await run_history_io(store.allocate, 1024 * 1024, np.float64)
        np.testing.assert_array_equal(first, np.arange(512))
        assert store.allocated_bytes <= store.max_bytes
    finally:
        await run_history_io(store.close)


@pytest.mark.asyncio
async def test_cancelled_history_write_finishes_before_lock_can_release() -> None:
    """A cancelled to_thread await must not leave writes racing final export."""
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def operation() -> None:
        started.set()
        assert release.wait(timeout=5)
        finished.set()

    task = asyncio.create_task(run_history_io(operation))
    assert await asyncio.to_thread(started.wait, 5)
    try:
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        task.cancel()
        for _ in range(3):
            await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


@pytest.mark.asyncio
async def test_slow_periodic_scrape_has_only_one_in_flight() -> None:
    """Repeated timer ticks cannot retain arbitrarily many pending responses."""
    collector = ServerMetricsDataCollector(endpoint_url="http://server:8000/metrics")
    release = asyncio.Event()
    collector.collect_and_process_metrics = AsyncMock(side_effect=release.wait)
    try:
        for _ in range(1000):
            await collector._collect_metrics_loop()
        assert len(collector.tasks) == 1
        release.set()
        await collector._collection_task
        collector.collect_and_process_metrics.assert_awaited_once()
    finally:
        release.set()
        await collector.wait_for_tasks()


@pytest.mark.asyncio
async def test_series_budget_stops_telemetry_and_exports_existing_results(
    benchmark_run: BenchmarkRun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changing label/name set is bounded and produces a terminal partial result."""
    benchmark_run.cfg.artifacts.dir = tmp_path
    monkeypatch.setattr(Environment.SERVER_METRICS, "MAX_SERIES", 3)
    manager = ServerMetricsManager(run=benchmark_run)
    manager.publish = AsyncMock()
    for index in range(150):
        manager._error_state.record(
            ErrorDetails(type="NetworkError", message=f"Endpoint failure {index}")
        )
    try:
        await manager._on_server_metrics_records([metric_record(100)], "collector")
        await manager._on_server_metrics_records(
            [metric_record(101, "new_gauge")], "collector"
        )
        assert manager._collection_failure.type == "ServerMetricsCollectionStopped"
        assert "3-series limit" in manager._collection_failure.message
        assert len(manager._error_state.error_counts) <= 102
        assert sum(manager._error_state.error_counts.values()) == 152
        before = manager._accumulator._timestamps_ns.data.copy()
        await manager._on_server_metrics_records(
            [metric_record(102, "another_gauge")], "collector"
        )
        np.testing.assert_array_equal(manager._accumulator._timestamps_ns.data, before)
        await manager.wait_for_tasks()
        await manager._publish_server_metrics_result(
            start_ns=1_100_000_000,
            end_ns=1_200_000_000,
            warmup_start_ns=None,
            warmup_end_ns=None,
        )
        result = manager.publish.await_args.args[0].server_metrics_result.results
        assert result is not None
        assert result.endpoint_summaries
        assert any(
            error.error_details.type == "ServerMetricsCollectionStopped"
            for error in result.error_summary
        )
    finally:
        await manager._accumulator._close_storage()


@pytest.mark.asyncio
async def test_disk_budget_exhaustion_exports_retained_samples_with_failure(
    benchmark_run: BenchmarkRun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real allocation failure must not discard the samples already measured."""
    benchmark_run.cfg.artifacts.dir = tmp_path
    monkeypatch.setattr(Environment.SERVER_METRICS, "STORAGE_MAX_BYTES", 1024**2)
    manager = ServerMetricsManager(run=benchmark_run)
    manager.publish = AsyncMock()
    try:
        for index in range(1100):
            record = metric_record(index)
            record.metrics["histogram"].samples[0].buckets = {
                str(bucket): index for bucket in range(64)
            }
            await manager._on_server_metrics_records([record], "collector")
            if manager._collection_failure is not None:
                break
        assert manager._collection_failure is not None
        assert "disk budget" in manager._collection_failure.message
        await manager.wait_for_tasks()
        await manager._publish_server_metrics_result(
            start_ns=1_100_000_000,
            end_ns=record.timestamp_ns,
            warmup_start_ns=1_000_000_000,
            warmup_end_ns=1_099_000_000,
        )
        result = manager.publish.await_args.args[0].server_metrics_result.results
        assert result.endpoint_summaries
        assert any(
            error.error_details.type == "ServerMetricsCollectionStopped"
            for error in result.error_summary
        )
        assert manager._accumulator._array_store.allocated_bytes <= 1024**2
    finally:
        await manager._accumulator._close_storage()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failed_reservation",
    [
        param(1, id="fetch-timestamps"),
        param(2, id="fetch-latencies"),
        param(3, id="latency-timestamps"),
        param(4, id="unique-update-timestamps"),
    ],
)  # fmt: skip
async def test_fetch_metadata_budget_failure_preserves_retained_export(
    benchmark_run: BenchmarkRun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_reservation: int,
) -> None:
    """Failure growing any fetch buffer must leave metadata and prior samples valid."""
    benchmark_run.cfg.artifacts.dir = tmp_path
    monkeypatch.setattr(Environment.SERVER_METRICS, "STORAGE_MAX_BYTES", 1024**2)
    manager = ServerMetricsManager(run=benchmark_run)
    manager.publish = AsyncMock()
    try:
        for index in range(256):
            record = metric_record(index)
            record.is_duplicate = False
            record.benchmark_phase = CreditPhase.PROFILING
            await manager._on_server_metrics_records([record], "collector")
        store = manager._accumulator._array_store
        endpoint = manager._accumulator.get_hierarchy_for_export().endpoints[
            record.endpoint_url
        ]
        metadata = (
            endpoint._fetch_timestamps_ns,
            endpoint._fetch_latencies_ns,
            endpoint._fetch_latency_timestamps_ns,
            endpoint._unique_update_timestamps,
        )
        before = [array.data.copy() for array in metadata]
        # Each full int64 metadata array grows from 256 to 512 entries.
        # Leave room for exactly the reservations preceding the failing one.
        padding = (
            store.max_bytes - store.allocated_bytes - 4096 * (failed_reservation - 1)
        )
        await run_history_io(store.allocate, padding, np.uint8)

        rejected = metric_record(256)
        rejected.is_duplicate = False
        rejected.benchmark_phase = CreditPhase.PROFILING
        await manager._on_server_metrics_records([rejected], "collector")
        assert manager._collection_failure is not None
        await manager.wait_for_tasks()
        await manager._publish_server_metrics_result(
            start_ns=1_000_000_000,
            end_ns=rejected.timestamp_ns,
            warmup_start_ns=None,
            warmup_end_ns=None,
        )
        result = manager.publish.await_args.args[0].server_metrics_result.results
        assert result.endpoint_summaries
        summary = next(iter(result.endpoint_summaries.values()))
        assert summary.metrics["gauge"].series[0].stats.avg == 127.5
        assert endpoint._total_fetch_count == 256
        assert endpoint._unique_update_count == 256
        assert endpoint.first_fetch_ns == endpoint.first_update_ns == 1_000_000_000
        assert endpoint.last_fetch_ns == endpoint.last_update_ns == record.timestamp_ns
        for array, expected in zip(metadata, before, strict=True):
            np.testing.assert_array_equal(array.data, expected)
        assert len(list(endpoint._fetch_latency_records_ns)) == 256
        assert all(len(entry.data) == 256 for entry in endpoint.metrics.values())
        assert any(
            error.error_details.type == "ServerMetricsCollectionStopped"
            for error in result.error_summary
        )
    finally:
        await manager._accumulator._close_storage()


@pytest.mark.parametrize(
    "time_filter,fetch_count,update_count,latency_ms,median_ms",
    [
        param(None, 5, 3, 0.001002, 1.5, id="all-samples"),
        param(TimeRangeFilter(), 5, 3, 0.001002, 1.5, id="unbounded"),
        param(TimeRangeFilter(1_001_000_000, 1_003_000_000), 3, 2, 0.001002, 1.0, id="inclusive-bounds"),
        param(TimeRangeFilter(start_ns=1_003_000_000), 2, 1, 0.0010035, None, id="start-only"),
        param(TimeRangeFilter(end_ns=1_001_000_000), 2, 1, 0.0010005, None, id="end-only"),
        param(TimeRangeFilter(start_ns=1_010_000_000), 0, 0, 0.0, None, id="empty-window"),
        param(TimeRangeFilter(1_001_500_000, 1_002_500_000), 1, 1, 0.0, None, id="missing-latency"),
    ],
)  # fmt: skip
def test_endpoint_metadata_filters_out_of_order_fetches_and_optional_latencies(
    time_filter: TimeRangeFilter | None,
    fetch_count: int,
    update_count: int,
    latency_ms: float,
    median_ms: float | None,
) -> None:
    """Vectorized metadata preserves inclusive windows and separate latency timestamps."""
    series = ServerMetricsTimeSeries()
    for index in (3, 0, 2, 1, 4):
        record = metric_record(index)
        if index == 2:
            record.endpoint_latency_ns = None
        series.append_snapshot(record)

    info = ServerMetricsAccumulator._compute_endpoint_info(series, time_filter)
    assert info.total_fetches == fetch_count
    assert info.unique_updates == update_count
    assert info.avg_fetch_latency_ms == pytest.approx(latency_ms)
    assert info.median_update_interval_ms == median_ms


@pytest.mark.parametrize("during_growth", [False, True])
def test_filesystem_allocation_failure_preserves_existing_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, during_growth: bool
) -> None:
    """ENOSPC must be caught before a writable sparse extent reaches NumPy."""
    native_linux = array_store.IS_LINUX
    store = ServerMetricsArrayStore(tmp_path, max_bytes=1024**2, flush_bytes=4096)
    first = None
    if during_growth:
        store.open()
        first = store.allocate(512, np.int64)
        first[:] = 42
    allocated = store.allocated_bytes
    monkeypatch.setattr(array_store, "IS_LINUX", True)
    reserve = MagicMock(side_effect=OSError(errno.ENOSPC, "No space left on device"))
    monkeypatch.setattr(os, "posix_fallocate", reserve, raising=False)
    try:
        with pytest.raises(ServerMetricsStorageError, match="No space left"):
            if during_growth:
                store.allocate(1024, np.int64)
            else:
                store.open()
        reserve.assert_called_once()
        assert store.allocated_bytes == allocated
        if first is not None:
            np.testing.assert_array_equal(first, np.full(512, 42))
        else:
            assert not store.initialized
    finally:
        monkeypatch.setattr(array_store, "IS_LINUX", native_linux)
        store.close()


def test_flush_visits_only_recently_written_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tiny write must not scan the retained multi-mapping history."""
    flushed: list[tuple[int, tuple[int, ...]]] = []
    advised: list[tuple[int, tuple[int, ...]]] = []

    class RecordingMapping(mmap.mmap):
        def flush(self, *args: int) -> None:
            flushed.append((len(self), args))
            return super().flush(*args)

        def madvise(self, option: int, *args: int) -> None:
            advised.append((len(self), args))
            return super().madvise(option, *args)

    monkeypatch.setattr(mmap, "mmap", RecordingMapping)
    store = ServerMetricsArrayStore(tmp_path, max_bytes=1024**2, flush_bytes=4096)
    store.open()
    try:
        first = store.allocate(4096, np.uint8)
        first[:] = 1
        later = store.allocate(128 * 1024, np.uint8)
        later[:] = 2
        store.flush()
        flushed.clear()
        advised.clear()
        later[-1] = 3
        store.note_write(later[-1:])
        store.flush()
        assert len(flushed) == 1
        length, (offset, size) = flushed[0]
        assert size <= mmap.ALLOCATIONGRANULARITY
        assert offset > 0
        assert offset + size <= length
        if array_store.IS_LINUX:
            assert advised == flushed
        flushed.clear()
        advised.clear()
        store.flush()
        assert not flushed
        assert not advised
        np.testing.assert_array_equal(first, np.ones(4096, dtype=np.uint8))
        assert later[-1] == 3
    finally:
        store.close()


@pytest.mark.parametrize("unsupported_fallocate", [False, True])
def test_extent_reservation_fallback_writes_before_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unsupported_fallocate: bool
) -> None:
    """Platforms or filesystems without fallocate still allocate initialized bytes."""
    native_linux = array_store.IS_LINUX
    monkeypatch.setattr(array_store, "IS_LINUX", unsupported_fallocate)
    reserve = MagicMock(side_effect=OSError(errno.EOPNOTSUPP, "Not supported"))
    monkeypatch.setattr(os, "posix_fallocate", reserve, raising=False)
    store = ServerMetricsArrayStore(tmp_path, max_bytes=4 * 1024**2, flush_bytes=4096)
    store.open()
    try:
        first = store.allocate(512, np.int64)
        first[:] = 42
        later = store.allocate(2 * 1024**2, np.uint8)
        assert not later.any()
        np.testing.assert_array_equal(first, np.full(512, 42))
        if unsupported_fallocate:
            assert reserve.call_count == 2
        else:
            reserve.assert_not_called()
    finally:
        monkeypatch.setattr(array_store, "IS_LINUX", native_linux)
        store.close()


@pytest.mark.parametrize("series_type", [ScalarTimeSeries, HistogramTimeSeries])
def test_in_order_append_tracks_previous_timestamp_read(
    series_type: type[ScalarTimeSeries] | type[HistogramTimeSeries],
) -> None:
    """The comparison row may occupy a different page from the appended row."""
    accessed: list[np.ndarray] = []
    series = series_type(on_access=accessed.append)
    sample = (
        MetricSample(value=1.0)
        if series_type is ScalarTimeSeries
        else MetricSample(buckets={"1": 1}, sum=1.0, count=1)
    )
    series.append(1, sample)
    accessed.clear()

    series.append(2, sample)

    assert any(np.shares_memory(view, series._timestamps[:1]) for view in accessed)
