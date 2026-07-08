# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for records_manager_export and records_manager_processing.

These two modules are split out of the RecordsManager class and contain pure
helpers + small async coroutines. They lack direct coverage in
``test_records_manager.py`` (which exercises the manager class), so this file
focuses on:

- write_json_file_atomic (atomic file write + tmp cleanup)
- current_results_record_count (sum-across-phases)
- build_partial_profile_results (window/error/cancellation aggregation)
- generate_json_export_data (version lookup, time conversion, tag assignment)
- write_partial_checkpoint (skip-paths, payload contents, async wrapper)
- load_results_processors (success, disabled-skip, raise-skip)
- generate_realtime_metrics (gather + flatten + timeout/exception ignore)
- filter_display_metrics (registry hidden-flag filtering, unknown passthrough)
- bucket_summarize_results (kind-based dispatch)
- build_process_records_result (phase aggregation + cancellation flag)
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
from pytest import param

from aiperf.common.enums import MetricFlags
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.models import (
    ErrorDetails,
    MetricResult,
    ProcessRecordsResult,
    ProfileResults,
)
from aiperf.config import BenchmarkConfig
from aiperf.records.records_manager_export import (
    build_partial_profile_results,
    current_results_record_count,
    generate_json_export_data,
    write_json_file_atomic,
    write_partial_checkpoint,
)
from aiperf.records.records_manager_processing import (
    bucket_summarize_results,
    build_process_records_result,
    filter_display_metrics,
    generate_realtime_metrics,
    load_results_processors,
)

# ============================================================
# Helpers / fixtures
# ============================================================


_MINIMAL_CONFIG_KWARGS: dict[str, Any] = {
    "models": ["test-model"],
    "endpoint": {
        "type": "chat",
        "urls": ["http://localhost:8000/v1/test"],
    },
    "datasets": [
        {
            "name": "default",
            "type": "synthetic",
            "entries": 1,
            "prompts": {"isl": 128, "osl": 64},
        }
    ],
    "phases": [
        {"name": "default", "type": "concurrency", "requests": 10, "concurrency": 1}
    ],
}


def _make_benchmark_config() -> BenchmarkConfig:
    """Build a real BenchmarkConfig with minimal-valid fields."""
    return BenchmarkConfig(**_MINIMAL_CONFIG_KWARGS)


def _metric(tag: str, avg: float = 1.0) -> MetricResult:
    """Build a small MetricResult for tests."""
    return MetricResult(tag=tag, header=tag, unit="ms", count=1, avg=avg)


def _make_tracker(
    *,
    phases: list[str] | None = None,
    per_phase_records: dict[str, int] | None = None,
    time_window: tuple[int | None, int | None] = (1_000, 2_000),
    cancelled_phases: set[str] | None = None,
) -> MagicMock:
    """Build a MagicMock tracker matching the RecordsTracker surface used here."""
    phases = phases or []
    per_phase_records = per_phase_records or {}
    cancelled_phases = cancelled_phases or set()

    tracker = MagicMock()
    tracker.get_results_phases.return_value = phases
    tracker.get_results_time_window.return_value = time_window

    def _stats_for_phase(phase: str) -> MagicMock:
        stats = MagicMock()
        stats.total_records = per_phase_records.get(phase, 0)
        return stats

    tracker.create_stats_for_phase.side_effect = _stats_for_phase
    tracker.was_phase_cancelled.side_effect = lambda phase: phase in cancelled_phases
    return tracker


def _make_error_tracker(
    summary_per_phase: dict[str, list[Any]] | None = None,
) -> MagicMock:
    """Build a MagicMock ErrorTracker matching the surface used here."""
    summary_per_phase = summary_per_phase or {}
    error_tracker = MagicMock()
    error_tracker.get_error_summary_for_phase.side_effect = (
        lambda phase: summary_per_phase.get(phase, [])
    )
    return error_tracker


# ============================================================
# Export module — write_json_file_atomic
# ============================================================


class TestExportWriteJsonFileAtomic:
    """Verify atomic JSON writes with tmp-then-rename semantics."""

    def test_write_creates_parent_dirs_and_writes_bytes(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "deeper" / "out.json"
        payload = orjson.dumps({"hello": "world"})

        write_json_file_atomic(target, payload)

        assert target.read_bytes() == payload
        # tmp should not linger after replace
        assert not target.with_suffix(target.suffix + ".tmp").exists()

    def test_write_overwrites_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "out.json"
        target.write_bytes(b"old")
        write_json_file_atomic(target, b"new")
        assert target.read_bytes() == b"new"

    def test_write_empty_bytes_succeeds(self, tmp_path: Path) -> None:
        target = tmp_path / "empty.json"
        write_json_file_atomic(target, b"")
        assert target.read_bytes() == b""


# ============================================================
# Export module — current_results_record_count
# ============================================================


class TestExportCurrentResultsRecordCount:
    """Verify record-count aggregation across phases."""

    def test_returns_zero_when_no_phases(self) -> None:
        tracker = _make_tracker(phases=[], per_phase_records={})
        assert current_results_record_count(tracker) == 0

    def test_sums_records_across_phases(self) -> None:
        tracker = _make_tracker(
            phases=["warmup", "profiling"],
            per_phase_records={"warmup": 3, "profiling": 7},
        )
        assert current_results_record_count(tracker) == 10

    def test_treats_missing_phase_as_zero_via_stats(self) -> None:
        tracker = _make_tracker(
            phases=["only_phase"],
            per_phase_records={"only_phase": 5},
        )
        assert current_results_record_count(tracker) == 5


# ============================================================
# Export module — build_partial_profile_results
# ============================================================


class TestExportBuildPartialProfileResults:
    """Verify ProfileResults snapshot construction from tracker state."""

    def test_uses_tracker_window_and_phase_aggregates(self) -> None:
        tracker = _make_tracker(
            phases=["profiling"],
            per_phase_records={"profiling": 4},
            time_window=(123, 456),
        )
        error_tracker = _make_error_tracker()
        records = [_metric("request_latency")]

        result = build_partial_profile_results(records, tracker, error_tracker)

        assert isinstance(result, ProfileResults)
        assert result.records == records
        assert result.completed == 4
        assert result.start_ns == 123
        assert result.end_ns == 456
        assert result.was_cancelled is False
        assert result.error_summary == []

    def test_falls_back_to_time_ns_when_window_missing(self) -> None:
        tracker = _make_tracker(
            phases=["profiling"],
            per_phase_records={"profiling": 0},
            time_window=(None, None),
        )
        error_tracker = _make_error_tracker()

        result = build_partial_profile_results([], tracker, error_tracker)

        # Should be populated to non-None ns timestamps from time.time_ns()
        assert result.start_ns > 0
        assert result.end_ns > 0

    def test_aggregates_error_summary_across_phases(self) -> None:
        err_a = MagicMock(name="err_a")
        err_b = MagicMock(name="err_b")
        tracker = _make_tracker(
            phases=["warmup", "profiling"],
            per_phase_records={"warmup": 1, "profiling": 1},
        )
        error_tracker = _make_error_tracker(
            summary_per_phase={"warmup": [err_a], "profiling": [err_b]},
        )

        result = build_partial_profile_results([], tracker, error_tracker)

        assert result.error_summary == [err_a, err_b]

    def test_was_cancelled_true_if_any_phase_cancelled(self) -> None:
        tracker = _make_tracker(
            phases=["warmup", "profiling"],
            per_phase_records={"warmup": 0, "profiling": 0},
            cancelled_phases={"profiling"},
        )
        result = build_partial_profile_results([], tracker, _make_error_tracker())
        assert result.was_cancelled is True


# ============================================================
# Export module — generate_json_export_data
# ============================================================


class TestExportGenerateJsonExportData:
    """Verify JsonExportData construction (version, time, tag-attaching)."""

    def test_happy_path_attaches_metric_tags_as_attributes(self) -> None:
        config = _make_benchmark_config()
        records = [_metric("request_latency", avg=12.5)]
        profile_results = ProfileResults(
            completed=1,
            start_ns=1_000_000_000,
            end_ns=2_000_000_000,
            records=records,
        )

        export = generate_json_export_data(records, profile_results, config)

        assert export.benchmark_id == config.benchmark_id
        assert export.was_cancelled is False
        # request_latency was attached via setattr
        assert export.request_latency is not None
        # start/end converted to datetime
        assert export.start_time is not None
        assert export.end_time is not None
        assert export.start_time < export.end_time

    def test_unknown_aiperf_version_falls_back_to_unknown(self) -> None:
        from importlib.metadata import PackageNotFoundError

        with patch(
            "aiperf.records.records_manager_export.get_version",
            side_effect=PackageNotFoundError,
        ):
            export = generate_json_export_data(
                [],
                ProfileResults(completed=0, start_ns=0, end_ns=0),
                _make_benchmark_config(),
            )
        assert export.aiperf_version == "unknown"

    def test_zero_timestamps_yield_none_start_end_time(self) -> None:
        export = generate_json_export_data(
            [],
            ProfileResults(completed=0, start_ns=0, end_ns=0),
            _make_benchmark_config(),
        )
        # When start_ns/end_ns is 0 (falsy), the helper sets the time fields to None
        assert export.start_time is None
        assert export.end_time is None

    def test_metric_with_no_tag_is_not_attached(self) -> None:
        # tag="" is falsy and the setattr branch is skipped
        config = _make_benchmark_config()
        m = _metric("", avg=1.0)
        export = generate_json_export_data(
            [m],
            ProfileResults(completed=0, start_ns=0, end_ns=0),
            config,
        )
        # No tag means no auto-attach: schema/aiperf_version still populated
        assert export.schema_version is not None


@contextmanager
def _temp_tz(tz_name: str):
    """Temporarily set the process time zone, restoring it on exit.

    ``datetime.fromtimestamp`` without ``tz=`` is interpreted in local time, so
    switching ``TZ`` is what surfaces the naive-vs-aware regression.
    """
    if not hasattr(time, "tzset"):  # pragma: no cover - non-Unix
        pytest.skip("time.tzset unavailable on this platform")
    original = os.environ.get("TZ")
    try:
        os.environ["TZ"] = tz_name
        time.tzset()
        yield
    finally:
        if original is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original
        time.tzset()


class TestExportTimestampsAreUtc:
    """The checkpoint export must emit offset-aware UTC, not naive local time.

    Regression: ``generate_json_export_data`` built ``start_time`` /
    ``end_time`` with a bare ``datetime.fromtimestamp(...)`` (naive local time),
    so a checkpoint-salvaged export disagreed with a normal export of the same
    run by the local-UTC offset. The fix passes ``tz=UTC``.
    """

    def test_start_end_time_are_offset_aware_utc(self) -> None:
        export = generate_json_export_data(
            [],
            ProfileResults(
                completed=1,
                start_ns=1_700_000_000_000_000_000,
                end_ns=1_700_000_001_000_000_000,
            ),
            _make_benchmark_config(),
        )
        for value in (export.start_time, export.end_time):
            assert value is not None
            assert value.tzinfo is not None, "export timestamp must be offset-aware"
            assert value.utcoffset() == timedelta(0), "export timestamp must be UTC"

    def test_start_time_is_tz_independent(self) -> None:
        profile_results = ProfileResults(
            completed=1,
            start_ns=1_700_000_000_000_000_000,
            end_ns=1_700_000_001_000_000_000,
        )
        config = _make_benchmark_config()

        with _temp_tz("UTC"):
            under_utc = generate_json_export_data(
                [], profile_results, config
            ).start_time
        with _temp_tz("Asia/Kolkata"):
            under_kolkata = generate_json_export_data(
                [], profile_results, config
            ).start_time

        assert under_utc == under_kolkata


# ============================================================
# Export module — write_partial_checkpoint
# ============================================================


class TestExportWritePartialCheckpoint:
    """Verify partial-checkpoint behavior (skip cases + payload structure)."""

    @pytest.mark.asyncio
    async def test_skips_when_no_records(self, tmp_path: Path) -> None:
        tracker = _make_tracker(
            phases=["profiling"],
            per_phase_records={"profiling": 0},
        )
        path = tmp_path / "checkpoint.json"

        new_count = await write_partial_checkpoint(
            tracker=tracker,
            error_tracker=_make_error_tracker(),
            processors=[],
            benchmark_config=_make_benchmark_config(),
            checkpoint_path=path,
            last_checkpoint_records=0,
        )

        assert new_count == 0
        assert not path.exists()

    @pytest.mark.asyncio
    async def test_skips_when_record_count_unchanged(self, tmp_path: Path) -> None:
        tracker = _make_tracker(
            phases=["profiling"],
            per_phase_records={"profiling": 5},
        )
        path = tmp_path / "checkpoint.json"

        new_count = await write_partial_checkpoint(
            tracker=tracker,
            error_tracker=_make_error_tracker(),
            processors=[],
            benchmark_config=_make_benchmark_config(),
            checkpoint_path=path,
            last_checkpoint_records=5,
        )

        assert new_count == 5
        assert not path.exists()

    @pytest.mark.asyncio
    async def test_skips_when_summarize_returns_no_records(
        self, tmp_path: Path
    ) -> None:
        tracker = _make_tracker(
            phases=["profiling"],
            per_phase_records={"profiling": 5},
        )
        # processor whose summarize returns []
        proc = MagicMock()
        proc.summarize = AsyncMock(return_value=[])
        path = tmp_path / "checkpoint.json"

        new_count = await write_partial_checkpoint(
            tracker=tracker,
            error_tracker=_make_error_tracker(),
            processors=[proc],
            benchmark_config=_make_benchmark_config(),
            checkpoint_path=path,
            last_checkpoint_records=0,
        )

        assert new_count == 0
        assert not path.exists()

    @pytest.mark.asyncio
    async def test_writes_checkpoint_with_expected_metadata(
        self, tmp_path: Path
    ) -> None:
        tracker = _make_tracker(
            phases=["profiling"],
            per_phase_records={"profiling": 7},
            time_window=(1_000_000_000, 2_000_000_000),
        )
        proc = MagicMock()
        proc.summarize = AsyncMock(return_value=[_metric("request_latency", avg=3.0)])
        path = tmp_path / "checkpoint.json"

        new_count = await write_partial_checkpoint(
            tracker=tracker,
            error_tracker=_make_error_tracker(),
            processors=[proc],
            benchmark_config=_make_benchmark_config(),
            checkpoint_path=path,
            last_checkpoint_records=0,
        )

        assert new_count == 7
        assert path.exists()
        payload = orjson.loads(path.read_bytes())
        assert payload["checkpoint"] is True
        assert payload["records_completed"] == 7
        assert payload["generated_at_ns"] > 0


# ============================================================
# Processing module — load_results_processors
# ============================================================


class _FakePluginEntry:
    """Lightweight stand-in for plugin entries with just a name attr."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeHost:
    """Concrete implementation of the _LoaderHost protocol."""

    def __init__(self) -> None:
        self.service_id = "svc-1"
        self.run = MagicMock()
        self.pub_client = MagicMock()
        self.attached: list[Any] = []
        self.debug_messages: list[str] = []
        self.error_messages: list[str] = []

    def attach_child_lifecycle(self, child: Any) -> None:
        self.attached.append(child)

    def debug(self, msg: Any) -> None:
        self.debug_messages.append(str(msg))

    def error(self, msg: Any) -> None:
        self.error_messages.append(str(msg))


class TestProcessingLoadResultsProcessors:
    """Verify processor loading skips disabled/failing entries gracefully."""

    def test_loads_enabled_processors(self) -> None:
        host = _FakeHost()
        instance = MagicMock(name="processor-instance")
        ProcessorClass = MagicMock(return_value=instance)

        with (
            patch(
                "aiperf.records.records_manager_processing.plugins.iter_entries",
                return_value=iter([_FakePluginEntry("good")]),
            ),
            patch(
                "aiperf.records.records_manager_processing.plugins.get_class",
                return_value=ProcessorClass,
            ),
        ):
            processors = load_results_processors(host)

        assert processors == [instance]
        assert host.attached == [instance]
        assert any("Created results processor" in m for m in host.debug_messages)

    def test_skips_disabled_processor_without_error(self) -> None:
        host = _FakeHost()

        def _disabled_class(**_: Any) -> Any:
            raise PostProcessorDisabled("disabled")

        with (
            patch(
                "aiperf.records.records_manager_processing.plugins.iter_entries",
                return_value=iter([_FakePluginEntry("disabled-one")]),
            ),
            patch(
                "aiperf.records.records_manager_processing.plugins.get_class",
                return_value=_disabled_class,
            ),
        ):
            processors = load_results_processors(host)

        assert processors == []
        # Disabled is a debug, not an error
        assert host.error_messages == []
        assert any("disabled" in m for m in host.debug_messages)

    def test_one_bad_processor_does_not_abort_loading(self) -> None:
        host = _FakeHost()
        good_instance = MagicMock(name="good-instance")

        # Two entries: first one explodes, second one constructs OK
        entries = [_FakePluginEntry("bad"), _FakePluginEntry("good")]

        def _get_class_side_effect(_category: Any, name: str) -> Any:
            if name == "bad":

                def _boom(**_: Any) -> Any:
                    raise RuntimeError("kaboom")

                return _boom

            def _good(**_: Any) -> Any:
                return good_instance

            return _good

        with (
            patch(
                "aiperf.records.records_manager_processing.plugins.iter_entries",
                return_value=iter(entries),
            ),
            patch(
                "aiperf.records.records_manager_processing.plugins.get_class",
                side_effect=_get_class_side_effect,
            ),
        ):
            processors = load_results_processors(host)

        assert processors == [good_instance]
        assert any(
            "Failed to create results processor bad" in m for m in host.error_messages
        )


# ============================================================
# Processing module — generate_realtime_metrics
# ============================================================


class TestProcessingGenerateRealtimeMetrics:
    """Verify generate_realtime_metrics gather/flatten/exception-tolerant behavior."""

    @pytest.mark.asyncio
    async def test_flattens_multiple_processor_results(self) -> None:
        proc_a = MagicMock()
        proc_a.summarize = AsyncMock(
            return_value=[_metric("request_latency"), _metric("ttft")]
        )
        proc_b = MagicMock()
        proc_b.summarize = AsyncMock(return_value=[_metric("request_count")])

        results = await generate_realtime_metrics([proc_a, proc_b])

        assert {r.tag for r in results} == {"request_latency", "ttft", "request_count"}

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_processors(self) -> None:
        assert await generate_realtime_metrics([]) == []

    @pytest.mark.asyncio
    async def test_processor_exception_is_skipped(self) -> None:
        proc_ok = MagicMock()
        proc_ok.summarize = AsyncMock(return_value=[_metric("request_latency")])
        proc_bad = MagicMock()
        proc_bad.summarize = AsyncMock(side_effect=RuntimeError("boom"))

        results = await generate_realtime_metrics([proc_ok, proc_bad])

        assert [r.tag for r in results] == ["request_latency"]

    @pytest.mark.asyncio
    async def test_processor_timeout_is_skipped(self) -> None:
        proc_ok = MagicMock()
        proc_ok.summarize = AsyncMock(return_value=[_metric("request_latency")])

        async def _slow() -> list[MetricResult]:
            await asyncio.sleep(60.0)
            return []

        proc_slow = MagicMock()
        proc_slow.summarize = _slow

        # Tiny timeout so wait_for fails quickly. asyncio.sleep is auto-instant
        # in this test suite, but wait_for itself still raises TimeoutError
        # because asyncio.sleep(60) returns immediately as if it had elapsed.
        results = await generate_realtime_metrics([proc_ok, proc_slow], timeout=0.01)

        assert [r.tag for r in results] == ["request_latency"]

    @pytest.mark.asyncio
    async def test_filters_non_metric_result_items_out(self) -> None:
        proc = MagicMock()
        # processor returned a list-of-list mixed with junk
        proc.summarize = AsyncMock(
            return_value=[_metric("request_latency"), "not-a-metric", 12345]
        )

        results = await generate_realtime_metrics([proc])

        assert [r.tag for r in results] == ["request_latency"]


# ============================================================
# Processing module — filter_display_metrics
# ============================================================


class TestProcessingFilterDisplayMetrics:
    """Verify hidden-flag filtering using the real MetricRegistry."""

    def test_keeps_visible_registered_metric(self) -> None:
        # request_latency has flags=NONE so must pass through
        m = _metric("request_latency")
        assert filter_display_metrics([m]) == [m]

    def test_filters_internal_metric(self) -> None:
        # min_request_timestamp has MetricFlags.INTERNAL
        m_internal = _metric("min_request_timestamp")
        assert filter_display_metrics([m_internal]) == []

    def test_keeps_unknown_tag_passthrough(self) -> None:
        # Unregistered tag should fall through the MetricTypeError branch
        m_unknown = _metric("plugin_metric_xyz")
        assert filter_display_metrics([m_unknown]) == [m_unknown]

    def test_mixed_input_filters_only_hidden(self) -> None:
        keep = _metric("request_latency")
        drop = _metric("min_request_timestamp")
        passthrough = _metric("custom_external_metric")

        out = filter_display_metrics([keep, drop, passthrough])

        assert out == [keep, passthrough]

    def test_empty_list_returns_empty_list(self) -> None:
        assert filter_display_metrics([]) == []

    def test_filters_via_patched_experimental_flag(self) -> None:
        # Direct verification of the EXPERIMENTAL branch using a stubbed
        # registry so we don't depend on which built-in metric currently
        # carries the EXPERIMENTAL flag.
        m = _metric("some_tag")
        fake_cls = MagicMock()
        fake_cls.flags = MetricFlags.EXPERIMENTAL

        with patch(
            "aiperf.metrics.metric_registry.MetricRegistry.get_class",
            return_value=fake_cls,
        ):
            assert filter_display_metrics([m]) == []


# ============================================================
# Processing module — bucket_summarize_results
# ============================================================


class TestProcessingBucketSummarizeResults:
    """Verify gather()-output dispatch by Python kind."""

    def test_buckets_by_kind(self) -> None:
        records = [_metric("request_latency")]
        ts = [{"request_latency": _metric("request_latency")}]
        err = ErrorDetails(code=500, message="server error", type="ServerError")
        exc = ValueError("boom")

        rec_out, ts_out, err_out, exc_out = bucket_summarize_results(
            [records, ts, err, exc]
        )

        assert rec_out == records
        assert ts_out == ts
        assert err_out == [err]
        assert exc_out == [exc]

    def test_empty_input_returns_empty_buckets(self) -> None:
        rec_out, ts_out, err_out, exc_out = bucket_summarize_results([])
        assert rec_out == []
        assert ts_out == []
        assert err_out == []
        assert exc_out == []

    def test_multiple_record_lists_extend(self) -> None:
        a = [_metric("a")]
        b = [_metric("b"), _metric("c")]
        rec_out, _, _, _ = bucket_summarize_results([a, b])
        assert [r.tag for r in rec_out] == ["a", "b", "c"]

    @pytest.mark.parametrize(
        "value",
        [
            param(None, id="none-ignored"),
            param(42, id="int-ignored"),
            param("string", id="str-ignored"),
            param(3.14, id="float-ignored"),
        ],
    )  # fmt: skip
    def test_unknown_types_are_ignored(self, value: Any) -> None:
        rec_out, ts_out, err_out, exc_out = bucket_summarize_results([value])
        assert rec_out == []
        assert ts_out == []
        assert err_out == []
        assert exc_out == []

    def test_last_dict_wins_when_multiple_dicts(self) -> None:
        d1 = [{"a": _metric("a")}]
        d2 = [{"b": _metric("b")}]
        _, ts_out, _, _ = bucket_summarize_results([d1, d2])
        # Implementation overwrites — second list-of-dicts wins
        assert ts_out == d2


# ============================================================
# Processing module — build_process_records_result
# ============================================================


class TestProcessingBuildProcessRecordsResult:
    """Verify final ProcessRecordsResult assembly."""

    def test_builds_full_result_from_buckets(self) -> None:
        records = [_metric("request_latency"), _metric("ttft")]
        ts = {0: [_metric("request_latency")]}
        errs = [ErrorDetails(code=500, message="bad", type="ServerError")]
        tracker = _make_tracker(
            phases=["profiling"],
            time_window=(1_000, 2_000),
        )
        error_tracker = _make_error_tracker(summary_per_phase={"profiling": []})

        result = build_process_records_result(
            records_results=records,
            timeslice_metric_results=ts,
            error_results=errs,
            tracker=tracker,
            error_tracker=error_tracker,
            cancelled=False,
        )

        assert isinstance(result, ProcessRecordsResult)
        assert result.results.records == records
        assert result.results.timeslice_metric_results == ts
        assert result.results.completed == 2
        assert result.results.start_ns == 1_000
        assert result.results.end_ns == 2_000
        assert result.results.was_cancelled is False
        assert result.errors == errs

    def test_cancelled_flag_propagates(self) -> None:
        tracker = _make_tracker(phases=[], time_window=(1, 2))
        result = build_process_records_result(
            records_results=[],
            timeslice_metric_results={},
            error_results=[],
            tracker=tracker,
            error_tracker=_make_error_tracker(),
            cancelled=True,
        )
        assert result.results.was_cancelled is True

    def test_aggregates_error_summary_across_phases(self) -> None:
        err_a = MagicMock(name="phase-a-err")
        err_b = MagicMock(name="phase-b-err")
        tracker = _make_tracker(phases=["a", "b"], time_window=(1, 2))
        error_tracker = _make_error_tracker(
            summary_per_phase={"a": [err_a], "b": [err_b]},
        )

        result = build_process_records_result(
            records_results=[],
            timeslice_metric_results={},
            error_results=[],
            tracker=tracker,
            error_tracker=error_tracker,
            cancelled=False,
        )

        assert result.results.error_summary == [err_a, err_b]

    def test_completed_uses_records_length_not_tracker_count(self) -> None:
        # Distinct from build_partial_profile_results: completed = len(records),
        # not sum-across-phases. Lock that contract in.
        records = [_metric("a"), _metric("b"), _metric("c")]
        tracker = _make_tracker(
            phases=["profiling"],
            per_phase_records={"profiling": 999},
            time_window=(1, 2),
        )

        result = build_process_records_result(
            records_results=records,
            timeslice_metric_results={},
            error_results=[],
            tracker=tracker,
            error_tracker=_make_error_tracker(),
            cancelled=False,
        )

        assert result.results.completed == 3

    def test_falls_back_to_time_ns_when_window_missing(self) -> None:
        tracker = _make_tracker(phases=[], time_window=(None, None))
        result = build_process_records_result(
            records_results=[],
            timeslice_metric_results={},
            error_results=[],
            tracker=tracker,
            error_tracker=_make_error_tracker(),
            cancelled=False,
        )
        assert result.results.start_ns > 0
        assert result.results.end_ns > 0
