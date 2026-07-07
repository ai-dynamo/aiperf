# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the sweep-line profile_export analysis tool.

Focuses on pure-logic surface:
- Metric / timestamp extraction helpers (``get_metric_ms``, lifecycle-start
  reconstruction, first-response derivation).
- ``build_events`` event-tuple ordering and contents.
- ``sweep`` state-machine: state transitions for each ``EventType``,
  pre/post-TTFT bookkeeping, peak tracking, error counting.
- ``detect_start_blocks`` gap-bridging behavior.
- ``build_expected_windows`` / ``build_wave_completion_windows`` derived
  from sweep output.
- ``load_records`` JSONL reading.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import orjson
import pytest
from pytest import param

from tools.analyze_profile_export import (
    EventType,
    ExpectedWindow,
    SweepState,
    TimeSeries,
    WaveCompletionWindow,
    build_events,
    build_expected_windows,
    build_wave_completion_windows,
    detect_start_blocks,
    get_first_response_ns,
    get_metric_ms,
    get_request_lifecycle_start_ns,
    get_request_send_start_ns,
    load_records,
    sweep,
)

# ============================================================
# Fixtures / helpers
# ============================================================


def _record(
    *,
    issued_ns: int | None = None,
    received_ns: int | None = None,
    request_start_ns: int | None = None,
    request_ack_ns: int | None = None,
    request_end_ns: int | None = None,
    clock_offset_ns: int | None = None,
    worker_id: str = "worker-0",
    metrics: dict[str, dict] | None = None,
    error: dict | None = None,
) -> dict:
    """Build a record dict matching the shape of profile_export.jsonl entries."""
    metadata: dict = {"worker_id": worker_id}
    if issued_ns is not None:
        metadata["credit_issued_ns"] = issued_ns
    if received_ns is not None:
        metadata["credit_received_ns"] = received_ns
    if request_start_ns is not None:
        metadata["request_start_ns"] = request_start_ns
    if request_ack_ns is not None:
        metadata["request_ack_ns"] = request_ack_ns
    if request_end_ns is not None:
        metadata["request_end_ns"] = request_end_ns
    if clock_offset_ns is not None:
        metadata["clock_offset_ns"] = clock_offset_ns
    rec: dict = {"metadata": metadata, "metrics": metrics or {}}
    if error is not None:
        rec["error"] = error
    else:
        rec["error"] = None
    return rec


# ============================================================
# get_metric_ms
# ============================================================


class TestGetMetricMs:
    def test_get_metric_ms_seconds_unit_converts_to_ms(self) -> None:
        rec = {"metrics": {"k": {"value": 1.5, "unit": "s"}}}
        assert get_metric_ms(rec, "k") == 1500.0

    def test_get_metric_ms_other_unit_passes_through(self) -> None:
        rec = {"metrics": {"k": {"value": 42.0, "unit": "ms"}}}
        assert get_metric_ms(rec, "k") == 42.0

    def test_get_metric_ms_no_unit_assumes_already_ms(self) -> None:
        rec = {"metrics": {"k": {"value": 17.0}}}
        assert get_metric_ms(rec, "k") == 17.0

    def test_get_metric_ms_missing_metric_returns_none(self) -> None:
        assert get_metric_ms({"metrics": {}}, "k") is None

    def test_get_metric_ms_missing_metrics_block_returns_none(self) -> None:
        assert get_metric_ms({}, "k") is None

    def test_get_metric_ms_value_none_returns_none(self) -> None:
        rec = {"metrics": {"k": {"value": None, "unit": "ms"}}}
        assert get_metric_ms(rec, "k") is None

    def test_get_metric_ms_falsy_metric_returns_none(self) -> None:
        # Empty dict for the metric short-circuits the falsy guard.
        rec = {"metrics": {"k": {}}}
        assert get_metric_ms(rec, "k") is None


# ============================================================
# Timestamp reconstruction
# ============================================================


class TestRequestLifecycleStartNs:
    def test_lifecycle_start_uses_http_total_when_present(self) -> None:
        rec = _record(
            request_start_ns=1_000_000_000,
            request_end_ns=2_000_000_000,
            metrics={"http_req_total": {"value": 500.0, "unit": "ms"}},
        )
        # 2_000_000_000 - 500ms = 1_500_000_000
        assert get_request_lifecycle_start_ns(rec) == 1_500_000_000

    def test_lifecycle_start_falls_back_to_request_start(self) -> None:
        rec = _record(request_start_ns=999, request_end_ns=2_000_000_000)
        assert get_request_lifecycle_start_ns(rec) == 999

    def test_lifecycle_start_no_request_end_falls_back(self) -> None:
        rec = _record(
            request_start_ns=42,
            metrics={"http_req_total": {"value": 1.0, "unit": "ms"}},
        )
        # No request_end_ns: cannot reconstruct, must use request_start_ns.
        assert get_request_lifecycle_start_ns(rec) == 42

    def test_lifecycle_start_missing_everything_returns_none(self) -> None:
        rec = _record()
        assert get_request_lifecycle_start_ns(rec) is None


class TestRequestSendStartNs:
    def test_send_start_uses_http_duration_when_present(self) -> None:
        rec = _record(
            request_start_ns=1,
            request_end_ns=3_000_000_000,
            metrics={"http_req_duration": {"value": 200.0, "unit": "ms"}},
        )
        assert get_request_send_start_ns(rec) == 3_000_000_000 - 200_000_000

    def test_send_start_falls_back_to_request_start(self) -> None:
        rec = _record(request_start_ns=7)
        assert get_request_send_start_ns(rec) == 7


class TestFirstResponseNs:
    def test_first_response_prefers_request_ack_ns(self) -> None:
        rec = _record(
            request_start_ns=1,
            request_end_ns=10,
            request_ack_ns=5,
            metrics={
                "http_req_receiving": {"value": 100.0, "unit": "ms"},
                "time_to_first_token": {"value": 50.0, "unit": "ms"},
            },
        )
        assert get_first_response_ns(rec) == 5

    def test_first_response_uses_http_receiving_when_no_ack(self) -> None:
        rec = _record(
            request_end_ns=2_000_000_000,
            metrics={"http_req_receiving": {"value": 100.0, "unit": "ms"}},
        )
        assert get_first_response_ns(rec) == 2_000_000_000 - 100_000_000

    def test_first_response_uses_ttft_fallback(self) -> None:
        rec = _record(
            request_start_ns=1_000_000_000,
            metrics={"time_to_first_token": {"value": 50.0, "unit": "ms"}},
        )
        # request_start + 50ms
        assert get_first_response_ns(rec) == 1_050_000_000

    def test_first_response_no_signals_returns_none(self) -> None:
        assert get_first_response_ns(_record()) is None


# ============================================================
# load_records
# ============================================================


class TestLoadRecords:
    def test_load_records_reads_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "p.jsonl"
        recs = [{"metadata": {"i": 1}}, {"metadata": {"i": 2}}]
        path.write_bytes(b"\n".join(orjson.dumps(r) for r in recs) + b"\n")

        loaded = load_records(path)

        assert loaded == recs

    def test_load_records_empty_file_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_bytes(b"")
        assert load_records(path) == []


# ============================================================
# build_events
# ============================================================


class TestBuildEvents:
    def test_build_events_emits_all_event_types_for_full_record(self) -> None:
        rec = _record(
            issued_ns=1000,
            received_ns=2000,
            request_start_ns=3000,
            request_ack_ns=4000,
            request_end_ns=5000,
        )
        events = build_events([rec])

        # Five events: issued, received, start, first_response, end.
        types = [e[1] for e in events]
        assert types == [
            EventType.CREDIT_ISSUED,
            EventType.CREDIT_RECEIVED,
            EventType.REQUEST_START,
            EventType.FIRST_RESPONSE,
            EventType.REQUEST_END,
        ]

    def test_build_events_skips_missing_timestamps(self) -> None:
        # Only request_end_ns present; nothing else.
        rec = _record(request_end_ns=999)
        events = build_events([rec])
        assert [e[1] for e in events] == [EventType.REQUEST_END]

    def test_build_events_orders_by_timestamp_then_event_type(self) -> None:
        # Two records with overlapping timestamps; ensure stable sort by (ts, type).
        rec_a = _record(issued_ns=10, received_ns=20)
        rec_b = _record(issued_ns=10, received_ns=15)
        events = build_events([rec_a, rec_b])
        # At t=10 we have two CREDIT_ISSUED events; their order is preserved by record.
        assert events[0][:2] == (10, EventType.CREDIT_ISSUED)
        assert events[1][:2] == (10, EventType.CREDIT_ISSUED)
        # At t=15 a CREDIT_RECEIVED comes before t=20.
        assert events[2][:2] == (15, EventType.CREDIT_RECEIVED)
        assert events[3][:2] == (20, EventType.CREDIT_RECEIVED)

    def test_build_events_marks_errors_and_extracts_type(self) -> None:
        rec = _record(
            request_end_ns=100,
            error={"type": "timeout", "message": "x"},
        )
        events = build_events([rec])
        ts, etype, idx, is_error, worker, error_type = events[0]
        assert is_error is True
        assert error_type == "timeout"
        assert worker == "worker-0"
        assert idx == 0

    def test_build_events_non_error_record_has_none_error_type(self) -> None:
        rec = _record(request_end_ns=100)
        events = build_events([rec])
        ts, etype, idx, is_error, worker, error_type = events[0]
        assert is_error is False
        assert error_type is None


# ============================================================
# sweep state machine
# ============================================================


class TestSweep:
    def test_sweep_credit_issued_increments_counters(self) -> None:
        rec = _record(issued_ns=1_000_000_000, received_ns=2_000_000_000)
        events = build_events([rec])
        state, _ts, _t0 = sweep(events, bucket_ns=1_000_000_000)
        assert state.credits_issued == 1
        assert state.credits_received == 1
        assert state.credits_pending == 0
        assert state.peak_credits_pending == 1

    def test_sweep_full_lifecycle_balances_in_flight(self) -> None:
        rec = _record(
            issued_ns=1_000_000_000,
            received_ns=1_500_000_000,
            request_start_ns=2_000_000_000,
            request_ack_ns=3_000_000_000,
            request_end_ns=4_000_000_000,
        )
        events = build_events([rec])
        state, _ts, _t0 = sweep(events, bucket_ns=1_000_000_000)
        # After full lifecycle, in_flight, pre_ttft, post_ttft all zero.
        assert state.in_flight == 0
        assert state.pre_ttft == 0
        assert state.post_ttft == 0
        assert state.requests_started == 1
        assert state.requests_acked == 1
        assert state.requests_completed == 1
        assert state.peak_in_flight == 1
        assert state.peak_pre_ttft == 1

    def test_sweep_request_end_without_ack_decrements_pre_ttft(self) -> None:
        # Failure path: request ends before any first response observed.
        rec = _record(
            request_start_ns=1_000_000_000,
            request_end_ns=2_000_000_000,
            error={"type": "boom", "message": ""},
        )
        events = build_events([rec])
        state, _ts, _t0 = sweep(events, bucket_ns=1_000_000_000)
        assert state.pre_ttft == 0
        assert state.post_ttft == 0
        assert state.in_flight == 0
        assert state.errors_total == 1

    def test_sweep_records_error_count_in_time_series(self) -> None:
        rec = _record(
            request_start_ns=1_000_000_000,
            request_end_ns=2_500_000_000,
            error={"type": "timeout", "message": ""},
        )
        events = build_events([rec])
        state, ts, _t0 = sweep(events, bucket_ns=1_000_000_000)
        assert state.errors_total == 1
        # The request_end falls in bucket index 1 (relative to t0 = 1_000_000_000).
        assert ts.errors[1] == 1.0

    def test_sweep_concurrent_requests_track_peak(self) -> None:
        # Two requests overlapping in time: peak in_flight should be 2.
        rec_a = _record(
            request_start_ns=1_000_000_000,
            request_ack_ns=1_500_000_000,
            request_end_ns=4_000_000_000,
        )
        rec_b = _record(
            request_start_ns=2_000_000_000,
            request_ack_ns=2_500_000_000,
            request_end_ns=3_000_000_000,
        )
        events = build_events([rec_a, rec_b])
        state, _ts, _t0 = sweep(events, bucket_ns=1_000_000_000)
        assert state.peak_in_flight == 2
        assert state.in_flight == 0


# ============================================================
# detect_start_blocks
# ============================================================


def _ts_from_starts(starts: list[float]) -> TimeSeries:
    ts = TimeSeries(len(starts))
    ts.starts = np.array(starts, dtype=float)
    return ts


class TestDetectStartBlocks:
    def test_detect_start_blocks_single_contiguous_block(self) -> None:
        ts = _ts_from_starts([0.0, 1.0, 2.0, 3.0, 0.0, 0.0])
        # Activity at indices 1, 2, 3 (above min_rate=1.0).
        assert detect_start_blocks(ts) == [(1, 3)]

    def test_detect_start_blocks_no_activity_returns_empty(self) -> None:
        ts = _ts_from_starts([0.0, 0.0, 0.0])
        assert detect_start_blocks(ts) == []

    def test_detect_start_blocks_bridges_small_gaps(self) -> None:
        # Default gap_s=10 should bridge a 5-bucket gap.
        ts = _ts_from_starts([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        assert detect_start_blocks(ts) == [(0, 6)]

    def test_detect_start_blocks_does_not_bridge_large_gap(self) -> None:
        starts = [1.0] + [0.0] * 15 + [1.0]  # 15-bucket gap > gap_s=10
        ts = _ts_from_starts(starts)
        assert detect_start_blocks(ts) == [(0, 0), (16, 16)]


# ============================================================
# build_expected_windows
# ============================================================


class TestBuildExpectedWindows:
    def test_build_expected_windows_no_latency_returns_empty(self) -> None:
        ts = _ts_from_starts([1.0, 1.0])
        assert build_expected_windows(ts, expected_latency_s=None) == []

    def test_build_expected_windows_shifts_blocks_by_latency(self) -> None:
        ts = _ts_from_starts([0.0, 1.0, 1.0, 0.0])
        windows = build_expected_windows(ts, expected_latency_s=2.5)
        assert windows == [
            ExpectedWindow(start_s=1, end_s=2, expected_start_s=3.5, expected_end_s=4.5)
        ]


# ============================================================
# build_wave_completion_windows
# ============================================================


class TestBuildWaveCompletionWindows:
    def test_wave_completion_windows_finds_first_end_for_each_block(self) -> None:
        # One record: lifecycle_start at t0+0, end at t0+4. start_block must
        # cover bucket 0 to match. We pass t0 explicitly so the bucket math
        # is deterministic.
        rec = _record(
            request_start_ns=1_000_000_000,
            request_end_ns=5_000_000_000,
        )
        # Build a TimeSeries with starts activity at bucket 0 only.
        ts = _ts_from_starts([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        windows = build_wave_completion_windows(ts, [rec], t0=1_000_000_000)
        assert len(windows) == 1
        assert windows[0].first_end_s == 4

    def test_wave_completion_windows_no_matching_completion(self) -> None:
        # Record's lifecycle_start bucket (10) lies outside the start block (0..0),
        # so the block reports no observed completion.
        rec = _record(
            request_start_ns=10_000_000_000,
            request_end_ns=11_000_000_000,
        )
        ts = _ts_from_starts([1.0, 0.0, 0.0])
        windows = build_wave_completion_windows(ts, [rec], t0=0)
        assert len(windows) == 1
        assert windows[0].first_end_s is None


# ============================================================
# Dataclass invariants
# ============================================================


class TestDataclasses:
    @pytest.mark.parametrize(
        "field_name",
        [
            "in_flight",
            "pre_ttft",
            "post_ttft",
            "credits_pending",
            "starts",
            "ends",
            "errors",
            "credits_issued",
        ],
    )  # fmt: skip
    def test_time_series_initializes_zero_arrays(self, field_name: str) -> None:
        ts = TimeSeries(n=4)
        arr = getattr(ts, field_name)
        assert arr.shape == (4,)
        assert np.all(arr == 0)

    def test_sweep_state_defaults_zero(self) -> None:
        state = SweepState()
        assert state.in_flight == 0
        assert state.peak_in_flight == 0
        assert state.errors_total == 0

    @pytest.mark.parametrize(
        "first_end",
        [
            param(None, id="no-completion"),
            param(42, id="completion-at-42s"),
        ],
    )  # fmt: skip
    def test_wave_completion_window_holds_first_end(
        self, first_end: int | None
    ) -> None:
        win = WaveCompletionWindow(start_s=0, end_s=10, first_end_s=first_end)
        assert win.first_end_s == first_end
