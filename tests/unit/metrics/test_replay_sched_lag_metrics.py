# SPDX-FileCopyrightText: Copyright (c) 2026 Baseten.co. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the replay schedule send-lag metrics.

Record-metric tests drive synthetic records that carry an intended schedule
timestamp on the dispatched turn (``Turn.timestamp``, ms) and an actual send
wall time (``RequestRecord.timestamp_ns``), exactly as fixed-schedule replay
produces.

The ``replay_sched_lag_*`` family is injected post-aggregation from the
``replay_send_schedule_offset`` column (see ``inject_replay_sched_lag_metrics``),
so the derived-metric tests build a :class:`ColumnStore` of offsets and assert on
the injected :class:`MetricResult` values. A golden-parity test pins the injected
values to the exact legacy formula (anchored-percentile over the offsets) that
the deleted ``MetricResultsProcessor`` used, so the columnar port is provably
byte-for-byte equivalent to legacy aiperf for identical offsets.
"""

import numpy as np
import pytest
from pytest import param

from aiperf.common.constants import NANOS_PER_MILLIS
from aiperf.common.enums import MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import MetricResult, ParsedResponseRecord, Turn
from aiperf.metrics.column_store import ColumnStore
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.replay_sched_lag_analyzer import inject_replay_sched_lag_metrics
from aiperf.metrics.types.replay_sched_lag_metrics import (
    REPLAY_SCHED_DEGRADED_THRESHOLD_MS,
    ReplaySchedDegradedMetric,
    ReplaySchedLagP50Metric,
    ReplaySchedLagP90Metric,
    ReplaySchedLagP99Metric,
    ReplaySendScheduleOffsetMetric,
)
from tests.unit.metrics.conftest import create_record, run_simple_metrics_pipeline


def _scheduled_record(
    intended_ms: float | None, actual_send_ms: float
) -> ParsedResponseRecord:
    """Record whose turn was scheduled at ``intended_ms`` (schedule-relative)
    and actually sent at ``actual_send_ms`` (wall clock, same test clock)."""
    record = create_record(start_ns=int(actual_send_ms * NANOS_PER_MILLIS))
    record.request.request_info.turns = [Turn(timestamp=intended_ms)]
    return record


def _store_with_offsets(offsets_ns: list[int]) -> ColumnStore:
    """Build a ColumnStore whose only column is ``replay_send_schedule_offset``,
    exactly as the accumulator ingests it during a fixed-schedule replay."""
    store = ColumnStore(initial_capacity=max(len(offsets_ns), 1))
    for idx, offset in enumerate(offsets_ns):
        store.ingest(
            idx=idx,
            record_metrics={ReplaySendScheduleOffsetMetric.tag: offset},
            start_ns=float(idx),
            end_ns=float(idx),
            generation_start_ns=None,
        )
    return store


def _inject(
    offsets_ms: list[float],
    *,
    mask: np.ndarray | None = None,
    warn=None,
) -> dict[str, MetricResult]:
    """Run the injection over a store built from ``offsets_ms`` (ms → ns)."""
    store = _store_with_offsets([int(ms * NANOS_PER_MILLIS) for ms in offsets_ms])
    results: dict[str, MetricResult] = {}
    inject_replay_sched_lag_metrics(store, results, mask=mask, warn_degraded=warn)
    return results


def _legacy_replay_lag(offsets_ns: np.ndarray) -> dict[str, float]:
    """The exact computation legacy aiperf's ``MetricResultsProcessor`` used
    (``_anchored_lag_ms`` + ``np.percentile`` + degraded threshold). Serves as
    the golden reference for the columnar injection port."""
    data = np.asarray(offsets_ns, dtype=np.float64)
    anchored = (data - data.min()) / NANOS_PER_MILLIS
    p50 = float(np.percentile(anchored, 50.0))
    p90 = float(np.percentile(anchored, 90.0))
    p99 = float(np.percentile(anchored, 99.0))
    return {
        ReplaySchedLagP50Metric.tag: p50,
        ReplaySchedLagP90Metric.tag: p90,
        ReplaySchedLagP99Metric.tag: p99,
        ReplaySchedDegradedMetric.tag: float(
            int(p99 > REPLAY_SCHED_DEGRADED_THRESHOLD_MS)
        ),
    }


def test_family_is_fixed_schedule_only() -> None:
    # Turn timestamps reach records in every timing mode, so applicability
    # must be gated on the run's timing mode, not on timestamp presence.
    for metric_cls in (
        ReplaySendScheduleOffsetMetric,
        ReplaySchedLagP50Metric,
        ReplaySchedLagP90Metric,
        ReplaySchedLagP99Metric,
        ReplaySchedDegradedMetric,
    ):
        assert metric_cls.has_flags(MetricFlags.FIXED_SCHEDULE_ONLY), (
            f"{metric_cls.__name__} must be FIXED_SCHEDULE_ONLY"
        )


class TestReplaySendScheduleOffsetMetric:
    def test_parse_record_returns_actual_minus_intended_ns(self) -> None:
        records = [_scheduled_record(intended_ms=2, actual_send_ms=5)]

        results = run_simple_metrics_pipeline(
            records, ReplaySendScheduleOffsetMetric.tag
        )

        assert results[ReplaySendScheduleOffsetMetric.tag] == [3 * NANOS_PER_MILLIS]

    def test_parse_record_no_turn_timestamp_yields_no_value(self) -> None:
        records = [_scheduled_record(intended_ms=None, actual_send_ms=5)]

        results = run_simple_metrics_pipeline(
            records, ReplaySendScheduleOffsetMetric.tag
        )

        assert ReplaySendScheduleOffsetMetric.tag not in results


class TestInjectReplaySchedLagPercentiles:
    def test_injection_anchors_at_least_late_request(self) -> None:
        # Send offsets of 5/5/15/105 ms anchor to [0, 0, 10, 100] ms (the
        # least-late requests define zero).
        results = _inject([5, 5, 15, 105])

        assert results[ReplaySchedLagP50Metric.tag].avg == pytest.approx(5.0)
        assert results[ReplaySchedLagP90Metric.tag].avg == pytest.approx(73.0)
        assert results[ReplaySchedLagP99Metric.tag].avg == pytest.approx(97.3)

    def test_injection_uniform_lag_reports_zero(self) -> None:
        # A constant delay on every request is invisible after anchoring --
        # the documented limitation of the post-hoc approximation.
        results = _inject([700, 700, 700])

        assert results[ReplaySchedLagP99Metric.tag].avg == 0.0

    def test_injection_noop_when_column_absent(self) -> None:
        results: dict[str, MetricResult] = {}
        inject_replay_sched_lag_metrics(ColumnStore(initial_capacity=4), results)
        assert results == {}

    def test_injection_noop_when_all_offsets_masked_out(self) -> None:
        results = _inject([5, 15], mask=np.array([False, False]))
        assert results == {}

    def test_mask_selects_subset(self) -> None:
        # Only the first two offsets (0, 10 ms anchored) count; the 500 ms tail
        # is masked out so it cannot inflate the percentiles.
        results = _inject([0, 10, 510], mask=np.array([True, True, False]))
        assert results[ReplaySchedLagP99Metric.tag].avg == pytest.approx(9.9)


class TestInjectReplaySchedDegraded:
    @pytest.mark.parametrize(
        "tail_lag_ms, expected",
        [
            param(400, 0.0, id="below_threshold"),
            param(500, 0.0, id="at_threshold_not_degraded"),
            param(600, 1.0, id="above_threshold"),
        ],
    )  # fmt: skip
    def test_degraded_flags_p99_over_threshold(
        self, tail_lag_ms: int, expected: float
    ) -> None:
        # Half the requests on time, half late by tail_lag_ms: p99 == tail_lag_ms.
        results = _inject([0.0] * 5 + [float(tail_lag_ms)] * 5)

        assert results[ReplaySchedDegradedMetric.tag].avg == expected

    def test_warn_degraded_called_once_with_percentiles(self) -> None:
        calls: list[tuple[float, float, float]] = []
        results = _inject([0.0] * 5 + [600.0] * 5, warn=lambda *a: calls.append(a))

        assert results[ReplaySchedDegradedMetric.tag].avg == 1.0
        assert len(calls) == 1
        p50, p90, p99 = calls[0]
        assert p99 == pytest.approx(600.0)

    def test_warn_degraded_not_called_when_healthy(self) -> None:
        calls: list[tuple[float, float, float]] = []
        _inject([0.0, 10.0, 20.0], warn=lambda *a: calls.append(a))
        assert calls == []


class TestDeferredDerivation:
    """The scalar summarize path cannot see the full offset array, so every
    metric in the family defers; the injection fills the values instead."""

    @pytest.mark.parametrize(
        "metric_cls",
        [
            ReplaySchedLagP50Metric,
            ReplaySchedLagP90Metric,
            ReplaySchedLagP99Metric,
            ReplaySchedDegradedMetric,
        ],
    )  # fmt: skip
    def test_derive_value_always_defers(self, metric_cls) -> None:
        with pytest.raises(NoMetricValue):
            metric_cls()._derive_value(MetricResultsDict())


class TestGoldenParityVsLegacy:
    """Prove the columnar injection is byte-for-byte identical to the legacy
    ``MetricResultsProcessor`` derive formula for the same per-record offsets."""

    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 1234])
    def test_injection_matches_legacy_formula(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        # Mixed magnitudes incl. a heavy tail so p99 straddles the degraded
        # threshold across seeds; offsets are epoch-scale ns like production.
        base = 1_700_000_000_000_000_000
        jitter_ns = rng.integers(0, 900_000_000, size=200)  # up to 900 ms late
        offsets_ns = (base + jitter_ns).astype(np.int64)

        golden = _legacy_replay_lag(offsets_ns)

        store = _store_with_offsets(offsets_ns.tolist())
        results: dict[str, MetricResult] = {}
        inject_replay_sched_lag_metrics(store, results)

        for tag, expected in golden.items():
            assert results[tag].avg == pytest.approx(expected, abs=1e-9), (
                f"{tag}: injection {results[tag].avg} != legacy {expected}"
            )
