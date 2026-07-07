# SPDX-FileCopyrightText: Copyright (c) 2026 Baseten.co. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the replay schedule send-lag metrics.

Record-metric tests drive synthetic records that carry an intended schedule
timestamp on the dispatched turn (``Turn.timestamp``, ms) and an actual send
wall time (``RequestRecord.timestamp_ns``), exactly as fixed-schedule replay
produces. Derived-metric tests build ``MetricResultsDict`` entries in the
production storage shape (``MetricArray`` for record metrics, floats for
already-derived dependencies; see
post_processors/metric_results_processor.py first-touch storage).
"""

import logging

import pytest
from pytest import param

from aiperf.common.constants import NANOS_PER_MILLIS
from aiperf.common.enums import MetricFlags
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord, Turn
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.replay_sched_lag_metrics import (
    ReplaySchedDegradedMetric,
    ReplaySchedLagP50Metric,
    ReplaySchedLagP90Metric,
    ReplaySchedLagP99Metric,
    ReplaySendScheduleOffsetMetric,
)
from tests.unit.metrics.conftest import (
    create_metric_array,
    create_record,
    run_simple_metrics_pipeline,
)


def _scheduled_record(
    intended_ms: float | None, actual_send_ms: float
) -> ParsedResponseRecord:
    """Record whose turn was scheduled at ``intended_ms`` (schedule-relative)
    and actually sent at ``actual_send_ms`` (wall clock, same test clock)."""
    record = create_record(start_ns=int(actual_send_ms * NANOS_PER_MILLIS))
    record.request.request_info.turns = [Turn(timestamp=intended_ms)]
    return record


def _offset_results(offsets_ms: list[float]) -> MetricResultsDict:
    """Results dict holding raw send offsets as production stores them."""
    results = MetricResultsDict()
    results[ReplaySendScheduleOffsetMetric.tag] = create_metric_array(
        [int(ms * NANOS_PER_MILLIS) for ms in offsets_ms]
    )
    return results


def _derived_lag_results(offsets_ms: list[float]) -> MetricResultsDict:
    """Results dict with the lag percentiles already derived, mirroring
    update_derived_metrics' dependency-ordered store-before-dependents."""
    results = _offset_results(offsets_ms)
    for metric_cls in (
        ReplaySchedLagP50Metric,
        ReplaySchedLagP90Metric,
        ReplaySchedLagP99Metric,
    ):
        results[metric_cls.tag] = metric_cls().derive_value(results)
    return results


def test_family_is_fixed_schedule_only():
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
    def test_parse_record_returns_actual_minus_intended_ns(self):
        records = [_scheduled_record(intended_ms=2, actual_send_ms=5)]

        results = run_simple_metrics_pipeline(
            records, ReplaySendScheduleOffsetMetric.tag
        )

        assert results[ReplaySendScheduleOffsetMetric.tag] == [3 * NANOS_PER_MILLIS]

    def test_parse_record_no_turn_timestamp_yields_no_value(self):
        records = [_scheduled_record(intended_ms=None, actual_send_ms=5)]

        results = run_simple_metrics_pipeline(
            records, ReplaySendScheduleOffsetMetric.tag
        )

        assert ReplaySendScheduleOffsetMetric.tag not in results


class TestReplaySchedLagPercentiles:
    def test_derive_value_anchors_at_least_late_request(self):
        # Send offsets of 5/5/15/105 ms anchor to [0, 0, 10, 100] ms (the
        # least-late requests define zero).
        results = _offset_results([5, 5, 15, 105])

        # np.percentile (linear interpolation) over [0, 0, 10, 100].
        assert ReplaySchedLagP50Metric().derive_value(results) == pytest.approx(5.0)
        assert ReplaySchedLagP90Metric().derive_value(results) == pytest.approx(73.0)
        assert ReplaySchedLagP99Metric().derive_value(results) == pytest.approx(97.3)

    def test_derive_value_uniform_lag_reports_zero(self):
        # A constant delay on every request is invisible after anchoring --
        # the documented limitation of the post-hoc approximation.
        results = _offset_results([700, 700, 700])

        assert ReplaySchedLagP99Metric().derive_value(results) == 0.0

    def test_derive_value_no_offsets_raises_no_metric_value(self):
        with pytest.raises(NoMetricValue):
            ReplaySchedLagP99Metric().derive_value(MetricResultsDict())

    def test_derive_value_empty_offset_array_raises_no_metric_value(self):
        with pytest.raises(NoMetricValue):
            ReplaySchedLagP99Metric().derive_value(_offset_results([]))


class TestReplaySchedDegradedMetric:
    @pytest.mark.parametrize(
        "tail_lag_ms, expected",
        [
            param(400, 0, id="below_threshold"),
            param(500, 0, id="at_threshold_not_degraded"),
            param(600, 1, id="above_threshold"),
        ],
    )  # fmt: skip
    def test_derive_value_flags_p99_over_threshold(
        self, tail_lag_ms: int, expected: int
    ):
        # Half the requests on time, half late by tail_lag_ms: p99 == tail_lag_ms.
        results = _derived_lag_results([0.0] * 5 + [float(tail_lag_ms)] * 5)

        assert ReplaySchedDegradedMetric().derive_value(results) == expected

    def test_derive_value_missing_percentiles_raises_no_metric_value(self):
        with pytest.raises(NoMetricValue):
            ReplaySchedDegradedMetric().derive_value(MetricResultsDict())

    def test_degraded_warning_logged_once_per_run(
        self, caplog: pytest.LogCaptureFixture
    ):
        # summarize() re-derives every realtime tick; the warning must not
        # repeat while the metric value stays continuous.
        results = _derived_lag_results([0.0] * 5 + [600.0] * 5)
        metric = ReplaySchedDegradedMetric()

        with caplog.at_level(logging.WARNING):
            assert metric.derive_value(results) == 1
            assert metric.derive_value(results) == 1
            assert metric.derive_value(results) == 1

        warnings = [
            r for r in caplog.records if "Replay schedule degraded" in r.message
        ]
        assert len(warnings) == 1
