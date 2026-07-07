# SPDX-FileCopyrightText: Copyright (c) 2026 Baseten.co. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the replay schedule send-lag metrics.

Synthetic records carry an intended schedule timestamp on the dispatched turn
(``Turn.timestamp``, ms) and an actual send wall time
(``RequestRecord.timestamp_ns``), exactly as fixed-schedule replay produces.
"""

import pytest
from pytest import param

from aiperf.common.constants import NANOS_PER_MILLIS
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
        # Intended at 0/10/20/30 ms; actual lags of 5/5/15/105 ms anchor to
        # [0, 0, 10, 100] ms (the least-late requests define zero).
        records = [
            _scheduled_record(intended_ms=0, actual_send_ms=5),
            _scheduled_record(intended_ms=10, actual_send_ms=15),
            _scheduled_record(intended_ms=20, actual_send_ms=35),
            _scheduled_record(intended_ms=30, actual_send_ms=135),
        ]

        results = run_simple_metrics_pipeline(
            records,
            ReplaySchedLagP50Metric.tag,
            ReplaySchedLagP90Metric.tag,
            ReplaySchedLagP99Metric.tag,
        )

        # np.percentile (linear interpolation) over [0, 0, 10, 100].
        assert results[ReplaySchedLagP50Metric.tag] == pytest.approx(5.0)
        assert results[ReplaySchedLagP90Metric.tag] == pytest.approx(73.0)
        assert results[ReplaySchedLagP99Metric.tag] == pytest.approx(97.3)

    def test_derive_value_uniform_lag_reports_zero(self):
        # A constant delay on every request is invisible after anchoring --
        # the documented limitation of the post-hoc approximation.
        records = [
            _scheduled_record(intended_ms=0, actual_send_ms=700),
            _scheduled_record(intended_ms=10, actual_send_ms=710),
            _scheduled_record(intended_ms=20, actual_send_ms=720),
        ]

        results = run_simple_metrics_pipeline(records, ReplaySchedLagP99Metric.tag)

        assert results[ReplaySchedLagP99Metric.tag] == 0.0

    def test_derive_value_no_offsets_raises_no_metric_value(self):
        with pytest.raises(NoMetricValue):
            ReplaySchedLagP99Metric().derive_value(MetricResultsDict())

    def test_derive_value_metric_array_results_returns_exact_percentile(self):
        # Production stores scalar record metrics as MetricArray, not the
        # plain list run_simple_metrics_pipeline uses (see
        # post_processors/metric_results_processor.py first-touch storage).
        results = MetricResultsDict()
        results[ReplaySendScheduleOffsetMetric.tag] = create_metric_array(
            [5 * NANOS_PER_MILLIS, 15 * NANOS_PER_MILLIS, 105 * NANOS_PER_MILLIS]
        )

        # Anchored lag [0, 10, 100] ms -> p99 = 10 + 0.98 * 90 = 98.2.
        assert ReplaySchedLagP99Metric().derive_value(results) == pytest.approx(98.2)


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
        records = [
            _scheduled_record(intended_ms=i, actual_send_ms=i) for i in range(5)
        ] + [
            _scheduled_record(intended_ms=5 + i, actual_send_ms=5 + i + tail_lag_ms)
            for i in range(5)
        ]

        results = run_simple_metrics_pipeline(records, ReplaySchedDegradedMetric.tag)

        assert results[ReplaySchedDegradedMetric.tag] == expected
