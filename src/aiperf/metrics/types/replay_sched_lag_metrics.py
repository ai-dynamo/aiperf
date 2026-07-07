# SPDX-FileCopyrightText: Copyright (c) 2026 Baseten.co. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay schedule send-lag metrics (offered vs. actual send-time fidelity).

In fixed-schedule trace replay, every absolutely-scheduled turn carries its
intended (offered) send time in ``Turn.timestamp`` (milliseconds relative to
the schedule zero), and the worker stamps ``RequestRecord.timestamp_ns``
(wall clock) when the request is dispatched to the transport. Their
difference is the per-request send offset. The wall-clock instant of the
schedule zero is not recorded anywhere, so the offset's absolute value is
meaningless post-hoc; lag is therefore reported relative to the least-late
request of the run::

    lag_i = offset_i - min(offset)

What this measures: the dispersion of send lag across the run. Schedule
degradation (event-loop stalls, worker saturation, queue buildup) shows up
as growing lag percentiles.

What this cannot measure:
- A constant delay applied uniformly to every request: anchoring makes the
  least-late request define zero, so a run-wide constant lateness is
  invisible.
- Lag of delay-scheduled continuation turns: under back-pressure replay they
  fire relative to the prior turn's completion, carry no absolute intended
  time (``Turn.timestamp is None``), and are excluded.
- Pure scheduler infidelity for continuation turns in non-strict open-loop
  mode: they keep absolute intended times, but fixed_schedule only dispatches
  them after the prior turn's credit returns, so their lag also includes
  prior-turn service time. The values are still honest send lateness against
  the recorded schedule.
- True wire time: ``timestamp_ns`` is stamped worker-side at transport
  dispatch, before the TCP write.
"""

from typing import ClassVar

import numpy as np

from aiperf.common.constants import NANOS_PER_MILLIS
from aiperf.common.enums import (
    GenericMetricUnit,
    MetricConsoleGroup,
    MetricFlags,
    MetricTimeUnit,
)
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.logging import AIPerfLogger
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics.base_derived_metric import BaseDerivedMetric
from aiperf.metrics.base_record_metric import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict, MetricResultsDict

_logger = AIPerfLogger(__name__)

REPLAY_SCHED_DEGRADED_THRESHOLD_MS: float = 500.0
"""Anchored send-lag p99 (ms) above which a replay run is flagged degraded.

At p99 lag beyond this, the tail of the offered schedule was delivered half a
second or more late relative to the least-late request, so tail latency and
concurrency measurements no longer reflect the recorded arrival process.
"""


class ReplaySendScheduleOffsetMetric(BaseRecordMetric[int]):
    """Raw per-request send offset: actual send wall time minus the turn's
    intended schedule timestamp.

    Internal building block for the ``replay_sched_lag_*`` metrics. Values are
    epoch-scale nanoseconds (wall clock minus a schedule-relative offset) and
    only differences between them are meaningful; see the module docstring.

    Formula:
        Offset = RequestRecord.timestamp_ns - Turn.timestamp * NANOS_PER_MILLIS
    """

    tag = "replay_send_schedule_offset"
    header = "Replay Send Schedule Offset"
    short_header = "Sched Offset"
    unit = MetricTimeUnit.NANOSECONDS
    flags = MetricFlags.INTERNAL | MetricFlags.FIXED_SCHEDULE_ONLY
    console_group = MetricConsoleGroup.NONE
    required_metrics = None

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        """Return actual-minus-intended send time in nanoseconds.

        Raises:
            NoMetricValue: If the dispatched turn is unavailable or carries no
                absolute schedule timestamp (e.g. delay-scheduled continuation
                turns, non-replay datasets).
        """
        request_info = record.request.request_info
        if request_info is None or not request_info.turns:
            raise NoMetricValue("Request info or turns not available in record.")

        intended_ms = request_info.turns[-1].timestamp
        if intended_ms is None:
            raise NoMetricValue(
                "Turn has no absolute schedule timestamp (not fixed-schedule replay)."
            )

        return record.timestamp_ns - int(intended_ms * NANOS_PER_MILLIS)


def _anchored_lag_ms(metric_results: MetricResultsDict) -> np.ndarray:
    """Return per-request send lag in ms, anchored at the least-late request.

    Raises:
        NoMetricValue: If no absolutely-scheduled request produced an offset.
    """
    values = metric_results.get_or_raise(ReplaySendScheduleOffsetMetric)
    data = np.asarray(values.data, dtype=np.float64)
    if len(data) == 0:
        raise NoMetricValue("No absolutely-scheduled requests were recorded.")
    return (data - data.min()) / NANOS_PER_MILLIS


class _ReplaySchedLagPercentileBase(BaseDerivedMetric[float]):
    """Shared derive logic for the anchored send-lag percentile metrics."""

    __is_abstract__ = True
    percentile: float

    unit = MetricTimeUnit.MILLISECONDS
    flags = MetricFlags.FIXED_SCHEDULE_ONLY
    console_group = MetricConsoleGroup.NONE
    required_metrics: ClassVar[set[str]] = {ReplaySendScheduleOffsetMetric.tag}

    def _derive_value(self, metric_results: MetricResultsDict) -> float:
        return float(np.percentile(_anchored_lag_ms(metric_results), self.percentile))


class ReplaySchedLagP50Metric(_ReplaySchedLagPercentileBase):
    """Median anchored send lag of a fixed-schedule replay run, in ms.

    Formula:
        p50(offset - min(offset)) / NANOS_PER_MILLIS
    """

    __is_abstract__ = False
    percentile = 50.0
    tag = "replay_sched_lag_p50"
    header = "Replay Schedule Lag p50"
    short_header = "Sched Lag p50"


class ReplaySchedLagP90Metric(_ReplaySchedLagPercentileBase):
    """90th-percentile anchored send lag of a fixed-schedule replay run, in ms.

    Formula:
        p90(offset - min(offset)) / NANOS_PER_MILLIS
    """

    __is_abstract__ = False
    percentile = 90.0
    tag = "replay_sched_lag_p90"
    header = "Replay Schedule Lag p90"
    short_header = "Sched Lag p90"


class ReplaySchedLagP99Metric(_ReplaySchedLagPercentileBase):
    """99th-percentile anchored send lag of a fixed-schedule replay run, in ms.

    Formula:
        p99(offset - min(offset)) / NANOS_PER_MILLIS
    """

    __is_abstract__ = False
    percentile = 99.0
    tag = "replay_sched_lag_p99"
    header = "Replay Schedule Lag p99"
    short_header = "Sched Lag p99"


class ReplaySchedDegradedMetric(BaseDerivedMetric[int]):
    """Boolean (0/1) signal that the replay could not keep up with the offered
    schedule: anchored send-lag p99 exceeded
    :data:`REPLAY_SCHED_DEGRADED_THRESHOLD_MS`.

    Logs a one-line warning (at most once per run) with the lag percentiles
    when degraded, so runs whose timing fidelity is compromised are called out
    even though these metrics are hidden from the console table. The metric
    value itself stays continuous: it is re-derived on every summarize tick.

    Formula:
        1 if p99(offset - min(offset)) > REPLAY_SCHED_DEGRADED_THRESHOLD_MS else 0
    """

    tag = "replay_sched_degraded"
    header = "Replay Schedule Degraded"
    short_header = "Sched Degraded"
    unit = GenericMetricUnit.COUNT
    flags = MetricFlags.FIXED_SCHEDULE_ONLY
    console_group = MetricConsoleGroup.NONE
    required_metrics: ClassVar[set[str]] = {
        ReplaySchedLagP50Metric.tag,
        ReplaySchedLagP90Metric.tag,
        ReplaySchedLagP99Metric.tag,
    }

    def __init__(self) -> None:
        super().__init__()
        self._warned = False

    def _derive_value(self, metric_results: MetricResultsDict) -> int:
        p50 = metric_results.get_or_raise(ReplaySchedLagP50Metric)
        p90 = metric_results.get_or_raise(ReplaySchedLagP90Metric)
        p99 = metric_results.get_or_raise(ReplaySchedLagP99Metric)
        degraded = p99 > REPLAY_SCHED_DEGRADED_THRESHOLD_MS
        if degraded and not self._warned:
            self._warned = True
            _logger.warning(
                f"Replay schedule degraded: anchored send lag p50={p50:.0f} ms, "
                f"p90={p90:.0f} ms, p99={p99:.0f} ms exceeds "
                f"{REPLAY_SCHED_DEGRADED_THRESHOLD_MS:.0f} ms. Request timing no "
                f"longer tracks the recorded schedule; consider lowering "
                f"replay_speedup or the offered load."
            )
        return int(degraded)
