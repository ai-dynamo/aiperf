# SPDX-FileCopyrightText: Copyright (c) 2026 Baseten.co. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay send-lag analyzer: compute the ``replay_sched_lag_*`` family from the
column store at summarize-time.

The metrics themselves are defined in
:mod:`aiperf.metrics.types.replay_sched_lag_metrics` and defer their derivation;
this module owns the columnar computation and injection, mirroring
:mod:`aiperf.metrics.network_adjusted_analyzer` and
:mod:`aiperf.metrics.derived_latency`.

The family is a distribution over the run-global ``replay_send_schedule_offset``
column anchored at the least-late request, so it is computed once over the full
(masked) column and never per timeslice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from aiperf.common.constants import NANOS_PER_MILLIS
from aiperf.common.models import MetricResult
from aiperf.metrics.types.replay_sched_lag_metrics import (
    REPLAY_SCHED_DEGRADED_THRESHOLD_MS,
    ReplaySchedDegradedMetric,
    ReplaySchedLagP50Metric,
    ReplaySchedLagP90Metric,
    ReplaySchedLagP99Metric,
    ReplaySchedLagPercentileBase,
    ReplaySendScheduleOffsetMetric,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from aiperf.metrics.column_store import ColumnStore


# The ``percentile`` each class declares is the single source of truth for both
# its identity and the value the analyzer computes.
_PERCENTILE_METRICS: tuple[type[ReplaySchedLagPercentileBase], ...] = (
    ReplaySchedLagP50Metric,
    ReplaySchedLagP90Metric,
    ReplaySchedLagP99Metric,
)


def inject_replay_sched_lag_metrics(
    store: ColumnStore,
    results: dict[str, MetricResult],
    mask: NDArray[np.bool_] | None = None,
    *,
    warn_degraded: Callable[[float, float, float], None] | None = None,
) -> None:
    """Inject the anchored replay send-lag percentiles + degraded flag.

    Run-scoped: the family anchors at the run-global least-late request, so it is
    computed once over the full (masked) ``replay_send_schedule_offset`` column
    and never per timeslice. No-op when the column is absent (non-fixed-schedule
    runs) or holds no absolutely-scheduled offsets. ``warn_degraded``, if given,
    is called with ``(p50, p90, p99)`` once when the run is flagged degraded so
    the caller can emit a single run-level warning. Pure side effect on
    ``results``.
    """
    if ReplaySendScheduleOffsetMetric.tag not in store.numeric_tags():
        return
    offsets = store.numeric(ReplaySendScheduleOffsetMetric.tag)
    if mask is not None:
        offsets = offsets[mask]
    offsets = offsets[~np.isnan(offsets)]
    if offsets.size == 0:
        return

    anchored_ms = (offsets - offsets.min()) / NANOS_PER_MILLIS
    # Each percentile is a single run-level derived scalar (count=1), matching
    # the legacy derive path and the sibling network_rtt injection.
    lag: dict[str, float] = {}
    for cls in _PERCENTILE_METRICS:
        value = float(np.percentile(anchored_ms, cls.percentile))
        lag[cls.tag] = value
        results[cls.tag] = MetricResult(
            tag=cls.tag,
            header=cls.header,
            unit=str(cls.unit),
            avg=value,
            count=1,
            console_group=cls.console_group,
        )

    degraded = lag[ReplaySchedLagP99Metric.tag] > REPLAY_SCHED_DEGRADED_THRESHOLD_MS
    results[ReplaySchedDegradedMetric.tag] = MetricResult(
        tag=ReplaySchedDegradedMetric.tag,
        header=ReplaySchedDegradedMetric.header,
        unit=str(ReplaySchedDegradedMetric.unit),
        avg=float(int(degraded)),
        count=1,
        console_group=ReplaySchedDegradedMetric.console_group,
    )
    if degraded and warn_degraded is not None:
        warn_degraded(
            lag[ReplaySchedLagP50Metric.tag],
            lag[ReplaySchedLagP90Metric.tag],
            lag[ReplaySchedLagP99Metric.tag],
        )
