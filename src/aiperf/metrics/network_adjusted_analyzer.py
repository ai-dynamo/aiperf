# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Network-RTT-adjusted latency metrics, computed at summarize-time.

When network latency calibration is enabled, a single run-level mean RTT (ns) is
subtracted from each request-start-anchored latency metric's per-record array,
producing non-destructive ``network_adjusted_*`` variants plus a ``network_rtt``
summary scalar. The raw metrics are preserved.

Subtracting a constant RTT shifts the mean and every percentile by that constant
and leaves the standard deviation unchanged; the per-record subtraction is
clamped at 0 so a measured RTT larger than a (rare) sub-RTT latency cannot go
negative. This reads the columnar source arrays the accumulator already owns,
without re-aggregating a separate metric container.

Inter-token / inter-chunk latencies are intentionally NOT adjusted: the RTT
cancels in ``(request_latency - ttft)``, so those metrics are already
network-invariant (see ``NETWORK_ADJUSTED_SOURCES``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from aiperf.common.enums import MetricConsoleGroup
from aiperf.common.models import MetricResult
from aiperf.metrics.metric_dicts import metric_result_from_array
from aiperf.metrics.types.network_adjusted_metrics import (
    NETWORK_ADJUSTED_SOURCES,
    NetworkAdjustedRequestLatencyMetric,
    NetworkAdjustedTimeToFirstOutputTokenMetric,
    NetworkAdjustedTTFTMetric,
    NetworkRttMetric,
)

if TYPE_CHECKING:
    from aiperf.metrics.column_store import ColumnStore

_NS_PER_MS = 1_000_000.0

# Header metadata for each injected tag, sourced from the registered metric
# classes so console/export headers match the rest of the pipeline.
_ADJUSTED_METRICS = (
    NetworkAdjustedRequestLatencyMetric,
    NetworkAdjustedTTFTMetric,
    NetworkAdjustedTimeToFirstOutputTokenMetric,
)
_HEADER_BY_TAG = {m.tag: m.header for m in _ADJUSTED_METRICS}


def _array_to_metric_result(
    *, tag: str, header: str, values_ms: NDArray[np.float64]
) -> MetricResult:
    """Build a fully-populated :class:`MetricResult` from a fresh 1-D ms ndarray."""
    result = metric_result_from_array(
        tag,
        header,
        "ms",
        values_ms,
        float(values_ms.sum()),
    )
    result.console_group = MetricConsoleGroup.DEFAULT
    return result


def inject_network_adjusted_metrics(
    store: ColumnStore,
    results: dict[str, MetricResult],
    rtt_ns: float,
    mask: NDArray[np.bool_] | None = None,
) -> None:
    """Inject ``network_adjusted_*`` distributions and the ``network_rtt`` scalar.

    ``rtt_ns`` is the run-level mean network RTT (nanoseconds) resolved by the
    RecordsManager (manual ``--network-latency-mean`` override or the mean over
    successful probe samples). Caller guarantees ``rtt_ns`` is truthy; a 0/None
    RTT is a no-op handled upstream. ``mask`` restricts the distributions to the
    export window's records (e.g. a phase-scoped export); None means all records.
    """
    rtt_ms = rtt_ns / _NS_PER_MS

    # network_rtt: single run-level scalar (no per-record distribution).
    results[NetworkRttMetric.tag] = MetricResult(
        tag=NetworkRttMetric.tag,
        header=NetworkRttMetric.header,
        unit="ms",
        count=1,
        sum=rtt_ms,
        avg=rtt_ms,
        std=0.0,
        min=rtt_ms,
        max=rtt_ms,
        p1=rtt_ms,
        p5=rtt_ms,
        p10=rtt_ms,
        p25=rtt_ms,
        p50=rtt_ms,
        p75=rtt_ms,
        p90=rtt_ms,
        p95=rtt_ms,
        p99=rtt_ms,
        console_group=MetricConsoleGroup.DEFAULT,
    )

    for adjusted_tag, source_tag in NETWORK_ADJUSTED_SOURCES.items():
        source_ns = store.numeric(source_tag)
        if mask is not None:
            source_ns = source_ns[mask]
        valid_ns = source_ns[~np.isnan(source_ns)]
        if valid_ns.size == 0:
            continue
        adjusted_ms = np.maximum(valid_ns - rtt_ns, 0.0) / _NS_PER_MS
        results[adjusted_tag] = _array_to_metric_result(
            tag=adjusted_tag,
            header=_HEADER_BY_TAG[adjusted_tag],
            values_ms=adjusted_ms,
        )
