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
    """Build a fully-populated :class:`MetricResult` from a 1-D ms ndarray."""
    p1, p5, p10, p25, p50, p75, p90, p95, p99 = np.percentile(
        values_ms, [1, 5, 10, 25, 50, 75, 90, 95, 99]
    )
    return MetricResult(
        tag=tag,
        header=header,
        unit="ms",
        count=int(values_ms.size),
        sum=float(values_ms.sum()),
        avg=float(values_ms.mean()),
        std=float(values_ms.std()),
        min=float(values_ms.min()),
        max=float(values_ms.max()),
        p1=float(p1),
        p5=float(p5),
        p10=float(p10),
        p25=float(p25),
        p50=float(p50),
        p75=float(p75),
        p90=float(p90),
        p95=float(p95),
        p99=float(p99),
        console_group=MetricConsoleGroup.DEFAULT,
    )


def _network_rtt_result(rtt_ns: float) -> MetricResult:
    """Build the single run-level ``network_rtt`` scalar MetricResult (ms)."""
    rtt_ms = rtt_ns / _NS_PER_MS
    return MetricResult(
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


def compute_network_adjusted_arrays(
    store: ColumnStore, rtt_ns: float
) -> dict[str, NDArray[np.float64]]:
    """Compute the clamped per-record network-adjusted latency arrays ONCE.

    The adjustment ``max(latency - rtt, 0)`` is a per-record transform independent
    of any aggregation window, so it is computed a single time over each full
    source column (ms, full length, NaN preserved). The caller then aggregates
    masked views of these arrays for the overall summary and each timeslice via
    :func:`inject_network_adjusted_from_arrays` -- no redundant per-window
    subtraction. ``rtt_ns`` is the run-level mean RTT (nanoseconds); caller
    guarantees it is truthy.
    """
    return {
        adjusted_tag: np.maximum(store.numeric(source_tag) - rtt_ns, 0.0) / _NS_PER_MS
        for adjusted_tag, source_tag in NETWORK_ADJUSTED_SOURCES.items()
    }


def inject_network_adjusted_from_arrays(
    adjusted_arrays: dict[str, NDArray[np.float64]],
    results: dict[str, MetricResult],
    rtt_ns: float,
    mask: NDArray[np.bool_] | None = None,
) -> None:
    """Aggregate precomputed adjusted arrays into ``results`` for one window.

    ``adjusted_arrays`` comes from :func:`compute_network_adjusted_arrays` (built
    once per summarize). ``mask`` restricts to the window's records (a timeslice
    bin or a phase-scoped export); None means all records. Also injects the
    run-level ``network_rtt`` scalar.
    """
    results[NetworkRttMetric.tag] = _network_rtt_result(rtt_ns)
    for adjusted_tag, adjusted_ms in adjusted_arrays.items():
        values_ms = adjusted_ms if mask is None else adjusted_ms[mask]
        valid_ms = values_ms[~np.isnan(values_ms)]
        if valid_ms.size == 0:
            continue
        results[adjusted_tag] = _array_to_metric_result(
            tag=adjusted_tag,
            header=_HEADER_BY_TAG[adjusted_tag],
            values_ms=valid_ms,
        )
