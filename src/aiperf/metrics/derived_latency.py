# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Derived per-record latency metrics computed from the column store at
summarize-time.

These metrics are not part of the per-record metric set the worker submits;
they are reconstructed at the records-manager from stored timestamps and
metadata columns:

- ``credit_to_start_latency``: ``request_start_ns - credit_issued_ns``.
  Surface controller queue saturation.
- ``effective_latency``: ``end_ns - credit_issued_ns``. Coordinated-omission-
  aware request latency that includes credit-queue wait time. Measures the
  latency a saturating user actually perceives.

All percentiles are computed exactly via numpy on the per-record arrays;
results are emitted in display units (milliseconds) so consumers don't need
the metric registry to convert them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from aiperf.common.enums import MetricConsoleGroup, MetricFlags
from aiperf.common.models import MetricResult

if TYPE_CHECKING:
    from aiperf.metrics.base_metric import BaseMetric
    from aiperf.metrics.column_store import ColumnStore


_NS_PER_MS = 1_000_000.0


def _array_to_metric_result(
    *,
    tag: str,
    header: str,
    unit: str,
    values_ms: NDArray[np.float64],
    console_group: MetricConsoleGroup = MetricConsoleGroup.EFFECTIVE,
) -> MetricResult:
    """Build a fully-populated :class:`MetricResult` from a 1-D ndarray."""
    p1, p5, p10, p25, p50, p75, p90, p95, p99 = np.percentile(
        values_ms, [1, 5, 10, 25, 50, 75, 90, 95, 99]
    )
    return MetricResult(
        tag=tag,
        header=header,
        unit=unit,
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
        console_group=console_group,
    )


def _delta_ms(
    end: NDArray[np.float64], begin: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Compute ``(end - begin) / 1e6`` and drop NaN entries.

    NaN propagates through the subtraction when either side has missing data
    (typically ``credit_issued_ns`` absent for fixed-schedule workloads), so a
    single ``isnan`` filter at the end suffices.
    """
    delta_ns = end - begin
    valid = delta_ns[~np.isnan(delta_ns)]
    return valid / _NS_PER_MS


def compute_credit_to_start_latency(
    store: ColumnStore, mask: NDArray[np.bool_] | None = None
) -> MetricResult | None:
    """Per-record credit-queue wait — ``request_start_ns - credit_issued_ns``.

    Returns ``None`` when no records have ``credit_issued_ns`` populated
    (e.g. fixed-schedule workloads that bypass the credit issuer).
    """
    n = store.count
    if n == 0:
        return None
    issued_col = store.metadata_numeric("credit_issued_ns")
    if issued_col.size == 0:
        return None
    start_ns = store.start_ns[:n]
    if mask is not None:
        start_ns = start_ns[mask]
        issued_col = issued_col[mask]
    values_ms = _delta_ms(start_ns, issued_col)
    if values_ms.size == 0:
        return None
    return _array_to_metric_result(
        tag="credit_to_start_latency",
        header="Credit-to-Start Latency",
        unit="ms",
        values_ms=values_ms,
        console_group=MetricConsoleGroup.NONE,
    )


def compute_effective_latency(
    store: ColumnStore, mask: NDArray[np.bool_] | None = None
) -> MetricResult | None:
    """Coordinated-omission-aware latency — ``end_ns - credit_issued_ns``.

    Captures the latency a user perceives under a saturating load generator:
    the request finishes at ``end_ns`` but the user issued it at
    ``credit_issued_ns``, so the queue wait is part of the perceived latency.
    Compare to ``request_latency`` (``end_ns - start_ns``) to see how much
    of perceived latency is queue-induced vs server-induced.
    """
    n = store.count
    if n == 0:
        return None
    issued_col = store.metadata_numeric("credit_issued_ns")
    if issued_col.size == 0:
        return None
    end_ns = store.end_ns[:n]
    if mask is not None:
        end_ns = end_ns[mask]
        issued_col = issued_col[mask]
    values_ms = _delta_ms(end_ns, issued_col)
    if values_ms.size == 0:
        return None
    return _array_to_metric_result(
        tag="effective_latency",
        header="Effective Latency (CO-aware)",
        unit="ms",
        values_ms=values_ms,
    )


def inject_derived_latency_metrics(
    store: ColumnStore,
    results: dict[str, MetricResult],
    mask: NDArray[np.bool_] | None = None,
) -> None:
    """Inject ``credit_to_start_latency`` and ``effective_latency`` into
    ``results`` if their prerequisite columns are populated. Pure side-effect."""
    for tag, result in (
        ("credit_to_start_latency", compute_credit_to_start_latency(store, mask)),
        ("effective_latency", compute_effective_latency(store, mask)),
    ):
        if result is not None:
            results[tag] = result


# Percentile points for the failure-inflated band — same as the regular band.
_ADJ_FULL_PERCENTILE_QS = np.array([1, 5, 10, 25, 50, 75, 90, 95, 99], dtype=np.float64)


def inject_adjusted_latency_metrics(
    results: dict[str, MetricResult],
    record_arrays: dict[str, tuple[NDArray[np.float64], float]],
    error_count: int,
    metric_classes: dict[str, type[BaseMetric]],
) -> None:
    """For each metric in ``results`` flagged with
    ``PERCENTILE_INCLUDES_FAILED_REQUESTS``, emit a separate ``adj_<tag>``
    MetricResult containing the failure-inflated distribution. Issue #688.

    Treating the adjusted view as its own MetricResult (rather than as
    sidecar fields on the base metric) matches the convention used by every
    load-testing and observability tool surveyed during design — k6 community
    workflow, HDR Histogram raw-vs-corrected pattern, Prometheus / OTel
    metric-name-as-key. Fields that don't fit the success-only model
    (``adj_avg``, ``adj_min``, ``adj_max``, ``adj_std``, ``adj_count``,
    plus the full p1/p5/p10/p25/p75 band) all populate naturally because
    the result is a regular ``MetricResult``.

    No-op when ``error_count == 0`` — there's nothing to inflate so the
    adjusted distribution would equal the success-only one.

    The success arrays in ``record_arrays`` are sorted ascending in-place by
    :func:`metric_result_from_array`, which the caller has already invoked.
    Appending ``+inf`` keeps the array sorted; ``np.percentile(..., method="nearest")``
    avoids the linear-interpolation NaN that hits ``inf - inf`` boundaries.
    """
    if error_count <= 0:
        return
    for tag, (clean, _arr_sum) in record_arrays.items():
        mc = metric_classes.get(tag)
        if mc is None:
            continue
        has_flags = getattr(mc, "has_flags", None)
        if not (
            has_flags and has_flags(MetricFlags.PERCENTILE_INCLUDES_FAILED_REQUESTS)
        ):
            continue
        adj = _build_adjusted_metric_result(tag, mc, clean, error_count)
        results[adj.tag] = adj


def _build_adjusted_metric_result(
    tag: str,
    metric_cls: type[BaseMetric],
    clean_sorted: NDArray[np.float64],
    error_count: int,
) -> MetricResult:
    """Compute the failure-inflated MetricResult for a flagged latency metric.

    ``clean_sorted`` is the success-only sample set (sorted ascending by the
    in-place sort inside :func:`metric_result_from_array`). The native unit
    is preserved on the returned MetricResult; ``to_display_unit`` later
    falls back to the parent tag for unit conversion (see
    :mod:`aiperf.metrics.display_units`).
    """
    inflated = np.concatenate(
        [clean_sorted, np.full(error_count, np.inf, dtype=np.float64)]
    )
    n = int(inflated.size)
    pcts = np.percentile(inflated, _ADJ_FULL_PERCENTILE_QS, method="nearest")
    # ``avg`` and ``sum`` correctly become ``inf`` when any failure is
    # present — that's the user-perceived-latency-under-failure reading
    # the metric is meant to surface. ``std`` is mathematically undefined
    # in a distribution containing ``inf`` (yields NaN through the
    # variance computation), so we clamp it to ``None`` to avoid emitting
    # a NaN that the JSON encoder may render as ``null`` or reject.
    return MetricResult(
        tag=f"adj_{tag}",
        header=f"{metric_cls.header} (error-adjusted)",
        unit=str(metric_cls.unit),
        count=n,
        sum=float(np.sum(inflated)),
        avg=float(np.mean(inflated)),
        min=float(np.min(inflated)),
        max=float(np.max(inflated)),
        std=None,
        p1=float(pcts[0]),
        p5=float(pcts[1]),
        p10=float(pcts[2]),
        p25=float(pcts[3]),
        p50=float(pcts[4]),
        p75=float(pcts[5]),
        p90=float(pcts[6]),
        p95=float(pcts[7]),
        p99=float(pcts[8]),
    )
