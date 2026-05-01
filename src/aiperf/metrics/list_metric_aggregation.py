# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run-level aggregator for list-valued record metrics.

Used by :class:`aiperf.post_processors.metric_results_processor.MetricResultsProcessor`
when a ``MetricType.RECORD`` metric arrives with a list value (today only
``inter_chunk_latency``, where each request contributes a list of inter-chunk
gap durations). At 1 M-request ramp scale the exact storage —
``records × (chunks-1) × 8 B`` — would dwarf the records-manager pod's
memory budget. T-digest bounds it to a few KB regardless of sample count.

Backed by :class:`crick.TDigest` (Cython/C). Throughput measured at
~12 M updates/s with worst-case relative percentile error under 0.05%
on 50M-sample workloads at the default compression — see
``docs/reference/list-metric-aggregation.md`` for the empirical band.

Stats:
- ``count``, ``sum``, ``min``, ``max``, ``avg``, ``std`` are exact
  (running side-channel scalars).
- ``p1``..``p99`` are approximate via t-digest.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from crick import TDigest

from aiperf.common.environment import Environment
from aiperf.common.models import MetricResult
from aiperf.common.types import MetricTagT


class TDigestListMetricAggregator:
    """Bounded-memory aggregator backed by a t-digest sketch."""

    def __init__(self) -> None:
        self._td = TDigest(compression=Environment.METRICS.TDIGEST_COMPRESSION)
        self._count: int = 0
        self._sum: float = 0.0
        self._sum_sq: float = 0.0
        self._min: float | None = None
        self._max: float | None = None

    def append(self, value: int | float) -> None:
        """Add a single sample."""
        v = float(value)
        self._td.update(v)
        self._count += 1
        self._sum += v
        self._sum_sq += v * v
        self._min = v if self._min is None else min(self._min, v)
        self._max = v if self._max is None else max(self._max, v)

    def extend(self, values: Iterable[int | float]) -> None:
        """Add many samples in a single C-level update.

        The numpy round-trip avoids per-sample Python overhead in the
        sketch path; the side-channel scalars are still updated per-element
        so ``count``/``sum``/``min``/``max`` stay exact.
        """
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return
        self._td.update(arr)
        self._count += int(arr.size)
        self._sum += float(arr.sum())
        self._sum_sq += float(np.dot(arr, arr))
        batch_min = float(arr.min())
        batch_max = float(arr.max())
        self._min = batch_min if self._min is None else min(self._min, batch_min)
        self._max = batch_max if self._max is None else max(self._max, batch_max)

    def to_result(self, tag: MetricTagT, header: str, unit: str) -> MetricResult:
        """Return a :class:`MetricResult` with the same field set as
        ``MetricArray.to_result``. Percentiles come from the t-digest;
        every other stat is exact."""
        if self._count == 0:
            return MetricResult(tag=tag, header=header, unit=unit, count=0)
        avg = self._sum / self._count
        # Population variance, matching numpy's default for np.std(arr).
        # max(0, ...) clamps tiny floating-point underflow when all samples
        # are equal (sum_sq/count == avg^2 to <1ulp).
        var = max(0.0, self._sum_sq / self._count - avg * avg)
        std = math.sqrt(var)
        return MetricResult(
            tag=tag,
            header=header,
            unit=unit,
            count=self._count,
            sum=self._sum,
            min=self._min,
            max=self._max,
            avg=avg,
            std=std,
            p1=float(self._td.quantile(0.01)),
            p5=float(self._td.quantile(0.05)),
            p10=float(self._td.quantile(0.10)),
            p25=float(self._td.quantile(0.25)),
            p50=float(self._td.quantile(0.50)),
            p75=float(self._td.quantile(0.75)),
            p90=float(self._td.quantile(0.90)),
            p95=float(self._td.quantile(0.95)),
            p99=float(self._td.quantile(0.99)),
        )
