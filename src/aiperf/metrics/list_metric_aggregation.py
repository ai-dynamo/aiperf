# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run-level aggregator for list-valued record metrics.

Used by :class:`aiperf.post_processors.metric_results_processor.MetricResultsProcessor`
when a ``MetricType.RECORD`` metric arrives with a list value (today only
``inter_chunk_latency``, where each request contributes a list of inter-chunk
gap durations). At 1 M-request ramp scale the exact storage —
``records × (chunks-1) × 8 B`` — would dwarf the records-manager pod's
memory budget. T-digest bounds it to a few KB regardless of sample count.

Stats:
- ``count``, ``sum``, ``min``, ``max``, ``avg``, ``std`` are exact
  (running side-channel scalars).
- ``p1``..``p99`` are approximate via t-digest (~0.5% relative error).
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import tdigest as _tdigest

from aiperf.common.models import MetricResult
from aiperf.common.types import MetricTagT


class TDigestListMetricAggregator:
    """Bounded-memory aggregator backed by a t-digest sketch."""

    def __init__(self) -> None:
        self._td = _tdigest.TDigest()
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
        """Add many samples. Iterable is consumed once."""
        for v in values:
            self.append(v)

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
            p1=self._td.percentile(1),
            p5=self._td.percentile(5),
            p10=self._td.percentile(10),
            p25=self._td.percentile(25),
            p50=self._td.percentile(50),
            p75=self._td.percentile(75),
            p90=self._td.percentile(90),
            p95=self._td.percentile(95),
            p99=self._td.percentile(99),
        )
