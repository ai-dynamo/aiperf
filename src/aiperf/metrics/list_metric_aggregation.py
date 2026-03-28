# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

from tdigest import TDigest

from aiperf.common.enums import ListMetricAggregationMode
from aiperf.common.models.record_models import MetricResult
from aiperf.common.types import MetricTagT
from aiperf.metrics.metric_dicts import MetricArray

__all__ = [
    "ExactListMetricAggregator",
    "ListMetricAggregator",
    "TDigestListMetricAggregator",
    "build_list_metric_aggregator",
]


class ListMetricAggregator:
    """Base accumulator for run-level list-valued metric summaries."""

    def __init__(self) -> None:
        self._count = 0
        self._sum = 0.0
        self._sum_squares = 0.0
        self._min: float | None = None
        self._max: float | None = None

    def append(self, value: float | int) -> None:
        """Append a single metric value."""
        float_value = float(value)
        self._observe(float_value)
        self._append(float_value)

    def extend(self, values: list[float] | list[int]) -> None:
        """Append multiple metric values."""
        for value in values:
            self.append(value)

    def to_result(self, tag: MetricTagT, header: str, unit: str) -> MetricResult:
        """Convert the aggregated values into the existing MetricResult shape."""
        self._raise_if_empty()
        avg = self._sum / self._count
        variance = max((self._sum_squares / self._count) - (avg * avg), 0.0)
        return MetricResult(
            tag=tag,
            header=header,
            unit=unit,
            min=self._min,
            max=self._max,
            avg=avg,
            sum=self._sum,
            std=math.sqrt(variance),
            p1=self._percentile(1),
            p5=self._percentile(5),
            p10=self._percentile(10),
            p25=self._percentile(25),
            p50=self._percentile(50),
            p75=self._percentile(75),
            p90=self._percentile(90),
            p95=self._percentile(95),
            p99=self._percentile(99),
            count=self._count,
        )

    def _observe(self, value: float) -> None:
        """Update summary statistics for an observed metric value."""
        self._count += 1
        self._sum += value
        self._sum_squares += value * value
        self._min = value if self._min is None else min(self._min, value)
        self._max = value if self._max is None else max(self._max, value)

    def _append(self, value: float) -> None:
        """Store a single metric value in the backing accumulator."""
        raise NotImplementedError

    def _percentile(self, percentile: int) -> float:
        """Read a percentile from the backing accumulator."""
        raise NotImplementedError

    def _raise_if_empty(self) -> None:
        """Validate that at least one metric value has been observed."""
        if self._count == 0:
            raise IndexError("Cannot summarize an empty list metric aggregator")


class ExactListMetricAggregator(ListMetricAggregator):
    """Exact list metric accumulator backed by MetricArray."""

    def __init__(self) -> None:
        super().__init__()
        self._values = MetricArray()

    def append(self, value: float | int) -> None:
        """Append a single metric value."""
        super().append(value)

    def extend(self, values: list[float] | list[int]) -> None:
        """Append multiple metric values."""
        if not values:
            return
        float_values = [float(value) for value in values]
        for value in float_values:
            self._observe(value)
        self._values.extend(float_values)

    def to_result(self, tag: MetricTagT, header: str, unit: str) -> MetricResult:
        """Delegate exact summaries to MetricArray."""
        self._raise_if_empty()
        return self._values.to_result(tag, header, unit)

    def _append(self, value: float) -> None:
        """Store a single metric value in the MetricArray."""
        self._values.append(value)

    def _percentile(self, percentile: int) -> float:
        """Exact aggregation delegates summary generation to MetricArray."""
        raise NotImplementedError


class TDigestListMetricAggregator(ListMetricAggregator):
    """Approximate list metric accumulator backed by a t-digest sketch."""

    def __init__(self) -> None:
        super().__init__()
        self._digest = TDigest()

    def _append(self, value: float) -> None:
        """Store a single metric value in the digest."""
        self._digest.update(value)

    def _percentile(self, percentile: int) -> float:
        """Read an approximate percentile from the digest."""
        self._raise_if_empty()
        return float(self._digest.percentile(percentile))


def build_list_metric_aggregator(
    mode: ListMetricAggregationMode,
) -> ListMetricAggregator:
    """Build the configured list metric accumulator implementation."""
    if mode == ListMetricAggregationMode.EXACT:
        return ExactListMetricAggregator()
    if mode == ListMetricAggregationMode.TDIGEST:
        return TDigestListMetricAggregator()
    raise ValueError(f"Unsupported list metric aggregation mode: {mode}")
