# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING

import numpy as np

from aiperf.common.enums import MetricType
from aiperf.common.enums.metric_enums import (
    MetricDictValueTypeT,
    MetricUnitT,
    MetricValueTypeT,
)
from aiperf.common.models.record_models import MetricResult
from aiperf.common.types import MetricTagT

if TYPE_CHECKING:
    from aiperf.metrics.base_metric import BaseMetric


class MetricRecordDict(dict[MetricTagT, MetricValueTypeT]):
    """
    A dict of metrics for a single record. This is used to store the current values
    of all metrics that have been computed for a single record.

    This will include:
    - The current value of any `BaseRecordMetric` that has been computed for this record.
    - The new value of any `BaseAggregateMetric` that has been computed for this record.
    - No `BaseDerivedMetric`s will be included.
    """

    def get_converted(
        self, metric: type["BaseMetric"], other_unit: MetricUnitT
    ) -> float:
        """Get the value of a metric, but converted to a different unit."""
        return metric.unit.convert_to(other_unit, self[metric.tag])  # type: ignore


class MetricResultsDict(dict[MetricTagT, MetricDictValueTypeT]):
    """
    A dict of metrics over an entire run. This is used to store the final values
    of all metrics that have been computed for an entire run.

    This will include:
    - All `BaseRecordMetric`s as a deque of their values.
    - The most recent value of each `BaseAggregateMetric`.
    - The value of any `BaseDerivedMetric` that has already been computed.
    """

    def get_converted(
        self, metric: type["BaseMetric"], other_unit: MetricUnitT
    ) -> float:
        """Get the value of a metric, but converted to a different unit."""
        if metric.type == MetricType.RECORD:
            # Record metrics are a deque of values, so we can't convert them directly.
            raise ValueError(
                f"Cannot convert a record metric to a different unit: {metric.tag}"
            )
        return metric.unit.convert_to(other_unit, self[metric.tag])  # type: ignore


class MetricArray:
    """NumPy backed array for metric data."""

    def __init__(self, initial_capacity: int = 10000):
        self._capacity = initial_capacity
        self._data = np.empty(self._capacity)
        self._size = 0

    def append(self, value) -> None:
        """Append a value to the array."""
        if self._size >= self._capacity:
            # Double capacity when full
            self._capacity *= 2
            new_data = np.empty(self._capacity)
            new_data[: self._size] = self._data[: self._size]
            self._data = new_data

        self._data[self._size] = value
        self._size += 1

    @property
    def data(self) -> np.ndarray:
        """Return view of actual data"""
        return self._data[: self._size]

    def to_result(self, tag: MetricTagT, header: str, unit: str) -> MetricResult:
        """Compute metric stats with zero-copy"""

        arr = self.data
        p1, p5, p25, p50, p75, p90, p95, p99 = np.percentile(
            arr, [1, 5, 25, 50, 75, 90, 95, 99]
        )
        return MetricResult(
            tag=tag,
            header=header,
            unit=unit,
            min=np.min(arr),
            max=np.max(arr),
            avg=float(np.mean(arr)),
            std=float(np.std(arr)),
            p1=p1,
            p5=p5,
            p25=p25,
            p50=p50,
            p75=p75,
            p90=p90,
            p95=p95,
            p99=p99,
            count=self._size,
        )
