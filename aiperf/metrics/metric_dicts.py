# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

from aiperf.common.enums import MetricType
from aiperf.common.enums.metric_enums import (
    MetricDictValueTypeT,
    MetricUnitT,
    MetricValueTypeT,
    MetricValueTypeVarT,
)
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models.record_models import MetricResult
from aiperf.common.types import MetricTagT

if TYPE_CHECKING:
    from aiperf.metrics.base_metric import BaseMetric


MetricDictValueTypeVarT = TypeVar(
    "MetricDictValueTypeVarT", bound="MetricValueTypeT | MetricDictValueTypeT"
)


class BaseMetricDict(
    Generic[MetricDictValueTypeVarT], dict[MetricTagT, MetricDictValueTypeVarT]
):
    """Base class for all metric dicts."""

    def get_or_raise(self, metric: type["BaseMetric"]) -> MetricDictValueTypeT:
        """Get the value of a metric, or raise NoMetricValue if it is not available."""
        value = self.get(metric.tag)
        if not value:
            raise NoMetricValue(f"Metric {metric.tag} is not available for the record.")
        return value

    def get_converted_or_raise(
        self, metric: type["BaseMetric"], other_unit: MetricUnitT
    ) -> float:
        """Get the value of a metric, but converted to a different unit, or raise NoMetricValue if it is not available."""
        return metric.unit.convert_to(other_unit, self.get_or_raise(metric))  # type: ignore


class MetricRecordDict(BaseMetricDict[MetricValueTypeT]):
    """
    A dict of metrics for a single record. This is used to store the current values
    of all metrics that have been computed for a single record.

    This will include:
    - The current value of any `BaseRecordMetric` that has been computed for this record.
    - The new value of any `BaseAggregateMetric` that has been computed for this record.
    - No `BaseDerivedMetric`s will be included.
    """

    pass  # Everything is handled by the BaseMetricDict class.


class MetricResultsDict(BaseMetricDict[MetricDictValueTypeT]):
    """
    A dict of metrics over an entire run. This is used to store the final values
    of all metrics that have been computed for an entire run.

    This will include:
    - All `BaseRecordMetric`s as a MetricArray of their values.
    - The most recent value of each `BaseAggregateMetric`.
    - The value of any `BaseDerivedMetric` that has already been computed.
    """

    def get_converted_or_raise(
        self, metric: type["BaseMetric"], other_unit: MetricUnitT
    ) -> float:
        """Get the value of a metric, but converted to a different unit, or raise NoMetricValue if it is not available."""
        if metric.type == MetricType.RECORD:
            # Record metrics are a MetricArray of values, so we can't convert them directly.
            raise ValueError(
                f"Cannot convert a record metric to a different unit: {metric.tag}"
            )
        return super().get_converted_or_raise(metric, other_unit)


class MetricArray(Generic[MetricValueTypeVarT]):
    """NumPy backed array for metric data.

    This is used to store the values of a metric over time.
    """

    def __init__(self, initial_capacity: int = 10000):
        self._capacity = initial_capacity
        self._data = np.empty(self._capacity)
        self._size = 0
        self._sum: MetricValueTypeVarT = 0  # type: ignore

    def append(self, value: MetricValueTypeVarT) -> None:
        """Append a value to the array."""
        if self._size >= self._capacity:
            # Double capacity when full
            self._capacity *= 2
            new_data = np.empty(self._capacity)
            new_data[: self._size] = self._data[: self._size]
            self._data = new_data

        self._data[self._size] = value
        self._size += 1
        self._sum += value  # type: ignore

    @property
    def sum(self) -> MetricValueTypeVarT:
        """Get the sum of the array."""
        return self._sum

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

class PerGpuMetricArray(Generic[MetricValueTypeVarT]):
    """Stores a MetricArray for each GPU id."""

    def __init__(self):
        self._gpu_arrays: dict[int, MetricArray[MetricValueTypeVarT]] = {}

    def append(self, gpu_id: int, value: MetricValueTypeVarT) -> None:
        """Append a value for a specific GPU."""
        if gpu_id not in self._gpu_arrays:
            self._gpu_arrays[gpu_id] = MetricArray()
        self._gpu_arrays[gpu_id].append(value)

    def get(self, gpu_id: int) -> MetricArray[MetricValueTypeVarT]:
        return self._gpu_arrays[gpu_id]

    def items(self):
        return self._gpu_arrays.items()

    def to_results(self, tag: MetricTagT, header: str, unit: str) -> dict[int, MetricResult]:
        """Return a dict of MetricResult per GPU."""
        return {
            gpu_id: arr.to_result(f"{tag}_gpu{gpu_id}", f"{header} (GPU {gpu_id})", unit)
            for gpu_id, arr in self._gpu_arrays.items()
        }

class MetricTelemetryDict(BaseMetricDict[PerGpuMetricArray]):
    """
    A dict of telemetry metrics, where each value is a PerGpuMetricArray.
    Used to store per-GPU time series data for telemetry metrics (e.g., power usage).
    """
    pass
