# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import MetricType
from aiperf.common.enums.metric_enums import (
    MetricDictValueTypeT,
    MetricUnitT,
    MetricValueTypeT,
    MetricValueTypeVarT,
)
from aiperf.common.exceptions import MetricUnitError, NoMetricValue
from aiperf.common.models.record_models import MetricResult, MetricValue
from aiperf.common.types import MetricTagT

if TYPE_CHECKING:
    from aiperf.metrics.base_metric import BaseMetric
    from aiperf.metrics.metric_registry import MetricRegistry


MetricDictValueTypeVarT = TypeVar(
    "MetricDictValueTypeVarT", bound="MetricValueTypeT | MetricDictValueTypeT"
)

_logger = AIPerfLogger(__name__)


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

    def to_display_dict(
        self, registry: "type[MetricRegistry]", show_internal: bool = False
    ) -> dict[str, MetricValue]:
        """Convert to display units with filtering applied.

        Args:
            registry: MetricRegistry class for looking up metric definitions
            show_internal: If True, include experimental/internal metrics

        Returns:
            Dictionary of {tag: {value, unit, header}} for export
        """
        from aiperf.common.enums import MetricFlags

        result = {}
        for tag, value in self.items():
            try:
                metric_class = registry.get_class(tag)
            except Exception:
                continue

            if not show_internal and not metric_class.missing_flags(
                MetricFlags.EXPERIMENTAL | MetricFlags.INTERNAL
            ):
                continue

            display_unit = metric_class.display_unit or metric_class.unit
            if display_unit != metric_class.unit:
                try:
                    if isinstance(value, list):
                        value = [
                            metric_class.unit.convert_to(display_unit, v) for v in value
                        ]
                    else:
                        value = metric_class.unit.convert_to(display_unit, value)
                except MetricUnitError as e:
                    _logger.warning(
                        f"Error converting {tag} from {metric_class.unit} to {display_unit}: {e}"
                    )

            result[tag] = MetricValue(
                value=value,
                unit=str(display_unit),
            )

        return result


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
        """Initialize the array with the given initial capacity."""
        if initial_capacity <= 0:
            raise ValueError("Initial capacity must be greater than 0")
        self._capacity = initial_capacity
        self._data = np.empty(self._capacity)
        self._size = 0
        self._sum: MetricValueTypeVarT = 0  # type: ignore

    def extend(self, values: list[MetricValueTypeVarT]) -> None:
        """Extend the array with a list of values."""
        self._resize_if_needed(len(values))

        end = self._size + len(values)
        self._data[self._size : end] = values
        self._sum += sum(values)  # type: ignore
        self._size = end

    def append(self, value: MetricValueTypeVarT) -> None:
        """Append a value to the array."""
        self._resize_if_needed(1)

        self._data[self._size] = value
        self._size += 1
        self._sum += value  # type: ignore

    def _resize_if_needed(self, additional_size: int) -> None:
        """Resize the array if needed."""
        if self._size + additional_size > self._capacity:
            self._capacity = max(self._capacity * 2, self._size + additional_size)
            new_data = np.empty(self._capacity)
            new_data[: self._size] = self._data[: self._size]
            self._data = new_data

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
