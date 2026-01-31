# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import (
    MetricDictValueTypeT,
    MetricType,
    MetricUnitT,
    MetricValueTypeT,
    MetricValueTypeVarT,
)
from aiperf.common.environment import Environment
from aiperf.common.exceptions import MetricTypeError, MetricUnitError, NoMetricValue
from aiperf.common.growable_array import GrowableArray
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
        if value is None:
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
        self,
        registry: "type[MetricRegistry]",
        show_internal: bool = False,
        show_experimental: bool = False,
    ) -> dict[str, MetricValue]:
        """Convert to display units with filtering applied.
        NOTE: This will not include metrics with the `NO_INDIVIDUAL_RECORDS` flag.

        Args:
            registry: MetricRegistry class for looking up metric definitions
            show_internal: If True, include experimental/internal metrics

        Returns:
            Dictionary of {tag: MetricValue} for export
        """
        from aiperf.common.enums import MetricFlags

        result = {}
        for tag, value in self.items():
            try:
                metric_class = registry.get_class(tag)
            except MetricTypeError:
                _logger.warning(f"Metric {tag} not found in registry")
                continue

            if (
                metric_class.has_flags(MetricFlags.EXPERIMENTAL)
                and not show_experimental
            ):
                continue
            if metric_class.has_flags(MetricFlags.INTERNAL) and not show_internal:
                continue
            if metric_class.has_flags(MetricFlags.NO_INDIVIDUAL_RECORDS):
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
    Uses GrowableArray internally for efficient storage with automatic growth.
    """

    def __init__(
        self, initial_capacity: int = Environment.METRICS.ARRAY_INITIAL_CAPACITY
    ):
        """Initialize the array with the given initial capacity."""
        self._array = GrowableArray(
            initial_capacity=initial_capacity,
            dtype=np.float64,
            track_sum=True,
        )

    def extend(self, values: list[MetricValueTypeVarT]) -> None:
        """Extend the array with a list of values."""
        self._array.extend(np.asarray(values, dtype=np.float64))

    def append(self, value: MetricValueTypeVarT) -> None:
        """Append a value to the array."""
        self._array.append(value)

    @property
    def sum(self) -> MetricValueTypeVarT:
        """Get the sum of the array."""
        return self._array.sum  # type: ignore

    @property
    def data(self) -> np.ndarray:
        """Return view of actual data."""
        return self._array.data

    @property
    def capacity(self) -> int:
        """Return current capacity."""
        return self._array.capacity

    def __len__(self) -> int:
        """Return number of elements."""
        return len(self._array)

    def to_result(self, tag: MetricTagT, header: str, unit: str) -> MetricResult:
        """Compute metric stats with zero-copy"""

        arr = self.data
        p1, p5, p10, p25, p50, p75, p90, p95, p99 = np.percentile(
            arr, [1, 5, 10, 25, 50, 75, 90, 95, 99]
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
            p10=p10,
            p25=p25,
            p50=p50,
            p75=p75,
            p90=p90,
            p95=p95,
            p99=p99,
            count=len(self._array),
        )
