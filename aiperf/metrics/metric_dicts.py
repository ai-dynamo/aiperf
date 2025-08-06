# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING

from aiperf.common.enums import MetricType
from aiperf.common.enums.metric_enums import (
    MetricDictValueTypeT,
    MetricUnitT,
    MetricValueTypeT,
)
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
