# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC
from typing import ClassVar, Generic

from aiperf.common.enums import MetricValueTypeVarT
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics.base_aggregate_metric import BaseAggregateMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class BaseAggregateCounterMetric(
    Generic[MetricValueTypeVarT], BaseAggregateMetric[MetricValueTypeVarT], ABC
):
    """
    A base class for aggregate counter metrics. These metrics increment a counter for each record.

    Examples:
    ```python
    class RequestCountMetric(BaseAggregateCounterMetric[int]):
        # ... Metric attributes ...
    ```
    """

    __is_abstract__: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs) -> None:
        cls.__is_abstract__ = False
        return super().__init_subclass__(**kwargs)

    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> MetricValueTypeVarT:
        """Return the value of the counter for this record."""
        return 1  # type: ignore

    def _aggregate_value(self, value: MetricValueTypeVarT) -> None:
        """Aggregate the metric value."""
        self._value += value  # type: ignore
