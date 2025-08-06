# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Generic

from aiperf.common.enums import MetricType, MetricValueTypeVarT
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics.base_metric import BaseMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


class BaseAggregateMetric(
    Generic[MetricValueTypeVarT], BaseMetric[MetricValueTypeVarT], ABC
):
    """A base class for aggregate metrics. These metrics keep track of a value or list of values over time.

    This metric type is unique in the fact that it is split into 2 distinct phases of processing, in order to support distributed processing.

    For each distributed RecordProcessor, an instance of this class is created. This instance is passed the record and the existing record metrics,
    and is responsible for returning the individual value for that record. It should not use or update the aggregate value here.

    The ResultsProcessor creates a singleton instance of this class, which will be used to aggregate the results from the distributed
    RecordProcessors. It calls the `_aggregate_value` method, which each metric class must implement to define how values from different
    processes are aggregated, such as summing the values, or taking the min/max/average, etc.

    Examples:
    ```python
    class RequestCountMetric(BaseAggregateMetric[int]):
        # ... Metric attributes ...

        def _parse_record(self, record: ParsedResponseRecord, record_metrics: MetricRecordDict) -> int:
            # We just return 1 since we are tracking the total count, and this is a single request.
            return 1

        def _aggregate_value(self, value: int) -> None:
            # We add the value to the aggregate value.
            self._value += value
    ```
    """

    type = MetricType.AGGREGATE

    def __init__(self, default_value: MetricValueTypeVarT | None = None) -> None:
        """Initialize the metric with optionally with a default value. If no default value is provided,
        the default value is automatically set based on the value type."""
        self._value: MetricValueTypeVarT = (  # type: ignore
            default_value
            if default_value is not None
            else self.value_type.default_factory()
        )
        self.aggregate_value: Callable[[MetricValueTypeVarT], None] = (
            self._aggregate_value
        )
        super().__init__()

    @property
    def current_value(self) -> MetricValueTypeVarT:
        """Get the current value of the metric."""
        return self._value

    def parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> MetricValueTypeVarT:
        """Parse the record and return the individual value.

        Raises:
            ValueError: If the metric cannot be computed for the given inputs.
        """
        self._require_valid_record(record)
        self._check_metrics(record_metrics)
        return self._parse_record(record, record_metrics)

    @abstractmethod
    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> MetricValueTypeVarT:
        """Parse the record and *return* the individual value base on this record, and this record alone. This
        method is implemented by subclasses.

        NOTE: Do not use or update the aggregate value here.

        This method is called after the required metrics are checked, so it can assume that the required metrics are available.
        This method is called after the record is checked, so it can assume that the record is valid.

        Raises:
            ValueError: If the metric cannot be computed for the given inputs.
        """
        raise NotImplementedError("Subclasses must implement this method")

    # NOTE: This method does not return a value on purpose, as a hint to the user that the
    #       internal value is supposed to be updated.
    @abstractmethod
    def _aggregate_value(self, value: MetricValueTypeVarT) -> None:
        """Aggregate the metric value. This method is implemented by subclasses.

        This method is called with the result value from the `_parse_record` method, from each distributed record processor.
        It is the responsibility of each metric class to implement how values from different processes are aggregated, such
        as summing the values, or taking the min/max/average, etc.

        NOTE: The order of the values is not guaranteed.
        """
        raise NotImplementedError("Subclasses must implement this method")
