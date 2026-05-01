# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC
from typing import ClassVar, Generic, TypeVar, get_args, get_origin

from aiperf.common.enums import MetricFlags, MetricValueTypeVarT
from aiperf.metrics.base_derived_metric import BaseDerivedMetric
from aiperf.metrics.base_record_metric import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricArray, MetricResultsDict

RecordMetricT = TypeVar("RecordMetricT", bound=BaseRecordMetric)


class DerivedSumMetric(
    Generic[MetricValueTypeVarT, RecordMetricT],
    BaseDerivedMetric[MetricValueTypeVarT],
    ABC,
):
    """
    This class defines the base class for derived sum metrics. These metrics are automatically derived from a record metric,
    by returning the sum of the values of the record metric.

    Examples:
    ```python
    class TotalReasoningTokensMetric(DerivedSumMetric[ReasoningTokenCountMetric, int]):
        # ... Metric attributes ...
    ```
    """

    record_metric_type: ClassVar[type[BaseRecordMetric]]
    __is_abstract__: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs):
        # Look through the class hierarchy for the first Generic[Type] definition
        for base in cls.__orig_bases__:  # type: ignore
            if get_origin(base) is not None:
                args = get_args(base)
                if args:
                    # the second argument is the record metric type
                    generic_type = args[1]
                    cls.record_metric_type = generic_type
                    cls.required_metrics = {generic_type.tag}
                    cls.flags = (
                        generic_type.flags
                        if cls.flags is MetricFlags.NONE
                        else cls.flags
                    )
                    cls.unit = generic_type.unit
                    cls.__is_abstract__ = False
                    break

        super().__init_subclass__(**kwargs)

    def _derive_value(self, metric_results: MetricResultsDict) -> MetricValueTypeVarT:
        metric_values = metric_results.get(self.record_metric_type.tag)
        if metric_values is None:
            raise ValueError(
                f"{self.record_metric_type.tag} is missing in the metrics."
            )
        if not isinstance(metric_values, MetricArray):
            raise ValueError(f"{self.record_metric_type.tag} is not a MetricArray.")
        return metric_values.sum
