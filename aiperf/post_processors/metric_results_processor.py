# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections import deque
from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd

from aiperf.common.config.user_config import UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import ResultsProcessorType
from aiperf.common.enums.metric_enums import (
    MetricDictValueTypeT,
    MetricType,
    MetricValueTypeT,
)
from aiperf.common.factories import ResultsProcessorFactory
from aiperf.common.models.record_models import MetricResult
from aiperf.common.protocols import ResultsProcessorProtocol
from aiperf.common.types import MetricTagT
from aiperf.metrics import BaseAggregateMetric
from aiperf.metrics.base_metric import BaseMetric
from aiperf.metrics.metric_dicts import MetricRecordDict, MetricResultsDict
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor


@implements_protocol(ResultsProcessorProtocol)
@ResultsProcessorFactory.register(ResultsProcessorType.METRIC_RESULTS)
class MetricResultsProcessor(BaseMetricsProcessor):
    """Processor for metric results.

    This is the final stage of the metrics processing pipeline, and is done is a unified manner by the RecordsManager.
    It is responsible for processing the results and returning them to the RecordsManager, as well as summarizing the results.
    """

    def __init__(self, user_config: UserConfig, **kwargs: Any):
        super().__init__(user_config=user_config, **kwargs)
        # For derived metrics, we don't care about splitting up the error metrics
        self.derive_funcs: dict[
            MetricTagT, Callable[[MetricResultsDict], MetricValueTypeT]
        ] = {
            metric.tag: metric.derive_value  # type: ignore
            for metric in self._setup_metrics(MetricType.DERIVED)
        }

        # Create the results dict, which will be used to store the results of non-derived metrics,
        # and then be updated with the derived metrics.
        self._results: MetricResultsDict = MetricResultsDict()

        # Get all of the metric classes.
        _all_metric_classes: list[type[BaseMetric]] = MetricRegistry.all_classes()

        # Pre-cache the types for the metrics.
        self._tags_to_types: dict[MetricTagT, MetricType] = {
            metric.tag: metric.type for metric in _all_metric_classes
        }

        # Pre-cache the instances for the metrics.
        self._instances_map: dict[MetricTagT, BaseMetric] = {
            tag: MetricRegistry.get_instance(tag) for tag in MetricRegistry.all_tags()
        }

        # Pre-cache the aggregate functions for the aggregate metrics.
        self._tags_to_aggregate_funcs: dict[
            MetricTagT, Callable[[MetricResultsDict], MetricValueTypeT]
        ] = {
            metric.tag: MetricRegistry.get_instance(metric.tag).aggregate_value  # type: ignore
            for metric in _all_metric_classes
            if metric.type == MetricType.AGGREGATE
        }

    async def process_result(self, incoming_metrics: MetricRecordDict) -> None:
        """Process a result from the metric record processor."""
        if self.is_trace_enabled:
            self.trace(f"Processing incoming metrics: {incoming_metrics}")

        for tag, value in incoming_metrics.items():
            try:
                metric_type = self._tags_to_types[tag]
                if metric_type == MetricType.RECORD:
                    if tag not in self._results:
                        self._results[tag] = deque()
                    self._results[tag].append(value)  # type: ignore

                elif metric_type == MetricType.AGGREGATE:
                    metric: BaseAggregateMetric = self._instances_map[tag]  # type: ignore
                    metric.aggregate_value(value)
                    self._results[tag] = metric.current_value

                else:
                    raise ValueError(f"Metric '{tag}' is not a valid metric type")
            except Exception as e:
                self.warning(f"Error processing metric '{tag}': {e}")

        if self.is_trace_enabled:
            self.trace(f"Results after processing incoming metrics: {self._results}")

    async def summarize(self) -> list[MetricResult]:
        """Summarize the results.

        This will compute the values for the derived metrics, and then create the MetricResult objects for each metric.
        """
        # Compute the values for the derived metrics, and store them in the results dict.
        for tag, derive_func in self.derive_funcs.items():
            self._results[tag] = derive_func(self._results)

        # Compute and return the metric results.
        return [
            self._create_metric_result(tag, values)
            for tag, values in self._results.items()
        ]

    def _create_metric_result(
        self, tag: MetricTagT, values: MetricDictValueTypeT
    ) -> MetricResult:
        """Create a MetricResult from a the current values of a metric."""

        metric_class = self._instances_map[tag]

        if isinstance(values, int | float):
            return MetricResult(
                tag=metric_class.tag,
                header=metric_class.header,
                unit=str(metric_class.unit),
                avg=values,
                count=1,
            )

        if isinstance(values, Iterable):
            series = pd.Series(values, dtype=metric_class.value_type.dtype)
            quantiles = series.quantile(
                [0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
            )
            return MetricResult(
                tag=metric_class.tag,
                header=metric_class.header,
                unit=str(metric_class.unit),
                avg=series.mean(),
                min=series.min(),
                max=series.max(),
                p1=quantiles[0.01],
                p5=quantiles[0.05],
                p25=quantiles[0.25],
                p50=quantiles[0.50],
                p75=quantiles[0.75],
                p90=quantiles[0.90],
                p95=quantiles[0.95],
                p99=quantiles[0.99],
                std=series.std(),
                count=len(series),
            )

        raise ValueError(f"Unexpected values type: {type(values)}")
