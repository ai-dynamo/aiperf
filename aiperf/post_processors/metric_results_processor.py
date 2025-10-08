# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from typing import Any

from aiperf.common.config import UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import MetricType
from aiperf.common.enums.metric_enums import MetricDictValueTypeT, MetricValueTypeT
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import MetricResult
from aiperf.common.protocols import ResultsProcessorProtocol
from aiperf.common.types import MetricTagT
from aiperf.metrics import BaseAggregateMetric
from aiperf.metrics.base_metric import BaseMetric
from aiperf.metrics.metric_dicts import MetricArray, MetricRecordDict, MetricResultsDict
from aiperf.metrics.metric_registry import MetricRegistry
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor


@implements_protocol(ResultsProcessorProtocol)
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
                        self._results[tag] = MetricArray()
                    if isinstance(value, list):
                        # NOTE: Right now we only support list-based metrics by extending the array.
                        #       In the future, we possibly could support having nested arrays.
                        self._results[tag].extend(value)  # type: ignore
                    else:
                        self._results[tag].append(value)  # type: ignore

                elif metric_type == MetricType.AGGREGATE:
                    metric: BaseAggregateMetric = self._instances_map[tag]  # type: ignore
                    metric.aggregate_value(value)
                    self._results[tag] = metric.current_value

                else:
                    raise ValueError(f"Metric '{tag}' is not a valid metric type")
            except NoMetricValue as e:
                self.debug(f"No metric value for metric '{tag}': {e!r}")
            except Exception as e:
                self.warning(f"Error processing metric '{tag}': {e!r}")

        if self.is_trace_enabled:
            self.trace(f"Results after processing incoming metrics: {self._results}")

    async def update_derived_metrics(self) -> None:
        """Computes the values for the derived metrics, and stores them in the results dict."""
        for tag, derive_func in self.derive_funcs.items():
            try:
                self._results[tag] = derive_func(self._results)
            except NoMetricValue as e:
                self.debug(f"No metric value for derived metric '{tag}': {e!r}")
            except Exception as e:
                self.warning(f"Error deriving metric '{tag}': {e!r}")

    async def summarize(self) -> list[MetricResult]:
        """Summarize the results.

        This will compute the values for the derived metrics, and then create the MetricResult objects for each metric.
        """
        await self.update_derived_metrics()

        # Compute and return the metric results.
        return [
            self._create_metric_result(tag, values)
            for tag, values in self._results.items()
        ]

    async def full_metrics(self) -> MetricResultsDict:
        """Returns the full metrics dict, including the derived metrics."""
        await self.update_derived_metrics()
        return self._results

    def _create_metric_result(
        self, tag: MetricTagT, values: MetricDictValueTypeT
    ) -> MetricResult:
        """Create a MetricResult from a the current values of a metric."""

        metric_class = self._instances_map[tag]

        if isinstance(values, MetricArray):
            return values.to_result(tag, metric_class.header, str(metric_class.unit))

        if isinstance(values, int | float):
            return MetricResult(
                tag=metric_class.tag,
                header=metric_class.header,
                unit=str(metric_class.unit),
                avg=values,
                count=1,
            )

        raise ValueError(f"Unexpected values type: {type(values)}")
