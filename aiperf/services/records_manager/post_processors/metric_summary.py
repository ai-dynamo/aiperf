# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

from aiperf.common.enums import EndpointType, MetricType, PostProcessorType
from aiperf.common.enums.endpoints_enums import EndpointType
from aiperf.common.enums.metric_enums import MetricTag
from aiperf.common.factories import PostProcessorFactory
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import MetricResult, ParsedResponseRecord
from aiperf.services.records_manager.metrics.base_metric import BaseMetric


@PostProcessorFactory.register(PostProcessorType.METRIC_SUMMARY)
class MetricSummary(AIPerfLoggerMixin):
    """
    MetricSummary is a post-processor that generates a summary of metrics from the records.
    It processes the records to extract relevant metrics and returns them in a structured format.
    """

    def __init__(self, endpoint_type: EndpointType | None = None, **kwargs):
        super().__init__(endpoint_type=endpoint_type, **kwargs)
        self.debug("Initializing MetricSummary post-processor")

        # Only include latency and throughput metrics for embeddings endpoint
        allowed_tags = None
        if (
            endpoint_type is not None
            and endpoint_type == EndpointType.OPENAI_EMBEDDINGS
        ):
            allowed_tags = {
                MetricTag.REQUEST_LATENCY,
                MetricTag.REQUEST_THROUGHPUT,
                MetricTag.BENCHMARK_DURATION,
                MetricTag.REQUEST_COUNT,
                MetricTag.MIN_REQUEST,
                MetricTag.MAX_RESPONSE,
            }

        self._metrics = []
        for metric_cls in BaseMetric.get_all().values():
            self._metrics.append(metric_cls())

    def process_record(self, record: ParsedResponseRecord) -> None:
        """Process a single record.

        Classifies and computes metrics in dependency order to ensure correctness.
        The metrics are categorized based on their dependency types:

        1. METRIC_OF_RECORDS:
            - Depend solely on each individual record.
            - Computed first, as they have no dependencies.

        2. METRIC_OF_BOTH:
            - Depend on both:
                - the current record, and
                - previously computed metrics (specifically, METRIC_OF_RECORDS).
            - Computed after all METRIC_OF_RECORDS have been processed.
            - Must not depend on other METRIC_OF_BOTH or METRIC_OF_METRICS.

        3. METRIC_OF_METRICS:
            Computed once after all records have been processed.
            see: :meth:`post_process`
        """
        if not record.valid:
            return

        # METRIC_OF_RECORDS
        for metric in self._metrics:
            if metric.type == MetricType.METRIC_OF_RECORDS:
                metric.update_value(record=record)

        # METRIC_OF_BOTH
        for metric in self._metrics:
            if metric.type == MetricType.METRIC_OF_BOTH:
                metric.update_value(
                    record=record, metrics={m.tag: m for m in self._metrics}
                )

    def post_process(self) -> None:
        """
        Classifies and computes metrics in dependency order to ensure correctness.
        The metrics are categorized based on their dependency types:

        1. METRIC_OF_RECORDS:
            - Computed for each individual record.
            see: :meth:`process_record`

        2. METRIC_OF_BOTH:
            - Computed for each individual record.
            see: :meth:`process_record`

        3. METRIC_OF_METRICS:
            - Computed based only on other metrics (not records).
            - May depend on any combination of:
                - METRIC_OF_RECORDS,
                - METRIC_OF_BOTH,
                - other METRIC_OF_METRICS (if dependency order is respected).
            - Computed using a dependency-resolution loop.

        This process ensures:
            - All metrics are computed exactly once, after dependencies are satisfied.
            - Misconfigured or cyclic dependencies will raise an explicit runtime error.
        """
        # METRIC_OF_METRICS
        # Precompute tags of all metrics already processed
        computed_tags = {
            m.tag
            for m in self._metrics
            if m.type in {MetricType.METRIC_OF_RECORDS, MetricType.METRIC_OF_BOTH}
        }

        remaining = [m for m in self._metrics if m.type == MetricType.METRIC_OF_METRICS]

        # Resolve dependencies: loop until all metrics are computed or a circular dependency is found
        while remaining:
            progress = False
            for metric in remaining[:]:
                # If required dependencies are all satisfied, compute this metric
                if metric.required_metrics.issubset(computed_tags):
                    metric.update_value(metrics={m.tag: m for m in self._metrics})
                    computed_tags.add(metric.tag)
                    remaining.remove(metric)
                    progress = True

            if not progress:
                # Circular dependencies
                missing = {m.tag: m.required_metrics - computed_tags for m in remaining}
                raise ValueError(
                    f"Circular or unsatisfiable dependencies detected in METRIC_OF_METRICS: {missing}"
                )

    def get_results(self) -> list[MetricResult]:
        """Gets the metrics summary."""
        metrics_summary = []

        df = pd.DataFrame({metric.tag: metric.values() for metric in self._metrics})

        for metric in self._metrics:
            res: MetricResult = record_from_dataframe(df, metric)
            metrics_summary.append(res)
        return metrics_summary


def record_from_dataframe(df: pd.DataFrame, metric: BaseMetric) -> MetricResult:
    """Create a Record from a DataFrame."""

    column = df[metric.tag]
    quantiles = column.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])

    return MetricResult(
        tag=metric.tag,
        header=metric.header,
        unit=metric.unit.short_name() if metric.unit else "",
        avg=column.mean(),
        min=column.min(),
        max=column.max(),
        p1=quantiles[0.01],
        p5=quantiles[0.05],
        p25=quantiles[0.25],
        p50=quantiles[0.50],
        p75=quantiles[0.75],
        p90=quantiles[0.90],
        p95=quantiles[0.95],
        p99=quantiles[0.99],
        std=column.std(),
        count=int(column.count()),
        streaming_only=metric.streaming_only,
    )
