# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from collections import defaultdict
from collections.abc import Awaitable, Callable

from prometheus_client.metrics_core import Metric
from prometheus_client.parser import text_string_to_metric_families

from aiperf.common.enums import PrometheusMetricType
from aiperf.common.environment import Environment
from aiperf.common.mixins import BaseMetricsCollectorMixin
from aiperf.common.models import ErrorDetails
from aiperf.common.models.server_metrics_models import (
    HistogramData,
    MetricFamily,
    MetricSample,
    ServerMetricsRecord,
    SummaryData,
)

__all__ = ["ServerMetricsDataCollector"]


class ServerMetricsDataCollector(BaseMetricsCollectorMixin[ServerMetricsRecord]):
    """Collects server metrics from Prometheus endpoint using async architecture.

    Modern async collector that fetches metrics from Prometheus-compatible endpoints
    and converts them to ServerMetricsRecord objects. Uses BaseMetricsCollectorMixin
    for HTTP collection patterns.
    - Extends BaseMetricsCollectorMixin for async HTTP collection
    - Uses prometheus_client for robust metric parsing
    - Sends ServerMetricsRecord list via callback function

    Args:
        endpoint_url: URL of the Prometheus metrics endpoint (e.g., "http://localhost:8081/metrics")
        collection_interval: Interval in seconds between metric collections (default: 1.0)
        record_callback: Optional async callback to receive collected records.
            Signature: async (records: list[ServerMetricsRecord], collector_id: str) -> None
        error_callback: Optional async callback to receive collection errors.
            Signature: async (error: ErrorDetails, collector_id: str) -> None
        collector_id: Unique identifier for this collector instance
    """

    def __init__(
        self,
        endpoint_url: str,
        collection_interval: float | None = None,
        reachability_timeout: float | None = None,
        record_callback: Callable[[list[ServerMetricsRecord], str], Awaitable[None]]
        | None = None,
        error_callback: Callable[[ErrorDetails, str], Awaitable[None]] | None = None,
        collector_id: str = "server_metrics_collector",
    ) -> None:
        super().__init__(
            endpoint_url=endpoint_url,
            collection_interval=collection_interval
            or Environment.SERVER_METRICS.COLLECTION_INTERVAL,
            reachability_timeout=reachability_timeout
            or Environment.SERVER_METRICS.REACHABILITY_TIMEOUT,
            record_callback=record_callback,
            error_callback=error_callback,
            id=collector_id,
        )

    async def _collect_and_process_metrics(self) -> None:
        """Collect metrics from Prometheus endpoint and process them into ServerMetricsRecord objects.

        Implements the abstract method from BaseMetricsCollectorMixin.

        Orchestrates the full collection flow:
        1. Fetches raw metrics data from Prometheus endpoint (via mixin's _fetch_metrics_text)
        2. Parses Prometheus-format data into ServerMetricsRecord objects
        3. Sends records via callback (via mixin's _send_records_via_callback)

        Raises:
            Exception: Any exception from fetch or parse is logged and re-raised
        """
        start_perf_ns = time.perf_counter_ns()
        metrics_data = await self._fetch_metrics_text()
        latency_ns = time.perf_counter_ns() - start_perf_ns
        records = self._parse_metrics_to_records(metrics_data, latency_ns)
        await self._send_records_via_callback(records)

    def _parse_metrics_to_records(
        self, metrics_data: str, latency_ns: int
    ) -> list[ServerMetricsRecord]:
        """Parse Prometheus metrics text into ServerMetricsRecord objects.

        Processes Prometheus exposition format metrics:
        1. Parses metric families using prometheus_client parser
        2. Groups metrics by type (counter, gauge, histogram, summary)
        3. De-duplicates by label combination (last value wins)
        4. Structures histogram and summary data

        Args:
            metrics_data: Raw metrics text from Prometheus endpoint in Prometheus format
            latency_ns: Nanoseconds it took to collect the metrics from the endpoint

        Returns:
            list[ServerMetricsRecord]: List with single ServerMetricsRecord containing complete snapshot.
                Returns empty list if metrics_data is empty or parsing fails.
        """
        if not metrics_data.strip():
            return []

        current_timestamp_ns = time.time_ns()
        metrics_dict: dict[str, MetricFamily] = {}

        try:
            for family in text_string_to_metric_families(metrics_data):
                # Skip _created metrics - these are timestamps indicating when the parent
                # histogram/summary/counter was created, not actual metric data
                if family.name.endswith("_created"):
                    continue

                metric_type = PrometheusMetricType(family.type)
                match metric_type:
                    case PrometheusMetricType.HISTOGRAM:
                        samples = self._process_histogram_family(family)
                    case PrometheusMetricType.SUMMARY:
                        samples = self._process_summary_family(family)
                    case (
                        PrometheusMetricType.COUNTER
                        | PrometheusMetricType.GAUGE
                        | PrometheusMetricType.UNKNOWN
                    ):
                        samples = self._process_simple_family(family)
                    case _:
                        self.warning(f"Unsupported metric type: {metric_type}")
                        continue

                metrics_dict[family.name] = MetricFamily(
                    type=metric_type,
                    help=family.documentation or "",
                    samples=samples,
                )
        except ValueError as e:
            self.warning(f"Failed to parse Prometheus metrics - invalid format: {e}")
            raise

        record = ServerMetricsRecord(
            timestamp_ns=current_timestamp_ns,
            endpoint_latency_ns=latency_ns,
            endpoint_url=self._endpoint_url,
            metrics=metrics_dict,
        )

        return [record]

    def _process_simple_family(self, family: Metric) -> list[MetricSample]:
        """Process counter, gauge, or untyped metrics with de-duplication.

        Args:
            family: Prometheus metric family

        Returns:
            List of MetricSample objects with de-duplicated values (last wins)
        """
        samples_by_labels: dict[tuple, float] = {}

        for sample in family.samples:
            label_key = tuple(sorted(sample.labels.items()))
            samples_by_labels[label_key] = sample.value

        return [
            MetricSample(labels=dict(label_tuple) if label_tuple else None, value=value)
            for label_tuple, value in samples_by_labels.items()
        ]

    def _process_histogram_family(self, family: Metric) -> list[MetricSample]:
        """Process histogram metrics into structured format.

        Args:
            family: Prometheus histogram metric family

        Returns:
            List of MetricSample objects with HistogramData
        """
        histograms: dict[tuple, HistogramData] = defaultdict(
            lambda: HistogramData(buckets={}, sum=None, count=None)
        )

        for sample in family.samples:
            base_labels = {k: v for k, v in sample.labels.items() if k != "le"}
            label_key = tuple(sorted(base_labels.items()))

            if sample.name.endswith("_bucket"):
                le_value = sample.labels.get("le", "+Inf")
                histograms[label_key].buckets[le_value] = sample.value
            elif sample.name.endswith("_sum"):
                histograms[label_key].sum = sample.value
            elif sample.name.endswith("_count"):
                histograms[label_key].count = sample.value

        samples = []
        for label_tuple, hist_data in histograms.items():
            if hist_data.sum is None or hist_data.count is None:
                self.debug(
                    lambda hist=hist_data: f"Skipping incomplete histogram: {hist}"
                )
                continue

            samples.append(
                MetricSample(
                    labels=dict(label_tuple) if label_tuple else None,
                    histogram=hist_data,
                )
            )

        return samples

    def _process_summary_family(self, family: Metric) -> list[MetricSample]:
        """Process summary metrics into structured format.

        Args:
            family: Prometheus summary metric family

        Returns:
            List of MetricSample objects with SummaryData
        """
        summaries: dict[tuple, SummaryData] = defaultdict(
            lambda: SummaryData(quantiles={}, sum=None, count=None)
        )

        for sample in family.samples:
            base_labels = {k: v for k, v in sample.labels.items() if k != "quantile"}
            label_key = tuple(sorted(base_labels.items()))

            if sample.name == family.name:
                quantile = sample.labels.get("quantile", "0")
                summaries[label_key].quantiles[quantile] = sample.value
            elif sample.name.endswith("_sum"):
                summaries[label_key].sum = sample.value
            elif sample.name.endswith("_count"):
                summaries[label_key].count = sample.value

        samples = []
        for label_tuple, summary_data in summaries.items():
            if summary_data.sum is None or summary_data.count is None:
                continue

            samples.append(
                MetricSample(
                    labels=dict(label_tuple) if label_tuple else None,
                    summary=summary_data,
                )
            )

        return samples
