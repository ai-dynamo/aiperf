# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from prometheus_client.metrics_core import Metric
from prometheus_client.parser import text_string_to_metric_families

from aiperf.common.enums import PrometheusMetricType
from aiperf.common.environment import Environment
from aiperf.common.mixins import BaseMetricsCollectorMixin
from aiperf.common.mixins.base_metrics_collector_mixin import (
    FetchResult,
    HttpTraceTiming,
)
from aiperf.common.models import ErrorDetails
from aiperf.common.models.server_metrics_models import (
    MetricFamily,
    MetricSample,
    ServerMetricsRecord,
)

__all__ = ["ServerMetricsDataCollector"]


@dataclass(slots=True)
class HistogramData:
    """Temporary histogram data accumulator during parsing.

    Lightweight dataclass for accumulating histogram bucket, sum, and count
    data during Prometheus metric parsing.
    """

    buckets: dict[str, float] = field(default_factory=dict)
    """Mapping of bucket upper bounds (le values) to cumulative counts."""

    sum: float | None = None
    """Cumulative sum of all observed values."""

    count: float | None = None
    """Total count of observations."""

    @property
    def valid(self) -> bool:
        """Check if the histogram data is valid (has buckets, sum, or count).

        Returns:
            True if at least one of buckets/sum/count is populated, False if all empty.
            Used to filter out empty histograms after parsing.
        """
        return len(self.buckets) > 0 or self.sum is not None or self.count is not None

    def to_metric_sample(
        self, labels: tuple[tuple[str, str], ...] | None = None
    ) -> MetricSample:
        """Convert to MetricSample for final record.

        Transforms the accumulated histogram data into a ``MetricSample`` and
        converts label tuple to dict format.

        Args:
            labels: Optional tuple of (key, value) pairs for metric labels

        Returns:
            MetricSample with histogram fields (buckets, sum, count) populated
        """
        return MetricSample(
            labels=dict(labels) if labels else None,
            buckets=self.buckets,
            sum=self.sum,
            count=self.count,
        )


class ServerMetricsDataCollector(BaseMetricsCollectorMixin[ServerMetricsRecord]):
    """Collects server metrics from Prometheus-compatible endpoints.

    Async collector that fetches metrics from Prometheus endpoints and converts them
    to ServerMetricsRecord objects. Extends BaseMetricsCollectorMixin for HTTP
    collection patterns and uses prometheus_client for robust metric parsing.

    Features:
        - Async HTTP collection with aiohttp
        - Prometheus exposition format parsing
        - Callback-based record delivery
        - Error handling with ErrorDetails

    Args:
        endpoint_url: URL of the Prometheus metrics endpoint (e.g., "http://localhost:8081/metrics")
        collection_interval: Interval in seconds between metric collections (default from environment)
        reachability_timeout: Timeout in seconds for endpoint reachability checks (default from environment)
        record_callback: Optional async callback to receive collected records.
            Signature: async (records: list[ServerMetricsRecord], collector_id: str) -> None
        error_callback: Optional async callback to receive collection errors.
            Signature: async (error: ErrorDetails, collector_id: str) -> None
        collector_id: Unique identifier for this collector instance
    """

    def __init__(
        self,
        endpoint_url: str,
        *,
        collection_interval: float | None = None,
        reachability_timeout: float | None = None,
        record_callback: Callable[[list[ServerMetricsRecord], str], Awaitable[None]] | None = None,
        error_callback: Callable[[ErrorDetails, str], Awaitable[None]] | None = None,
        collector_id: str = "server_metrics_collector",
    ) -> None:  # fmt: skip
        super().__init__(
            endpoint_url=endpoint_url,
            collection_interval=collection_interval or Environment.SERVER_METRICS.COLLECTION_INTERVAL,
            reachability_timeout=reachability_timeout or Environment.SERVER_METRICS.REACHABILITY_TIMEOUT,
            record_callback=record_callback,
            error_callback=error_callback,
            id=collector_id,
        )  # fmt: skip

        # Keep track of metrics we have already seen (logged once) to avoid spamming the logs
        self._seen_metadata_metrics = set()
        self._seen_summary_metrics = set()

    async def _collect_and_process_metrics(self) -> None:
        """Collect metrics from Prometheus endpoint and process them into ServerMetricsRecord objects.

        Implements the abstract method from BaseMetricsCollectorMixin.

        Orchestrates the full collection flow:
        1. Fetches raw metrics data from Prometheus endpoint (via mixin's _fetch_metrics_text)
        2. Parses Prometheus-format data into ServerMetricsRecord objects
        3. Sends records via callback (via mixin's _send_records_via_callback)

        Uses HTTP trace timing to capture precise request lifecycle timestamps for
        accurate correlation with client request timestamps.

        Raises:
            Exception: Any exception from fetch or parse is logged and re-raised
        """
        fetch_result = await self._fetch_metrics_text()
        record = self._parse_metrics_to_records(fetch_result)
        if record:
            await self._send_records_via_callback([record])

    def _parse_metrics_to_records(
        self, fetch_result: FetchResult
    ) -> ServerMetricsRecord | None:
        """Parse Prometheus metrics text into ServerMetricsRecord objects.

        Processes Prometheus exposition format metrics:
        1. Parses metric families using prometheus_client parser
        2. Groups metrics by type (counter, gauge, histogram)
        3. De-duplicates by label combination (last value wins)
        4. Structures histogram data

        Args:
            fetch_result: FetchResult containing raw metrics text and trace timing data

        Returns:
            ServerMetricsRecord | None: ServerMetricsRecord containing complete snapshot.
                Returns None if fetch_result.text is empty
        Raises:
            ValueError: If parsing fails
        """
        if not fetch_result.text or not fetch_result.text.strip():
            return None

        metrics_dict = self._build_metrics_dict(fetch_result.text)

        # Suppress empty snapshots to reduce I/O noise
        if not metrics_dict:
            return None

        trace_timing = fetch_result.trace_timing
        return ServerMetricsRecord(
            timestamp_ns=self._resolve_timestamp_ns(trace_timing),
            endpoint_latency_ns=trace_timing.latency_ns if trace_timing else None,
            endpoint_url=self._endpoint_url,
            metrics=metrics_dict,
            request_sent_ns=trace_timing.start_ns if trace_timing else None,
            first_byte_ns=trace_timing.first_byte_ns if trace_timing else None,
            is_duplicate=fetch_result.is_duplicate,
        )

    @staticmethod
    def _resolve_timestamp_ns(trace_timing: HttpTraceTiming | None) -> int:
        """Return the best available server snapshot timestamp in nanoseconds.

        Uses first_byte_ns from trace timing when available (best approximation of
        server snapshot time); otherwise falls back to current wall-clock time.
        """
        if trace_timing and trace_timing.first_byte_ns is not None:
            return trace_timing.first_byte_ns
        return time.time_ns()

    def _build_metrics_dict(self, text: str) -> dict[str, MetricFamily]:
        """Parse Prometheus text and assemble the per-family metrics dict."""
        metrics_dict: dict[str, MetricFamily] = {}
        try:
            for family in text_string_to_metric_families(text):
                entry = self._process_family(family)
                if entry is not None:
                    metrics_dict[family.name] = entry
        except ValueError as e:
            self.warning(f"Failed to parse Prometheus metrics - invalid format: {e!r}")
            raise
        return metrics_dict

    def _process_family(self, family: Metric) -> MetricFamily | None:
        """Classify a Prometheus metric family and return its MetricFamily entry.

        Returns None for metadata families, unsupported summary metrics, unknown
        metric types, or families whose samples all get filtered out.
        """
        if self._is_metadata_family(family.name):
            self._log_skipped_metadata(family.name)
            return None

        metric_type = PrometheusMetricType(family.type)
        samples = self._extract_samples(family, metric_type)
        if not samples:
            return None

        return MetricFamily(
            type=metric_type,
            description=family.documentation or "",
            samples=samples,
        )

    @staticmethod
    def _is_metadata_family(name: str) -> bool:
        """Return True for `_created`/`_uptime` timestamp families that aren't metric data."""
        return (
            name.endswith("_created") or name.endswith("_uptime") or "_uptime_" in name
        )

    def _log_skipped_metadata(self, name: str) -> None:
        """Log a skipped metadata metric family once, then suppress further logs."""
        if name not in self._seen_metadata_metrics:
            self.debug(lambda n=name: f"Skipping metadata metric: {n}")
            self._seen_metadata_metrics.add(name)

    def _extract_samples(
        self, family: Metric, metric_type: PrometheusMetricType
    ) -> list[MetricSample]:
        """Extract samples for a metric family based on its type.

        Returns an empty list for unsupported summary metrics and unknown types, so
        callers can uniformly skip them.
        """
        if metric_type == PrometheusMetricType.HISTOGRAM:
            return self._process_histogram_family(family)
        if metric_type == PrometheusMetricType.SUMMARY:
            # Summary metrics are not supported - they compute quantiles
            # cumulatively over server lifetime, not per-benchmark period
            if family.name not in self._seen_summary_metrics:
                self.info(
                    lambda name=family.name: f"Skipping unsupported summary metric: {name}"
                )
                self._seen_summary_metrics.add(family.name)
            return []
        if metric_type in (
            PrometheusMetricType.COUNTER,
            PrometheusMetricType.GAUGE,
            PrometheusMetricType.UNKNOWN,
        ):
            return self._process_simple_family(family)
        self.warning(f"Unsupported metric type: {metric_type}")
        return []

    def _process_simple_family(self, family: Metric) -> list[MetricSample]:
        """Process counter, gauge, or untyped metrics with de-duplication.

        Extracts all samples from a metric family and de-duplicates by label set.
        When multiple samples have identical labels (shouldn't happen in valid
        Prometheus output), keeps the last value encountered.

        Filters out invalid values (None or infinity) which can occur with
        missing data or uninitialized metrics.

        Args:
            family: Prometheus metric family from prometheus_client parser containing
                   metric type, name, and samples

        Returns:
            List of MetricSample objects with de-duplicated values (last value wins
            for duplicate label sets). Returns empty list if all samples filtered out.
        """
        samples_by_labels: dict[tuple, float] = {}

        for sample in family.samples:
            # Skip samples with None or infinity values (can happen with NaN or missing data)
            if sample.value is None or sample.value == float("inf"):
                continue
            label_key = tuple(sorted(sample.labels.items()))
            samples_by_labels[label_key] = sample.value

        return [
            MetricSample(labels=dict(label_tuple) if label_tuple else None, value=value)
            for label_tuple, value in samples_by_labels.items()
        ]

    def _process_histogram_family(self, family: Metric) -> list[MetricSample]:
        """Process histogram metrics into structured format.

        Prometheus histograms are represented as multiple metric samples:
        - metric_name_bucket{le="0.1"}: Cumulative count <= 0.1
        - metric_name_bucket{le="1.0"}: Cumulative count <= 1.0
        - metric_name_sum: Sum of all observed values
        - metric_name_count: Total observation count

        This function groups these related samples by their base labels (excluding "le")
        and assembles them into a single MetricSample per label set with buckets dict.

        Args:
            family: Prometheus histogram metric family from prometheus_client parser

        Returns:
            List of MetricSample objects where each contains:
            - buckets: Dict mapping le bounds to cumulative counts
            - sum: Total sum of observations
            - count: Total observation count
            - labels: Base labels (excluding "le" which is part of bucket structure)
        """
        histograms: dict[tuple, HistogramData] = defaultdict(HistogramData)

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

        return [
            hist.to_metric_sample(label_tuple)
            for label_tuple, hist in histograms.items()
            if hist.valid
        ]
