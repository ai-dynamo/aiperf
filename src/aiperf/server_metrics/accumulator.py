# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task
from aiperf.common.messages import RealtimeServerMetricsMessage
from aiperf.plugin.enums import UIType

if TYPE_CHECKING:
    from aiperf.common.protocols import PubClientProtocol
    from aiperf.config import BenchmarkRun
from aiperf.common.constants import (
    MILLIS_PER_SECOND,
    NANOS_PER_MILLIS,
    NANOS_PER_SECOND,
)
from aiperf.common.enums import PrometheusMetricType, ServerMetricsFormat
from aiperf.common.exceptions import DataExporterDisabled, PostProcessorDisabled
from aiperf.common.models import MetricResult
from aiperf.common.models.error_models import ErrorDetailsCount
from aiperf.common.models.server_metrics_models import (
    CounterMetricData,
    GaugeMetricData,
    HistogramMetricData,
    ServerMetricsEndpointInfo,
    ServerMetricsEndpointSummary,
    ServerMetricsRecord,
    ServerMetricsResults,
    TimeRangeFilter,
)
from aiperf.exporters.utils import normalize_endpoint_display
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor
from aiperf.server_metrics.export_stats import compute_stats
from aiperf.server_metrics.parquet_exporter import ServerMetricsParquetExporter
from aiperf.server_metrics.storage import ServerMetricsHierarchy


class ServerMetricsAccumulator(BaseMetricsProcessor):
    """Process individual ServerMetricsRecord objects into hierarchical storage.

    Results processor that accumulates server metrics from Prometheus endpoints
    and computes comprehensive statistics. Organizes data hierarchically by
    endpoint → metric → time series, supporting multi-endpoint profiling.

    Metric type support:
    - Gauge metrics: Point-in-time values (e.g., cache usage, queue depth)
      → Statistics: avg, min, max, std, percentiles
    - Counter metrics: Cumulative totals (e.g., total requests, total bytes)
      → Delta calculation from reference point + rate statistics
    - Histogram metrics: Bucket distributions (e.g., request latencies)
      → Count/sum rates + estimated percentiles using polynomial algorithm

    Time filtering:
    - Warmup period exclusion via start_ns (ignores metrics before profiling)
    - End buffer exclusion via end_ns (ignores metrics after profiling)
    - Reference point for deltas: last snapshot before start_ns (baseline)
    - Per-endpoint filters handle different collection timelines

    Optional timeslice analysis:
    - When slice_duration configured, computes windowed statistics
    - Enables analysis of metric variation over time (e.g., rate spikes)
    - All timeslices have identical duration for fair comparison

    Args:
        user_config: User configuration including server_metrics settings
        **kwargs: Additional arguments passed to base class

    Raises:
        PostProcessorDisabled: If --no-server-metrics flag is set

    Example:
        >>> from aiperf.config import BenchmarkConfig
        >>> from aiperf.common.models import ServerMetricsRecord, MetricFamily, MetricSample
        >>> # Create accumulator
        >>> config = BenchmarkConfig(...)
        >>> accumulator = ServerMetricsAccumulator(config=config)
        >>>
        >>> # Process records from collection
        >>> record = ServerMetricsRecord(
        ...     timestamp_ns=1_000_000_000,
        ...     endpoint_url="http://localhost:8081/metrics",
        ...     metrics={
        ...         "http_requests_total": MetricFamily(
        ...             type=PrometheusMetricType.COUNTER,
        ...             description="Total HTTP requests",
        ...             samples=[MetricSample(value=1500)]
        ...         )
        ...     }
        ... )
        >>> await accumulator.process_server_metrics_record(record)
        >>>
        >>> # Export results after profiling
        >>> results = accumulator.export_results(
        ...     start_ns=1_000_000_000,  # Profiling start
        ...     end_ns=10_000_000_000    # Profiling end
        ... )
        >>> results.endpoint_summaries["localhost:8081"].metrics["http_requests_total"]
        CounterMetricData(description="Total HTTP requests", series=[...])
    """

    def __init__(
        self,
        run: BenchmarkRun,
        pub_client: PubClientProtocol | None = None,
        **kwargs: Any,
    ) -> None:
        config = run.cfg
        if config.server_metrics_disabled:
            raise PostProcessorDisabled(
                "Server metrics results processor is disabled via --no-server-metrics"
            )

        self.pub_client = pub_client
        super().__init__(run=run, **kwargs)

        self._server_metrics_hierarchy = ServerMetricsHierarchy()
        # Use slice_duration from config for windowed stats
        self._slice_duration: float | None = config.output.slice_duration
        # Track last endpoint summaries hash to avoid duplicate publishes
        self._last_summaries_hash: int | None = None
        # Track profiling start time for realtime computations
        self._profiling_start_ns: int | None = None

    def get_hierarchy_for_export(self) -> ServerMetricsHierarchy:
        """Get server metrics hierarchy for export purposes.

        Provides read-only access to the internal hierarchical storage for exporters
        that need to access raw time-series data directly (e.g., Parquet exporter).

        Returns:
            ServerMetricsHierarchy containing all accumulated time-series data
        """
        return self._server_metrics_hierarchy

    async def process_server_metrics_record(self, record: ServerMetricsRecord) -> None:
        """Process individual server metrics record into hierarchical storage.

        Args:
            record: ServerMetricsRecord containing Prometheus metrics and metadata
        """
        self._server_metrics_hierarchy.add_record(record)

    async def export_results(
        self,
        start_ns: int,
        end_ns: int,
        error_summary: list[ErrorDetailsCount] | None = None,
    ) -> ServerMetricsResults | None:
        """Export accumulated server metrics as results for final reporting.

        Called at the end of profiling to generate the final ServerMetricsResults
        object containing all computed statistics. Applies time filtering to
        exclude warmup periods and computes per-endpoint summaries with stats.

        The time range [start_ns, end_ns] represents the profiling phase only,
        excluding warmup. Reference points before start_ns are used for counter
        and histogram delta calculations.

        Args:
            start_ns: Profiling phase start time in nanoseconds (excludes warmup period)
            end_ns: Profiling phase end time in nanoseconds (may extend beyond last collection)
            error_summary: Optional list of error counts from collection failures

        Returns:
            ServerMetricsResults containing endpoint summaries with computed statistics,
            or None if no endpoints were successfully scraped during profiling.
        """
        if not self._server_metrics_hierarchy.endpoints:
            return None

        endpoint_summaries = self._compute_endpoint_summaries(
            start_ns, end_ns, self._slice_duration
        )

        endpoint_list = list(self._server_metrics_hierarchy.endpoints.keys())
        results = ServerMetricsResults(
            benchmark_id=self.run.cfg.benchmark_id,
            endpoint_summaries=endpoint_summaries,
            start_ns=start_ns,
            end_ns=end_ns,
            endpoints_configured=endpoint_list,
            endpoints_successful=endpoint_list,
            error_summary=error_summary or [],
        )

        # Parquet spans all endpoints with a single filter; widen to bracket
        # any scrape data across endpoints so sub-scrape-interval benchmarks
        # still emit rows. Matches the per-endpoint widening in
        # _compute_endpoint_summaries.
        parquet_start = start_ns
        parquet_end = end_ns
        for ts in self._server_metrics_hierarchy.endpoints.values():
            if ts.last_update_ns:
                parquet_end = max(parquet_end, ts.last_update_ns)
                if ts.last_update_ns < parquet_start:
                    parquet_start = ts.last_update_ns

        await self._export_parquet_if_enabled(
            TimeRangeFilter(start_ns=parquet_start, end_ns=parquet_end)
        )

        return results

    def _compute_endpoint_summaries(
        self,
        profiling_start_ns: int,
        profiling_end_ns: int,
        slice_duration: float | None = None,
        fast_histogram_percentiles: bool = False,
    ) -> dict[str, ServerMetricsEndpointSummary]:
        """Compute all server metrics summaries with per-endpoint time filters.

        For each endpoint, computes:
        1. Per-metric statistics (gauge avg/min/max, counter deltas, histogram percentiles)
        2. Collection metadata (fetch count, latencies, update intervals)
        3. Optional timeslice-based analysis for rate variation over time

        Time filtering is applied per-endpoint to handle cases where different
        endpoints have different collection start/end times. The filter uses:
        - profiling_start_ns to exclude warmup metrics
        - max(profiling_end_ns, last_update_ns) to include final collection

        Args:
            profiling_start_ns: Profiling phase start time (excludes warmup period)
            profiling_end_ns: Profiling phase end time (benchmark completion time)
            slice_duration: Duration of each timeslice window in seconds for time-sliced stats.
                           If None, timeslice analysis is skipped (saves computation).
            fast_histogram_percentiles: Algorithm selection for histogram percentile estimation.
                                        True = Prometheus linear interpolation (fast, for realtime).
                                        False = Polynomial histogram (accurate, for final export).

        Returns:
            Dict mapping endpoint display names (e.g., "localhost:8081") to
            ServerMetricsEndpointSummary objects containing all computed statistics.
        """
        summaries: dict[str, ServerMetricsEndpointSummary] = {}

        for (
            endpoint_url,
            time_series,
        ) in self._server_metrics_hierarchy.endpoints.items():
            endpoint_display = normalize_endpoint_display(endpoint_url)

            # Construct per-endpoint TimeFilter
            # Use profiling_start_ns to exclude warmup period (reference point can be before start)
            # Widen the filter to include observed data on either side of the
            # profiling window. Short benchmarks (sub-scrape-interval) can
            # finish before any scrape lands inside the window; without this
            # widening, counter/gauge stats would be dropped entirely while
            # histogram stats are retained, producing inconsistent exports.
            endpoint_start_ns = profiling_start_ns
            endpoint_end_ns = max(profiling_end_ns, time_series.last_update_ns)
            if (
                time_series.last_update_ns
                and time_series.last_update_ns < endpoint_start_ns
            ):
                endpoint_start_ns = time_series.last_update_ns
            time_filter = TimeRangeFilter(
                start_ns=endpoint_start_ns,
                end_ns=endpoint_end_ns,
            )

            metrics: dict[
                str,
                GaugeMetricData | CounterMetricData | HistogramMetricData,
            ] = {}

            for metric_key, metric_entry in time_series.metrics.items():
                base_name = metric_key.name

                series_stats = compute_stats(
                    metric_entry.metric_type,
                    metric_entry.data,
                    time_filter=time_filter,
                    labels=metric_key.labels_dict,
                    slice_duration=slice_duration,
                    fast_histogram_percentiles=fast_histogram_percentiles,
                )

                if series_stats is None:
                    continue

                if base_name not in metrics:
                    # Create appropriate type-specific metric data
                    match metric_entry.metric_type:
                        case PrometheusMetricType.GAUGE:
                            metrics[base_name] = GaugeMetricData(
                                description=metric_entry.description,
                                series=[series_stats],
                            )
                        case PrometheusMetricType.COUNTER:
                            metrics[base_name] = CounterMetricData(
                                description=metric_entry.description,
                                series=[series_stats],
                            )
                        case PrometheusMetricType.HISTOGRAM:
                            metrics[base_name] = HistogramMetricData(
                                description=metric_entry.description,
                                series=[series_stats],
                            )
                else:
                    metrics[base_name].series.append(series_stats)

            # Unique update statistics
            unique_count = time_series._unique_update_count
            duration_seconds = (
                (time_series.last_update_ns - time_series.first_update_ns)
                / NANOS_PER_SECOND
                if unique_count > 0
                else 0.0
            )
            avg_update_interval_ms = (
                (duration_seconds * MILLIS_PER_SECOND) / (unique_count - 1)
                if unique_count > 1
                else 0.0
            )
            # Compute median from actual intervals (more robust to outliers)
            median_update_interval_ms: float | None = None
            if time_series._update_intervals_ns:
                intervals_ns = np.array(
                    time_series._update_intervals_ns, dtype=np.int64
                )
                median_update_interval_ms = (
                    float(np.median(intervals_ns)) / NANOS_PER_MILLIS
                )

            # Fetch statistics (all fetches including duplicates)
            avg_fetch_latency_ms = (
                sum(time_series._fetch_latencies_ns)
                / len(time_series._fetch_latencies_ns)
                / NANOS_PER_MILLIS
                if time_series._fetch_latencies_ns
                else 0.0
            )

            summaries[endpoint_display] = ServerMetricsEndpointSummary(
                endpoint_url=endpoint_url,
                info=ServerMetricsEndpointInfo(
                    # Fetch statistics
                    total_fetches=time_series._total_fetch_count,
                    first_fetch_ns=time_series.first_fetch_ns,
                    last_fetch_ns=time_series.last_fetch_ns,
                    avg_fetch_latency_ms=avg_fetch_latency_ms,
                    # Unique update statistics
                    unique_updates=unique_count,
                    first_update_ns=time_series.first_update_ns,
                    last_update_ns=time_series.last_update_ns,
                    duration_seconds=duration_seconds,
                    avg_update_interval_ms=avg_update_interval_ms,
                    median_update_interval_ms=median_update_interval_ms,
                ),
                metrics=metrics,
            )

        return summaries

    async def _export_parquet_if_enabled(self, time_filter: TimeRangeFilter) -> None:
        """Export server metrics to Parquet format if enabled.

        This method is called during export_results() to write the Parquet file
        directly from the accumulator (where the raw time-series data lives).
        This avoids needing to pass the accumulator through ZMQ.

        Args:
            time_filter: Time range filter for the profiling period
        """
        # Check if Parquet format is enabled
        if ServerMetricsFormat.PARQUET not in self.run.cfg.server_metrics_formats:
            self.debug("Parquet format not selected, skipping export")
            return

        try:
            exporter = ServerMetricsParquetExporter(self, time_filter)
            await exporter.export()
            self.info(
                f"Exported server metrics to Parquet: {exporter.get_export_info().file_path}"
            )

        except DataExporterDisabled as e:
            self.debug(f"Parquet export disabled: {e}")
        except ImportError as e:
            self.warning(f"Failed to import Parquet exporter dependencies: {e}")
        except (OSError, ValueError) as e:
            self.error(f"Failed to export server metrics to Parquet: {e!r}")
        except Exception as e:  # noqa: BLE001 - best-effort parquet export; pyarrow.lib.ArrowInvalid lives behind optional import
            self.error(f"Failed to export server metrics to Parquet: {e!r}")

    async def summarize(self) -> list[MetricResult]:
        """Summarize accumulated metrics into MetricResult list.

        Server metrics are exported separately via export_results() rather than
        through the standard summarize() pipeline. This method returns empty list
        to satisfy the BaseMetricsProcessor interface.

        Returns:
            Empty list (server metrics exported via export_results instead)
        """
        return []

    def set_profiling_start(self, start_ns: int) -> None:
        """Set the profiling start time for realtime computations.

        Args:
            start_ns: Profiling phase start time in nanoseconds
        """
        self._profiling_start_ns = start_ns

    @background_task(interval=None, immediate=True)
    async def _report_realtime_server_metrics_task(self) -> None:
        """Report server metrics periodically for realtime dashboard display.

        Only runs when dashboard UI is enabled and pub_client is available.
        Uses a longer interval than telemetry since server metrics computation
        is more expensive.
        """
        if not self.pub_client or not self.run:
            return

        # Only publish when there's a consumer: dashboard UI or API server
        has_dashboard = self.run.cfg.ui_type == UIType.DASHBOARD
        has_api = bool(self.run.cfg.runtime.api_port)
        if not has_dashboard and not has_api:
            return

        while not self.stop_requested:
            await self._report_realtime_server_metrics()
            # Use 2x the normal interval since server metrics computation is heavier
            await asyncio.sleep(Environment.UI.REALTIME_METRICS_INTERVAL * 2)

    async def _report_realtime_server_metrics(self) -> None:
        """Compute and publish realtime server metrics snapshot."""
        if not self.pub_client:
            return

        if not self._server_metrics_hierarchy.endpoints:
            return

        # Use current time as end, and earliest data or profiling start as start
        current_ns = time.time_ns()
        start_ns = self._profiling_start_ns or 0

        # Find earliest data timestamp if no profiling start set
        if start_ns == 0:
            for ts in self._server_metrics_hierarchy.endpoints.values():
                if ts.first_update_ns and (
                    start_ns == 0 or ts.first_update_ns < start_ns
                ):
                    start_ns = ts.first_update_ns

        if start_ns == 0 or start_ns >= current_ns:
            return

        # Compute endpoint summaries for realtime display (no timeslices needed).
        # Use fast Prometheus percentiles - accuracy matters less for live dashboard.
        endpoint_summaries = self._compute_endpoint_summaries(
            start_ns, current_ns, slice_duration=None, fast_histogram_percentiles=True
        )

        if not endpoint_summaries:
            return

        # Check if summaries changed using hash
        summaries_hash = hash(str(endpoint_summaries))
        if summaries_hash == self._last_summaries_hash:
            return

        self._last_summaries_hash = summaries_hash

        await self.pub_client.publish(
            RealtimeServerMetricsMessage(
                service_id=self.id,
                endpoint_summaries=endpoint_summaries,
            )
        )
