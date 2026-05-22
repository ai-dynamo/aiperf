# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
from numpy.typing import NDArray

from aiperf.common.config import UserConfig
from aiperf.common.constants import (
    MILLIS_PER_SECOND,
    NANOS_PER_MILLIS,
    NANOS_PER_SECOND,
)
from aiperf.common.enums import PrometheusMetricType, ServerMetricsFormat
from aiperf.common.exceptions import DataExporterDisabled, PostProcessorDisabled
from aiperf.common.growable_array import GrowableArray
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
        >>> from aiperf.common.config import UserConfig
        >>> from aiperf.common.models import ServerMetricsRecord, MetricFamily, MetricSample
        >>> # Create accumulator
        >>> config = UserConfig(...)
        >>> accumulator = ServerMetricsAccumulator(user_config=config)
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

    def __init__(self, user_config: UserConfig, **kwargs: Any):
        if user_config.server_metrics_disabled:
            raise PostProcessorDisabled(
                "Server metrics results processor is disabled via --no-server-metrics"
            )

        super().__init__(user_config=user_config, **kwargs)

        self._server_metrics_hierarchy = ServerMetricsHierarchy()
        # Use slice_duration from config for windowed stats
        self._slice_duration: float | None = user_config.output.slice_duration
        # Lightweight timestamp storage for query_time_range() (analyzer support)
        self._timestamps_ns = GrowableArray(initial_capacity=1024, dtype=np.int64)

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
        self._timestamps_ns.append(record.timestamp_ns)
        self._server_metrics_hierarchy.add_record(record)

    async def process_record(self, record: ServerMetricsRecord) -> None:
        """``AccumulatorProtocol``-compatible alias for ``process_server_metrics_record``."""
        await self.process_server_metrics_record(record)

    def query_time_range(self, start_ns: int, end_ns: int) -> NDArray[np.bool_]:
        """Return a boolean mask where True marks records in [start_ns, end_ns)."""
        if len(self._timestamps_ns) == 0:
            return np.array([], dtype=bool)
        ts = self._timestamps_ns.data
        return (ts >= start_ns) & (ts < end_ns)

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
            benchmark_id=self.user_config.benchmark_id,
            endpoint_summaries=endpoint_summaries,
            start_ns=start_ns,
            end_ns=end_ns,
            endpoints_configured=endpoint_list,
            endpoints_successful=endpoint_list,
            error_summary=error_summary or [],
        )

        # Export Parquet file directly from accumulator if format is enabled
        export_end_ns = max(
            end_ns,
            *(
                time_series.last_update_ns
                for time_series in self._server_metrics_hierarchy.endpoints.values()
            ),
        )
        await self._export_parquet_if_enabled(
            TimeRangeFilter(start_ns=start_ns, end_ns=export_end_ns)
        )

        return results

    def _compute_endpoint_summaries(
        self,
        profiling_start_ns: int,
        profiling_end_ns: int,
        slice_duration: float | None = None,
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
            # Use max(profiling_end, last_update) as end to include final collection
            # This ensures warmup metrics are excluded from aggregation
            endpoint_start_ns = profiling_start_ns
            endpoint_end_ns = max(profiling_end_ns, time_series.last_update_ns)
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
                    time_filter,
                    labels=metric_key.labels_dict,
                    slice_duration=slice_duration,
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
        if ServerMetricsFormat.PARQUET not in self.user_config.server_metrics_formats:
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
        except Exception as e:
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

    def realtime_snapshot(self) -> dict[str, float]:
        """Live snapshot of key server metrics for the realtime stats block.

        Returns a flat ``{metric_name: value}`` dict with the metrics most
        useful to display mid-run:

        - ``prefix_cache_hit_rate`` (% across all endpoints; counter-delta
          since the first observed sample; ``vllm:prefix_cache_hits`` /
          ``vllm:prefix_cache_queries``).
        - ``external_prefix_cache_hit_rate`` (% same shape, when CPU
          offload is active and ``vllm:external_prefix_cache_*`` are present).
        - ``kv_cache_usage_pct`` (latest gauge value, max across endpoints;
          v1 ``vllm:kv_cache_usage_perc`` with v0 ``vllm:gpu_cache_usage_perc``
          fallback).
        - ``cpu_kv_cache_usage_pct`` (only present when the server emits
          ``vllm:cpu_cache_usage_perc`` — i.e. CPU offload is active).
        - ``num_running`` / ``num_waiting`` (vLLM scheduler queue depth).
        - ``num_preemptions`` (cumulative total since first sample; vLLM
          ``vllm:num_preemptions`` or SGLang ``sglang:num_retracted_requests_total``).

        Returns ``{}`` when no server metrics have been received yet, so
        callers can suppress the row on early ticks.
        """
        endpoints = list(self._server_metrics_hierarchy.endpoints.values())
        if not endpoints:
            return {}
        out: dict[str, float] = {}

        hits = self._counter_delta(endpoints, "vllm:prefix_cache_hits")
        queries = self._counter_delta(endpoints, "vllm:prefix_cache_queries")
        if hits is not None and queries and queries > 0:
            out["prefix_cache_hit_rate"] = 100.0 * hits / queries

        # External (CPU-offload) prefix cache. Only emit when there has been
        # any query against the external tier — a 0/0 division otherwise
        # produces a misleading "ext_cache_hit=0.0%" row on offload=none
        # configs that share the metric family with offload=cpu peers.
        ext_hits = self._counter_delta(endpoints, "vllm:external_prefix_cache_hits")
        ext_queries = self._counter_delta(
            endpoints, "vllm:external_prefix_cache_queries"
        )
        if ext_hits is not None and ext_queries and ext_queries > 0:
            out["external_prefix_cache_hit_rate"] = 100.0 * ext_hits / ext_queries

        # GPU KV cache fill (gauge, 0–1 fraction in vLLM v1 → normalize to %).
        # v0 fallback: vllm:gpu_cache_usage_perc.
        kv = self._gauge_latest_max(endpoints, "vllm:kv_cache_usage_perc")
        if kv is None:
            kv = self._gauge_latest_max(endpoints, "vllm:gpu_cache_usage_perc")
        if kv is not None:
            out["kv_cache_usage_pct"] = kv * 100.0 if kv <= 1.0 else kv

        # CPU KV cache fill — present only on CPU-offload runs
        # (SimpleCPUOffloadConnector emits vllm:cpu_cache_usage_perc).
        cpu_kv = self._gauge_latest_max(endpoints, "vllm:cpu_cache_usage_perc")
        if cpu_kv is not None:
            out["cpu_kv_cache_usage_pct"] = cpu_kv * 100.0 if cpu_kv <= 1.0 else cpu_kv

        # vLLM scheduler queue depth — running + waiting. Catches backpressure
        # before it shows up as preemptions / latency.
        running = self._gauge_latest_max(endpoints, "vllm:num_requests_running")
        if running is not None:
            out["num_running"] = running
        waiting = self._gauge_latest_max(endpoints, "vllm:num_requests_waiting")
        if waiting is not None:
            out["num_waiting"] = waiting

        # Preemptions — vLLM retracts running requests on KV exhaustion;
        # SGLang exposes the same concept as a total counter.
        preempt = self._counter_delta(endpoints, "vllm:num_preemptions")
        if preempt is None:
            preempt = self._counter_delta(
                endpoints, "sglang:num_retracted_requests_total"
            )
        if preempt is not None:
            out["num_preemptions"] = preempt

        # Server-side running-average token throughput — counter delta over
        # the elapsed window between first and last sample. Equals what the
        # server itself observed across all in-flight + completed requests
        # (independent of aiperf's client-side accounting). Suppressed when
        # the counters are absent so SGLang / non-vLLM servers don't show
        # spurious zeroes.
        in_rate = self._counter_rate(endpoints, "vllm:prompt_tokens_total")
        if in_rate is not None:
            out["input_token_throughput_srv"] = in_rate
        out_rate = self._counter_rate(endpoints, "vllm:generation_tokens_total")
        if out_rate is not None:
            out["output_token_throughput_srv"] = out_rate

        return out

    @staticmethod
    def _counter_delta(endpoints: list, metric_name: str) -> float | None:
        """Sum (last - first) across endpoints for a counter metric.

        Returns None if no endpoint has at least two samples for the metric.
        """
        total = 0.0
        found = False
        for ep in endpoints:
            for key, entry in ep.metrics.items():
                if key.name != metric_name:
                    continue
                vals = entry.data.values
                if len(vals) >= 2:
                    total += float(vals[-1] - vals[0])
                    found = True
        return total if found else None

    @staticmethod
    def _gauge_latest_max(endpoints: list, metric_name: str) -> float | None:
        """Max of latest gauge values across endpoints, or None if absent."""
        best: float | None = None
        for ep in endpoints:
            for key, entry in ep.metrics.items():
                if key.name != metric_name:
                    continue
                vals = entry.data.values
                if len(vals) > 0:
                    v = float(vals[-1])
                    best = v if best is None else max(best, v)
        return best

    @staticmethod
    def _counter_rate(endpoints: list, metric_name: str) -> float | None:
        """Sum (last - first) across endpoints divided by elapsed wall seconds.

        Running-average rate for a Prometheus counter, in tokens/sec. Uses each
        endpoint's first and last observed timestamps as the window so warmup
        and post-stop samples are naturally excluded by the existing window
        (this snapshot is realtime-only, so the data span equals the run-so-far
        elapsed). Returns None if no endpoint observed the metric, or if every
        endpoint has only one sample (no elapsed time to divide by).
        """
        total_delta = 0.0
        max_elapsed_ns: float = 0.0
        found = False
        for ep in endpoints:
            for key, entry in ep.metrics.items():
                if key.name != metric_name:
                    continue
                vals = entry.data.values
                ts = entry.data.timestamps
                if len(vals) < 2 or len(ts) < 2:
                    continue
                total_delta += float(vals[-1] - vals[0])
                max_elapsed_ns = max(max_elapsed_ns, float(ts[-1] - ts[0]))
                found = True
        if not found or max_elapsed_ns <= 0:
            return None
        return total_delta / (max_elapsed_ns / 1e9)
