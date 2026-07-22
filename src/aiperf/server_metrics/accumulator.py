# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from aiperf.common.constants import (
    MILLIS_PER_SECOND,
    NANOS_PER_MILLIS,
    NANOS_PER_SECOND,
)
from aiperf.common.enums import (
    CreditPhase,
    PrometheusMetricType,
    ServerMetricsFormat,
)
from aiperf.common.exceptions import DataExporterDisabled, PostProcessorDisabled
from aiperf.common.growable_array import GrowableArray
from aiperf.common.models import MetricResult
from aiperf.common.models.server_metrics_models import (
    CounterMetricData,
    GaugeMetricData,
    HistogramMetricData,
    MetricSample,
    ServerMetricsEndpointInfo,
    ServerMetricsEndpointSummary,
    ServerMetricsRecord,
    ServerMetricsResults,
    TimeRangeFilter,
    UnknownMetricData,
)
from aiperf.exporters.utils import normalize_endpoint_display
from aiperf.post_processors.base_metrics_processor import BaseMetricsProcessor
from aiperf.server_metrics.export_stats import compute_stats
from aiperf.server_metrics.parquet_exporter import ServerMetricsParquetExporter
from aiperf.server_metrics.storage import (
    HistogramTimeSeries,
    ScalarTimeSeries,
    ServerMetricsHierarchy,
    ServerMetricsTimeSeries,
)

if TYPE_CHECKING:
    from aiperf.common.accumulator_protocols import ExportContext, SummaryContext
    from aiperf.config.resolution.plan import BenchmarkRun

_METRIC_DATA_CLASSES: dict[
    PrometheusMetricType,
    type[GaugeMetricData | CounterMetricData | HistogramMetricData | UnknownMetricData],
] = {
    PrometheusMetricType.GAUGE: GaugeMetricData,
    PrometheusMetricType.UNKNOWN: UnknownMetricData,
    PrometheusMetricType.COUNTER: CounterMetricData,
    PrometheusMetricType.HISTOGRAM: HistogramMetricData,
}


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
        run: BenchmarkRun carrying the BenchmarkConfig + per-run state.
        **kwargs: Additional arguments passed to base class

    Raises:
        PostProcessorDisabled: If --no-server-metrics flag is set
    """

    def __init__(self, run: BenchmarkRun, **kwargs: Any):
        if not run.cfg.server_metrics.enabled:
            raise PostProcessorDisabled(
                "Server metrics results processor is disabled via --no-server-metrics"
            )

        super().__init__(run=run, **kwargs)

        self._server_metrics_hierarchy = ServerMetricsHierarchy()
        # Use slice_duration from config for windowed stats
        self._slice_duration: float | None = self.run.cfg.artifacts.slice_duration
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
        """Return a boolean mask where True marks records in ``[start_ns, end_ns)``.

        Half-open by design to match ``AccumulatorProtocol.query_time_range``
        and the metrics accumulator. Distinct from per-series
        ``get_time_mask`` / ``get_indices_for_filter``, which use inclusive
        ``[start_ns, end_ns]`` for Prometheus sample windows.
        """
        if len(self._timestamps_ns) == 0:
            return np.array([], dtype=bool)
        ts = self._timestamps_ns.data
        return (ts >= start_ns) & (ts < end_ns)

    async def export_results(self, ctx: ExportContext) -> ServerMetricsResults | None:
        """Export accumulated server metrics as results for final reporting.

        Called at the end of profiling to generate the final ServerMetricsResults
        object containing all computed statistics. Applies time filtering to
        exclude warmup periods and computes per-endpoint summaries with stats.

        Reads the profiling window from ``ctx.start_ns/ctx.end_ns`` (excludes
        warmup; reference points before start_ns drive counter/histogram deltas)
        and the warmup window from ``ctx.warmup_start_ns/ctx.warmup_end_ns``.

        Returns:
            ServerMetricsResults containing endpoint summaries with computed statistics,
            or None if no endpoints were successfully scraped during profiling.
        """
        # ExportContext bounds are Optional (None = unbounded). Production callers
        # always pass concrete ints, but normalize here so a bare
        # ``export_results(ExportContext())`` can't reach the int-only max()/
        # comparison in _compute_endpoint_summaries and raise TypeError. 0 means
        # "from the beginning"; the per-endpoint max(end, last_update) still
        # captures the final scrape.
        start_ns = ctx.start_ns or 0
        end_ns = ctx.end_ns or 0
        error_summary = ctx.error_summary
        warmup_start_ns = ctx.warmup_start_ns
        warmup_end_ns = ctx.warmup_end_ns

        if not self._server_metrics_hierarchy.endpoints:
            return None

        endpoint_summaries = self._compute_phase_endpoint_summaries(
            CreditPhase.PROFILING,
            self._slice_duration,
            include_final_collection=not ctx.is_phase_scoped,
        )
        if not endpoint_summaries:
            endpoint_summaries = self._compute_endpoint_summaries(
                start_ns,
                end_ns,
                self._slice_duration,
                include_final_collection=not ctx.is_phase_scoped,
            )
        warmup_endpoint_summaries = None
        if (
            warmup_start_ns is not None
            and warmup_end_ns is not None
            and warmup_start_ns < warmup_end_ns
        ):
            warmup_endpoint_summaries = self._compute_phase_endpoint_summaries(
                CreditPhase.WARMUP,
                self._slice_duration,
                include_final_collection=False,
            )
            if not warmup_endpoint_summaries:
                warmup_endpoint_summaries = self._compute_endpoint_summaries(
                    warmup_start_ns,
                    warmup_end_ns,
                    self._slice_duration,
                    include_final_collection=False,
                )

        endpoint_list = list(self._server_metrics_hierarchy.endpoints.keys())
        results = ServerMetricsResults(
            benchmark_id=self.run.benchmark_id,
            endpoint_summaries=endpoint_summaries,
            warmup_endpoint_summaries=warmup_endpoint_summaries or None,
            start_ns=start_ns,
            end_ns=end_ns,
            endpoints_configured=endpoint_list,
            endpoints_successful=endpoint_list,
            error_summary=error_summary or [],
            warmup_start_ns=warmup_start_ns,
            warmup_end_ns=warmup_end_ns,
        )

        # Export Parquet file directly from accumulator if format is enabled.
        # Widen the export window to include the final per-endpoint collection,
        # which may land after end_ns (e.g. a scrape completing post-benchmark).
        export_end_ns = max(
            end_ns,
            *(
                time_series.last_update_ns
                for time_series in self._server_metrics_hierarchy.endpoints.values()
            ),
        )
        # Skip degenerate windows: TimeRangeFilter rejects start >= end, and a
        # raise here propagates out of export_results and is swallowed into a
        # None result (records_manager _publish_server_metrics_results), losing
        # ALL server metrics. Mirrors the guards at the per-endpoint / warmup /
        # json_exporter sites.
        if ctx.phase_index is None and start_ns < export_end_ns:
            await self._export_parquet_if_enabled(
                TimeRangeFilter(start_ns=start_ns, end_ns=export_end_ns)
            )

        return results

    def _compute_endpoint_summaries(
        self,
        profiling_start_ns: int,
        profiling_end_ns: int,
        slice_duration: float | None = None,
        *,
        include_final_collection: bool,
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
            # Use max(profiling_end, last_update) for profiling to include the
            # final collection. Phase-scoped warmup summaries must not extend
            # past their own completed request window.
            endpoint_start_ns = profiling_start_ns
            endpoint_end_ns = (
                max(profiling_end_ns, time_series.last_update_ns)
                if include_final_collection
                else profiling_end_ns
            )
            # Skip degenerate windows: TimeRangeFilter rejects start >= end.
            if endpoint_start_ns >= endpoint_end_ns:
                continue
            time_filter = TimeRangeFilter(
                start_ns=endpoint_start_ns,
                end_ns=endpoint_end_ns,
            )

            metrics: dict[
                str,
                GaugeMetricData
                | CounterMetricData
                | HistogramMetricData
                | UnknownMetricData,
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

                if base_name in metrics:
                    metrics[base_name].series.append(series_stats)
                    continue
                # Create appropriate type-specific metric data; unmapped types
                # are skipped (same semantics as the previous non-exhaustive match).
                DataClass = _METRIC_DATA_CLASSES.get(metric_entry.metric_type)
                if DataClass is not None:
                    metrics[base_name] = DataClass(
                        description=metric_entry.description,
                        series=[series_stats],
                    )

            info_filter = None if include_final_collection else time_filter
            summaries[endpoint_display] = ServerMetricsEndpointSummary(
                endpoint_url=endpoint_url,
                info=self._compute_endpoint_info(time_series, info_filter),
                metrics=metrics,
            )

        return summaries

    def _build_phase_filtered_scalar_series(
        self,
        data: ScalarTimeSeries,
        phase: CreditPhase,
    ) -> tuple[ScalarTimeSeries, int, int] | None:
        phase_indices = np.flatnonzero(data.get_phase_mask(phase))
        if len(phase_indices) == 0:
            return None

        filtered = ScalarTimeSeries()
        first_idx = int(phase_indices[0])
        if first_idx > 0:
            filtered.append(
                int(data.timestamps[first_idx - 1]),
                MetricSample(value=float(data.values[first_idx - 1])),
            )
        for idx in phase_indices:
            filtered.append(
                int(data.timestamps[idx]),
                MetricSample(value=float(data.values[idx])),
            )
        return (
            filtered,
            int(data.timestamps[first_idx]),
            int(data.timestamps[phase_indices[-1]]),
        )

    def _build_phase_filtered_histogram_series(
        self,
        data: HistogramTimeSeries,
        phase: CreditPhase,
    ) -> tuple[HistogramTimeSeries, int, int] | None:
        phase_indices = np.flatnonzero(data.get_phase_mask(phase))
        if len(phase_indices) == 0:
            return None

        filtered = HistogramTimeSeries()
        first_idx = int(phase_indices[0])

        def _append(idx: int) -> None:
            filtered.append(
                int(data.timestamps[idx]),
                MetricSample(
                    buckets={k: float(v) for k, v in data.get_bucket_dict(idx).items()},
                    sum=float(data.sums[idx]),
                    count=float(data.counts[idx]),
                ),
            )

        if first_idx > 0:
            _append(first_idx - 1)
        for idx in phase_indices:
            _append(int(idx))
        return (
            filtered,
            int(data.timestamps[first_idx]),
            int(data.timestamps[phase_indices[-1]]),
        )

    def _compute_phase_endpoint_summaries(
        self,
        phase: CreditPhase,
        slice_duration: float | None = None,
        *,
        include_final_collection: bool,
    ) -> dict[str, ServerMetricsEndpointSummary]:
        """Compute per-endpoint summaries from samples whose record phase matches ``phase``."""
        summaries: dict[str, ServerMetricsEndpointSummary] = {}

        for (
            endpoint_url,
            time_series,
        ) in self._server_metrics_hierarchy.endpoints.items():
            endpoint_display = normalize_endpoint_display(endpoint_url)
            metrics: dict[
                str,
                GaugeMetricData
                | CounterMetricData
                | HistogramMetricData
                | UnknownMetricData,
            ] = {}
            phase_start_ns: int | None = None
            phase_end_ns: int | None = None

            for metric_key, metric_entry in time_series.metrics.items():
                filtered_series: (
                    tuple[ScalarTimeSeries, int, int]
                    | tuple[HistogramTimeSeries, int, int]
                    | None
                )
                if isinstance(metric_entry.data, ScalarTimeSeries):
                    filtered_series = self._build_phase_filtered_scalar_series(
                        metric_entry.data,
                        phase,
                    )
                else:
                    filtered_series = self._build_phase_filtered_histogram_series(
                        metric_entry.data,
                        phase,
                    )
                if filtered_series is None:
                    continue

                series_data, start_ns, end_ns = filtered_series
                phase_start_ns = (
                    start_ns
                    if phase_start_ns is None
                    else min(phase_start_ns, start_ns)
                )
                phase_end_ns = (
                    end_ns if phase_end_ns is None else max(phase_end_ns, end_ns)
                )
                time_filter = TimeRangeFilter(
                    start_ns=start_ns,
                    end_ns=end_ns if end_ns > start_ns else start_ns + 1,
                )
                series_stats = compute_stats(
                    metric_entry.metric_type,
                    series_data,
                    time_filter,
                    labels=metric_key.labels_dict,
                    slice_duration=slice_duration,
                )
                if series_stats is None:
                    continue

                base_name = metric_key.name
                if base_name in metrics:
                    metrics[base_name].series.append(series_stats)
                    continue

                DataClass = _METRIC_DATA_CLASSES.get(metric_entry.metric_type)
                if DataClass is not None:
                    metrics[base_name] = DataClass(
                        description=metric_entry.description,
                        series=[series_stats],
                    )

            if not metrics:
                continue

            info_filter = None
            if not include_final_collection and phase_start_ns is not None:
                info_filter = TimeRangeFilter(
                    start_ns=phase_start_ns,
                    end_ns=(
                        phase_end_ns
                        if phase_end_ns is not None and phase_end_ns > phase_start_ns
                        else phase_start_ns + 1
                    ),
                )
            summaries[endpoint_display] = ServerMetricsEndpointSummary(
                endpoint_url=endpoint_url,
                info=self._compute_endpoint_info(time_series, info_filter),
                metrics=metrics,
            )

        return summaries

    @staticmethod
    def _compute_endpoint_info(
        time_series: ServerMetricsTimeSeries,
        time_filter: TimeRangeFilter | None,
    ) -> ServerMetricsEndpointInfo:
        if time_filter is None:
            fetch_timestamps = list(time_series._fetch_timestamps_ns)
            fetch_latencies_ns = list(time_series._fetch_latencies_ns)
            update_timestamps = sorted(time_series._unique_update_timestamps)
        else:
            fetch_timestamps = [
                timestamp_ns
                for timestamp_ns in time_series._fetch_timestamps_ns
                if time_filter.includes(timestamp_ns)
            ]
            fetch_latencies_ns = [
                latency_ns
                for timestamp_ns, latency_ns in time_series._fetch_latency_records_ns
                if time_filter.includes(timestamp_ns)
            ]
            update_timestamps = sorted(
                timestamp_ns
                for timestamp_ns in time_series._unique_update_timestamps
                if time_filter.includes(timestamp_ns)
            )

        unique_count = len(update_timestamps)
        first_update_ns = update_timestamps[0] if update_timestamps else 0
        last_update_ns = update_timestamps[-1] if update_timestamps else 0
        duration_seconds = (
            (last_update_ns - first_update_ns) / NANOS_PER_SECOND
            if unique_count > 0
            else 0.0
        )
        avg_update_interval_ms = (
            (duration_seconds * MILLIS_PER_SECOND) / (unique_count - 1)
            if unique_count > 1
            else 0.0
        )

        median_update_interval_ms: float | None = None
        if unique_count > 1:
            intervals_ns = np.diff(np.array(update_timestamps, dtype=np.int64))
            median_update_interval_ms = (
                float(np.median(intervals_ns)) / NANOS_PER_MILLIS
            )

        avg_fetch_latency_ms = (
            sum(fetch_latencies_ns) / len(fetch_latencies_ns) / NANOS_PER_MILLIS
            if fetch_latencies_ns
            else 0.0
        )

        return ServerMetricsEndpointInfo(
            total_fetches=len(fetch_timestamps),
            first_fetch_ns=min(fetch_timestamps) if fetch_timestamps else 0,
            last_fetch_ns=max(fetch_timestamps) if fetch_timestamps else 0,
            avg_fetch_latency_ms=avg_fetch_latency_ms,
            unique_updates=unique_count,
            first_update_ns=first_update_ns,
            last_update_ns=last_update_ns,
            duration_seconds=duration_seconds,
            avg_update_interval_ms=avg_update_interval_ms,
            median_update_interval_ms=median_update_interval_ms,
        )

    async def _export_parquet_if_enabled(self, time_filter: TimeRangeFilter) -> None:
        """Export server metrics to Parquet format if enabled.

        This method is called during export_results() to write the Parquet file
        directly from the accumulator (where the raw time-series data lives).
        This avoids needing to pass the accumulator through ZMQ.

        Args:
            time_filter: Time range filter for the profiling period
        """
        # Check if Parquet format is enabled
        if ServerMetricsFormat.PARQUET not in self.run.cfg.server_metrics.formats:
            self.debug("Parquet format not selected, skipping export")
            return

        try:
            exporter = ServerMetricsParquetExporter(self, time_filter)
            await exporter.export()
            self.info(
                f"Exported server metrics to Parquet: {exporter.get_export_info().file_path}"
            )

        except DataExporterDisabled as e:
            # Parquet was explicitly requested (checked above), so surface the
            # reason it was skipped (e.g. pyarrow unavailable on Windows-on-ARM)
            # rather than hiding it at debug level.
            self.warning(f"Parquet export disabled: {e}")
        except ImportError as e:
            self.warning(f"Failed to import Parquet exporter dependencies: {e}")
        except Exception as e:
            self.error(f"Failed to export server metrics to Parquet: {e!r}")

    async def summarize(self, ctx: SummaryContext | None = None) -> list[MetricResult]:
        """Summarize accumulated metrics into MetricResult list.

        Server metrics are exported separately via export_results() rather than
        through the standard summarize() pipeline. This method returns empty list
        to satisfy the BaseMetricsProcessor interface.

        Returns:
            Empty list (server metrics exported via export_results instead)
        """
        return []

    def realtime_snapshot(self, start_ns: int | None = None) -> dict[str, float]:
        """Live snapshot of key server metrics for the realtime stats block.

        Returns a flat ``{metric_name: value}`` dict with the metrics most
        useful to display mid-run. Each field is sourced from vLLM first and
        falls back to the SGLang equivalent when vLLM names are absent, so
        the realtime ``srv`` row populates for both backends.

        - ``prefix_cache_hit_rate`` — vLLM counter pair
          ``vllm:prefix_cache_hits`` / ``vllm:prefix_cache_queries`` (delta
          from ``start_ns`` when supplied), or SGLang counter pair
          ``sglang:cached_tokens_total`` / ``sglang:prompt_tokens_total``
          (same shape; cumulative rate, combined L1+L2+L3 via RadixAttention).
          Falls back last to the per-batch ``sglang:cache_hit_rate`` gauge
          for older SGLang builds.
        - ``unique_input_tokens_srv`` — derived from either counter pair as
          ``queries - hits`` (vLLM) or ``prompt - cached`` (SGLang). Empty
          when only the SGLang gauge is available.
        - ``external_prefix_cache_hit_rate`` — vLLM
          ``vllm:external_prefix_cache_*`` only. SGLang folds HiCache hits
          into ``sglang:cache_hit_rate`` and exposes no separate hit rate.
        - ``kv_cache_usage_pct`` — vLLM ``vllm:kv_cache_usage_perc`` (v0
          fallback ``vllm:gpu_cache_usage_perc``) or SGLang
          ``sglang:token_usage``.
        - ``cpu_kv_cache_usage_pct`` — vLLM ``vllm:cpu_cache_usage_perc``
          (SimpleCPUOffloadConnector) or SGLang derived ratio
          ``sglang:hicache_host_used_tokens`` / ``sglang:hicache_host_total_tokens``
          (HiCache-enabled runs only).
        - ``num_running`` / ``num_waiting`` — vLLM ``vllm:num_requests_running``
          / ``vllm:num_requests_waiting`` or SGLang ``sglang:num_running_reqs``
          / ``sglang:num_queue_reqs``.
        - ``num_preemptions`` — vLLM ``vllm:num_preemptions`` or SGLang
          ``sglang:num_retracted_reqs_total`` (counter delta).
        - ``input_token_throughput_srv`` / ``output_token_throughput_srv`` —
          counter rate over the elapsed window from ``vllm:prompt_tokens_total``
          / ``vllm:generation_tokens_total`` or SGLang
          ``sglang:prompt_tokens_total`` / ``sglang:generation_tokens_total``.

        Counter lookups internally use the parser-stripped form (no ``_total``
        suffix) because ``prometheus_client.parser.text_string_to_metric_families``
        strips it from the family name. Helpers gate by ``metric_type`` to keep
        gauge/counter name collisions (e.g. SGLang's ``num_retracted_reqs``
        gauge vs ``num_retracted_reqs_total`` counter) from cross-contaminating.

        Returns ``{}`` when no server metrics have been received yet, so
        callers can suppress the row on early ticks.
        """
        endpoints = list(self._server_metrics_hierarchy.endpoints.values())
        if not endpoints:
            return {}
        out: dict[str, float] = {}

        self._add_prefix_cache_hit_rate(out, endpoints, start_ns)
        self._add_external_prefix_cache_hit_rate(out, endpoints, start_ns)
        self._add_kv_cache_usage_pct(out, endpoints)
        self._add_cpu_kv_cache_usage_pct(out, endpoints)
        self._add_queue_depth(out, endpoints)
        self._add_preemptions(out, endpoints, start_ns)
        self._add_token_throughputs(out, endpoints, start_ns)

        return out

    def _add_prefix_cache_hit_rate(
        self,
        out: dict[str, float],
        endpoints: list[ServerMetricsTimeSeries],
        start_ns: int | None,
    ) -> None:
        hits = self._counter_delta(endpoints, "vllm:prefix_cache_hits", start_ns)
        queries = self._counter_delta(endpoints, "vllm:prefix_cache_queries", start_ns)
        if hits is not None and queries and queries > 0:
            # hits and queries are deltas from independently-latched counter
            # series; a query series lagging a batched hits update can make
            # hits > queries. Cap at 100% so the row never reports an
            # impossible hit rate.
            out["prefix_cache_hit_rate"] = 100.0 * min(hits, queries) / queries
            out["unique_input_tokens_srv"] = max(queries - hits, 0.0)
            return
        # SGLang counter pair: `cached_tokens_total` / `prompt_tokens_total`
        # — structurally identical to vLLM's hits / queries pair, so the
        # cumulative cache-hit rate (and the uncached-tokens delta) follow
        # the same formula. Use this in preference to `sglang:cache_hit_rate`,
        # which is a per-batch gauge that reads 0 between requests and gives
        # misleading values during idle scrape windows in low-concurrency
        # agentic replay.
        sgl_cached = self._counter_delta(endpoints, "sglang:cached_tokens", start_ns)
        sgl_prompt = self._counter_delta(endpoints, "sglang:prompt_tokens", start_ns)
        if sgl_cached is not None and sgl_prompt and sgl_prompt > 0:
            out["prefix_cache_hit_rate"] = (
                100.0 * min(sgl_cached, sgl_prompt) / sgl_prompt
            )
            out["unique_input_tokens_srv"] = max(sgl_prompt - sgl_cached, 0.0)
            return
        # Last-resort fallback for SGLang versions that emit only the gauge.
        sgl_rate = self._gauge_latest_max(endpoints, "sglang:cache_hit_rate")
        if sgl_rate is not None:
            out["prefix_cache_hit_rate"] = self._to_pct(sgl_rate)

    def _add_external_prefix_cache_hit_rate(
        self,
        out: dict[str, float],
        endpoints: list[ServerMetricsTimeSeries],
        start_ns: int | None,
    ) -> None:
        # Only emit when there has been any query against the external tier
        # — a 0/0 division otherwise produces a misleading "ext_cache_hit=0.0%"
        # row on offload=none configs that share the metric family with
        # offload=cpu peers. SGLang has no equivalent: HiCache hits are
        # folded into sglang:cache_hit_rate and not broken out.
        ext_hits = self._counter_delta(
            endpoints, "vllm:external_prefix_cache_hits", start_ns
        )
        ext_queries = self._counter_delta(
            endpoints, "vllm:external_prefix_cache_queries", start_ns
        )
        if ext_hits is not None and ext_queries and ext_queries > 0:
            out["external_prefix_cache_hit_rate"] = (
                100.0 * min(ext_hits, ext_queries) / ext_queries
            )

    def _add_kv_cache_usage_pct(
        self, out: dict[str, float], endpoints: list[ServerMetricsTimeSeries]
    ) -> None:
        kv = self._first_gauge(
            endpoints,
            "vllm:kv_cache_usage_perc",
            "vllm:gpu_cache_usage_perc",
            "sglang:token_usage",
        )
        if kv is not None:
            out["kv_cache_usage_pct"] = self._to_pct(kv)

    def _add_cpu_kv_cache_usage_pct(
        self, out: dict[str, float], endpoints: list[ServerMetricsTimeSeries]
    ) -> None:
        # vLLM emits a gauge directly (SimpleCPUOffloadConnector); SGLang
        # HiCache only emits used/total token counts on the host tier, so
        # the ratio is computed here.
        cpu_kv = self._gauge_latest_max(endpoints, "vllm:cpu_cache_usage_perc")
        if cpu_kv is None:
            # Pair used/total WITHIN each endpoint and take the busiest node's
            # ratio. Taking max(used) and max(total) independently across
            # endpoints could combine the numerator from one node with the
            # denominator from another, yielding a ratio matching no real node.
            cpu_kv = self._max_endpoint_gauge_ratio(
                endpoints,
                "sglang:hicache_host_used_tokens",
                "sglang:hicache_host_total_tokens",
            )
        if cpu_kv is not None:
            out["cpu_kv_cache_usage_pct"] = self._to_pct(cpu_kv)

    def _add_queue_depth(
        self, out: dict[str, float], endpoints: list[ServerMetricsTimeSeries]
    ) -> None:
        running = self._first_gauge(
            endpoints, "vllm:num_requests_running", "sglang:num_running_reqs"
        )
        if running is not None:
            out["num_running"] = running
        waiting = self._first_gauge(
            endpoints, "vllm:num_requests_waiting", "sglang:num_queue_reqs"
        )
        if waiting is not None:
            out["num_waiting"] = waiting

    def _add_preemptions(
        self,
        out: dict[str, float],
        endpoints: list[ServerMetricsTimeSeries],
        start_ns: int | None,
    ) -> None:
        # SGLang exposes the same concept as `num_retracted_reqs_total` (counter).
        # That name collides with `num_retracted_reqs` (gauge) after parser
        # stripping, so the counter-type filter in `_counter_delta` is what
        # keeps the lookup from picking up the gauge by mistake.
        preempt = self._first_counter_delta(
            endpoints,
            start_ns,
            "vllm:num_preemptions",
            "sglang:num_retracted_reqs",
        )
        if preempt is not None:
            out["num_preemptions"] = preempt

    def _add_token_throughputs(
        self,
        out: dict[str, float],
        endpoints: list[ServerMetricsTimeSeries],
        start_ns: int | None,
    ) -> None:
        # Counter delta over the elapsed window between first and last sample
        # — what the server itself observed across all in-flight + completed
        # requests (independent of aiperf's client-side accounting). Suppressed
        # when the counters are absent so non-vLLM/non-SGLang servers don't
        # show spurious zeroes. NOTE: the `_total` suffix is intentionally
        # absent — `prometheus_client.parser.text_string_to_metric_families`
        # strips it from the family name, so the stored key is the base form.
        in_rate = self._first_counter_rate(
            endpoints,
            start_ns,
            "vllm:prompt_tokens",
            "sglang:prompt_tokens",
        )
        if in_rate is not None:
            out["input_token_throughput_srv"] = in_rate
        out_rate = self._first_counter_rate(
            endpoints,
            start_ns,
            "vllm:generation_tokens",
            "sglang:generation_tokens",
        )
        if out_rate is not None:
            out["output_token_throughput_srv"] = out_rate

    def _first_gauge(
        self, endpoints: list[ServerMetricsTimeSeries], *names: str
    ) -> float | None:
        """First non-None gauge value across candidate metric names."""
        for name in names:
            v = self._gauge_latest_max(endpoints, name)
            if v is not None:
                return v
        return None

    def _first_counter_delta(
        self,
        endpoints: list[ServerMetricsTimeSeries],
        start_ns: int | None,
        *names: str,
    ) -> float | None:
        """First non-None counter delta across candidate metric names."""
        for name in names:
            v = self._counter_delta(endpoints, name, start_ns)
            if v is not None:
                return v
        return None

    def _first_counter_rate(
        self,
        endpoints: list[ServerMetricsTimeSeries],
        start_ns: int | None,
        *names: str,
    ) -> float | None:
        """First non-None counter rate across candidate metric names."""
        for name in names:
            v = self._counter_rate(endpoints, name, start_ns)
            if v is not None:
                return v
        return None

    @staticmethod
    def _to_pct(fraction: float) -> float:
        """Normalize a Prometheus ratio gauge to a 0-100 percentage.

        Values in ``[0, 1]`` are treated as ratios (e.g. SGLang
        ``sglang:cache_hit_rate``, ``sglang:token_usage``) and scaled by 100.
        Values ``> 1`` are treated as already-percent (some vLLM ``*_perc``
        series emit 0-100). A server that encoded a sub-1% reading as an
        already-percent value in ``(0, 1]`` (e.g. ``0.8`` meaning 0.8%) cannot
        be distinguished from an 80% ratio and will be scaled — prefer
        counter-pair sources when available.
        """
        return fraction * 100.0 if fraction <= 1.0 else fraction

    @staticmethod
    def _counter_delta(
        endpoints: list[ServerMetricsTimeSeries],
        metric_name: str,
        start_ns: int | None = None,
    ) -> float | None:
        """Sum (last - first) across endpoints for a counter metric.

        When ``start_ns`` is provided, use the last sample before ``start_ns`` as
        the baseline when present. This mirrors final export accounting so
        realtime rows can exclude warmup.

        Skips entries whose stored metric_type is not COUNTER — guards against
        the case where a gauge and a counter parse to the same family name
        (e.g. SGLang's ``num_retracted_reqs`` gauge collides with
        ``num_retracted_reqs_total`` counter after parser stripping).

        Returns None if no endpoint has at least two samples for the metric.
        """
        total = 0.0
        found = False
        for ep in endpoints:
            for key, entry in ep.metrics.items():
                if key.name != metric_name:
                    continue
                if entry.metric_type != PrometheusMetricType.COUNTER:
                    continue
                vals = entry.data.values
                if len(vals) >= 2:
                    baseline_idx = ServerMetricsAccumulator._counter_baseline_idx(
                        entry.data, start_ns
                    )
                    if baseline_idx is None or baseline_idx == len(vals) - 1:
                        continue
                    # Clamp counter resets to 0 (server restart drops the
                    # counter below its prior value), mirroring the export path
                    # (export_stats: max(raw_delta, 0)). Without this the
                    # realtime row emits negative rates / hit-rates.
                    total += max(float(vals[-1] - vals[baseline_idx]), 0.0)
                    found = True
        return total if found else None

    @staticmethod
    def _counter_baseline_idx(time_series: Any, start_ns: int | None) -> int | None:
        """Return the counter baseline index for an optional realtime start."""
        vals = time_series.values
        if len(vals) < 2:
            return None
        if start_ns is None:
            return 0

        first_in_window = int(
            np.searchsorted(time_series.timestamps, start_ns, side="left")
        )
        if first_in_window >= len(vals):
            return None
        return first_in_window - 1 if first_in_window > 0 else first_in_window

    @staticmethod
    def _counter_rate_baseline_idx(
        time_series: Any, start_ns: int | None
    ) -> int | None:
        """Rate-window baseline: the first sample AT/AFTER ``start_ns``.

        Unlike the delta baseline (``_counter_baseline_idx``, which picks the
        last sample BEFORE ``start_ns`` to mirror export delta accounting), the
        realtime rate must measure FROM ``start_ns`` so the warmup->start idle
        gap is excluded from the denominator. Returns None when fewer than two
        samples exist; the caller skips endpoints whose baseline is the final
        sample (no two-point window after ``start_ns``).
        """
        vals = time_series.values
        if len(vals) < 2:
            return None
        if start_ns is None:
            return 0
        return int(np.searchsorted(time_series.timestamps, start_ns, side="left"))

    @staticmethod
    def _max_endpoint_gauge_ratio(
        endpoints: list[ServerMetricsTimeSeries], num_name: str, den_name: str
    ) -> float | None:
        """Max per-endpoint ratio of two gauges, pairing numerator and
        denominator WITHIN each endpoint (never mixing across endpoints).

        Returns None if no endpoint has both gauges with a positive denominator.
        """
        best: float | None = None
        for ep in endpoints:
            num: float | None = None
            den: float | None = None
            for key, entry in ep.metrics.items():
                if entry.metric_type != PrometheusMetricType.GAUGE:
                    continue
                vals = entry.data.values
                if len(vals) == 0:
                    continue
                if key.name == num_name:
                    num = float(vals[-1])
                elif key.name == den_name:
                    den = float(vals[-1])
            if num is not None and den is not None and den > 0:
                ratio = num / den
                best = ratio if best is None else max(best, ratio)
        return best

    @staticmethod
    def _gauge_latest_max(
        endpoints: list[ServerMetricsTimeSeries], metric_name: str
    ) -> float | None:
        """Max of latest gauge values across endpoints, or None if absent.

        Skips entries whose stored metric_type is not GAUGE so a counter sharing
        the same name (after parser ``_total`` stripping) can't be misread as a
        gauge value.
        """
        best: float | None = None
        for ep in endpoints:
            for key, entry in ep.metrics.items():
                if key.name != metric_name:
                    continue
                if entry.metric_type != PrometheusMetricType.GAUGE:
                    continue
                vals = entry.data.values
                if len(vals) > 0:
                    v = float(vals[-1])
                    best = v if best is None else max(best, v)
        return best

    @staticmethod
    def _counter_rate(
        endpoints: list[ServerMetricsTimeSeries],
        metric_name: str,
        start_ns: int | None = None,
    ) -> float | None:
        """Sum (last - first) across endpoints divided by elapsed wall seconds.

        Running-average rate for a Prometheus counter, in tokens/sec. The window
        runs from each endpoint's rate baseline to its last observed sample.
        When ``start_ns`` is given the baseline is the first sample AT/AFTER
        ``start_ns`` (``_counter_rate_baseline_idx``), so the rate measures the
        profiling window only -- the warmup->start idle gap is NOT folded into
        the denominator. Skips entries whose stored metric_type is not COUNTER
        (see ``_counter_delta`` for the gauge-collision rationale).

        Returns None if no endpoint observed the metric, or if no endpoint has
        two samples at/after ``start_ns``.
        """
        total_delta = 0.0
        max_elapsed_ns: float = 0.0
        found = False
        for ep in endpoints:
            for key, entry in ep.metrics.items():
                if key.name != metric_name:
                    continue
                if entry.metric_type != PrometheusMetricType.COUNTER:
                    continue
                vals = entry.data.values
                ts = entry.data.timestamps
                if len(vals) < 2 or len(ts) < 2:
                    continue
                baseline_idx = ServerMetricsAccumulator._counter_rate_baseline_idx(
                    entry.data, start_ns
                )
                if baseline_idx is None or baseline_idx >= len(vals) - 1:
                    continue
                # Clamp counter resets to 0 (see _counter_delta) so a restart
                # cannot produce a negative throughput rate.
                total_delta += max(float(vals[-1] - vals[baseline_idx]), 0.0)
                max_elapsed_ns = max(max_elapsed_ns, float(ts[-1] - ts[baseline_idx]))
                found = True
        if not found or max_elapsed_ns <= 0:
            return None
        return total_delta / (max_elapsed_ns / NANOS_PER_SECOND)
