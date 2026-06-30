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
        *,
        warmup_start_ns: int | None = None,
        warmup_end_ns: int | None = None,
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
            start_ns,
            end_ns,
            self._slice_duration,
            include_final_collection=True,
        )
        warmup_endpoint_summaries = None
        if warmup_start_ns is not None and warmup_end_ns is not None:
            warmup_endpoint_summaries = self._compute_endpoint_summaries(
                warmup_start_ns,
                warmup_end_ns,
                self._slice_duration,
                include_final_collection=False,
            )

        endpoint_list = list(self._server_metrics_hierarchy.endpoints.keys())
        results = ServerMetricsResults(
            benchmark_id=self.user_config.benchmark_id,
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
        return self._build_realtime_snapshot(endpoints, start_ns)

    def realtime_snapshots(
        self, start_ns: int | None = None
    ) -> dict[str, dict[str, float]]:
        """Return one live snapshot per server-metrics endpoint.

        Dynamo workers are labeled from their Prometheus
        ``dynamo_component`` label. ``prefill`` is shown directly and
        ``backend`` is rendered as ``decode`` for disaggregated deployments.
        Stable per-role indices follow sorted endpoint URL order. Endpoints
        without Dynamo labels fall back to their normalized host and port.
        """
        endpoint_snapshots: list[tuple[str, Any, dict[str, float]]] = []
        for endpoint_url, endpoint in sorted(
            self._server_metrics_hierarchy.endpoints.items()
        ):
            snapshot = self._build_realtime_snapshot([endpoint], start_ns)
            if snapshot:
                endpoint_snapshots.append((endpoint_url, endpoint, snapshot))

        snapshots: dict[str, dict[str, float]] = {}
        role_counts: dict[str, int] = {}
        for endpoint_url, endpoint, snapshot in endpoint_snapshots:
            role = self._dynamo_worker_role(endpoint)
            if role is None:
                label = (
                    ""
                    if len(endpoint_snapshots) == 1
                    else normalize_endpoint_display(endpoint_url)
                )
            else:
                role_index = role_counts.get(role, 0)
                role_counts[role] = role_index + 1
                label = f"{role} {role_index}"
            snapshots[label] = snapshot
        return snapshots

    def _build_realtime_snapshot(
        self, endpoints: list, start_ns: int | None
    ) -> dict[str, float]:
        """Build the realtime metric fields for the supplied endpoints."""
        out: dict[str, float] = {}

        self._add_prefix_cache_hit_rate(out, endpoints, start_ns)
        self._add_external_prefix_cache_hit_rate(out, endpoints, start_ns)
        self._add_kv_cache_usage_pct(out, endpoints)
        self._add_cpu_kv_cache_usage_pct(out, endpoints)
        self._add_queue_depth(out, endpoints)
        self._add_preemptions(out, endpoints, start_ns)
        self._add_token_throughputs(out, endpoints, start_ns)

        return out

    @staticmethod
    def _dynamo_worker_role(endpoint: Any) -> str | None:
        """Extract a concise worker role from an endpoint's metric labels."""
        for key in endpoint.metrics:
            labels = dict(key.labels)
            component = labels.get("dynamo_component")
            if component == "prefill":
                return "prefill"
            if component in {"backend", "decode"}:
                return "decode"
        return None

    def _add_prefix_cache_hit_rate(
        self, out: dict[str, float], endpoints: list, start_ns: int | None
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
        self, out: dict[str, float], endpoints: list, start_ns: int | None
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

    def _add_kv_cache_usage_pct(self, out: dict[str, float], endpoints: list) -> None:
        kv = self._first_gauge(
            endpoints,
            "vllm:kv_cache_usage_perc",
            "vllm:gpu_cache_usage_perc",
            "sglang:token_usage",
        )
        if kv is not None:
            out["kv_cache_usage_pct"] = self._to_pct(kv)

    def _add_cpu_kv_cache_usage_pct(
        self, out: dict[str, float], endpoints: list
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

    def _add_queue_depth(self, out: dict[str, float], endpoints: list) -> None:
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
        self, out: dict[str, float], endpoints: list, start_ns: int | None
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
        self, out: dict[str, float], endpoints: list, start_ns: int | None
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

    def _first_gauge(self, endpoints: list, *names: str) -> float | None:
        """First non-None gauge value across candidate metric names."""
        for name in names:
            v = self._gauge_latest_max(endpoints, name)
            if v is not None:
                return v
        return None

    def _first_counter_delta(
        self, endpoints: list, start_ns: int | None, *names: str
    ) -> float | None:
        """First non-None counter delta across candidate metric names."""
        for name in names:
            v = self._counter_delta(endpoints, name, start_ns)
            if v is not None:
                return v
        return None

    def _first_counter_rate(
        self, endpoints: list, start_ns: int | None, *names: str
    ) -> float | None:
        """First non-None counter rate across candidate metric names."""
        for name in names:
            v = self._counter_rate(endpoints, name, start_ns)
            if v is not None:
                return v
        return None

    @staticmethod
    def _to_pct(fraction: float) -> float:
        """Normalize a 0–1 gauge fraction to a 0–100 percentage."""
        return fraction * 100.0 if fraction <= 1.0 else fraction

    @staticmethod
    def _counter_delta(
        endpoints: list, metric_name: str, start_ns: int | None = None
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
        endpoints: list, num_name: str, den_name: str
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
    def _gauge_latest_max(endpoints: list, metric_name: str) -> float | None:
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
        endpoints: list, metric_name: str, start_ns: int | None = None
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
        return total_delta / (max_elapsed_ns / 1e9)
