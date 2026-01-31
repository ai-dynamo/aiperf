# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

import orjson

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import PrometheusMetricType, ServerMetricsFormat
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.models.server_metrics_models import (
    CounterMetricData,
    GaugeMetricData,
    HistogramMetricData,
    ServerMetricsEndpointInfo,
    ServerMetricsExportData,
    ServerMetricsSummary,
)
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
from aiperf.exporters.metrics_base_exporter import MetricsBaseExporter
from aiperf.server_metrics.units import infer_unit


class ServerMetricsJsonExporter(MetricsBaseExporter):
    """Export server metrics to a separate JSON file in hybrid format.

    Exports server metrics with metrics keyed by name for O(1) lookup,
    while keeping stats flat within each series for easy access.

    Format: data["metrics"]["metric_name"]["series"][0]["stats"]["p99"]
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        """Initialize the JSON exporter for server metrics.

        Args:
            exporter_config: Configuration containing user settings and server metrics results
            **kwargs: Additional arguments passed to base class

        Raises:
            DataExporterDisabled: If server metrics are disabled or no data is available
        """
        if exporter_config.user_config.server_metrics_disabled:
            raise DataExporterDisabled("Server metrics is disabled")

        # Check if JSON format is enabled
        if (
            ServerMetricsFormat.JSON
            not in exporter_config.user_config.server_metrics_formats
        ):
            raise DataExporterDisabled(
                "Server metrics JSON export disabled: format not selected"
            )

        # Check if server metrics data is available before initializing
        if (
            not exporter_config.server_metrics_results
            or not exporter_config.server_metrics_results.endpoint_summaries
        ):
            raise DataExporterDisabled(
                "Server metrics JSON export disabled: no server metrics data available"
            )

        super().__init__(exporter_config, **kwargs)
        self._file_path = (
            exporter_config.user_config.output.server_metrics_export_json_file
        )
        self.trace_or_debug(
            lambda: f"Initializing ServerMetricsJsonExporter with config: {exporter_config}",
            lambda: f"Initializing ServerMetricsJsonExporter with file path: {self._file_path}",
        )

    def get_export_info(self) -> FileExportInfo:
        """Return export metadata for logging and user feedback.

        Returns:
            FileExportInfo with export type description and target file path
        """
        return FileExportInfo(
            export_type="Server Metrics JSON Export",
            file_path=self._file_path,
        )

    def _generate_content(self) -> str:
        """Generate JSON content for server metrics data in hybrid format.

        The hybrid format provides:
        - O(1) metric lookup by name (metrics keyed by name)
        - Type-specific stats within each series (Gauge/Counter/HistogramSeriesStats)
        - Unit parsed from metric name suffix
        - Both normalized endpoint and full endpoint_url

        Returns:
            str: JSON content with hybrid server metrics format
        """
        if not self._server_metrics_results:
            return "{}"

        metrics, endpoint_info = self._build_hybrid_metrics()

        endpoints_configured = [
            url for url in self._server_metrics_results.endpoints_configured
        ]
        endpoints_successful = [
            url for url in self._server_metrics_results.endpoints_successful
        ]

        summary = ServerMetricsSummary(
            endpoints_configured=endpoints_configured,
            endpoints_successful=endpoints_successful,
            start_time=datetime.fromtimestamp(
                self._server_metrics_results.start_ns / NANOS_PER_SECOND
            ),
            end_time=datetime.fromtimestamp(
                self._server_metrics_results.end_ns / NANOS_PER_SECOND
            ),
            endpoint_info=endpoint_info if endpoint_info else None,
        )

        # Serialize user config with exclude_unset=True to only include explicitly set values
        input_config = self._user_config.model_dump(
            mode="json",
            exclude_unset=True,
        )

        # Get AIPerf version from installed package
        from importlib.metadata import version as get_version

        try:
            aiperf_version = get_version("aiperf")
        except Exception:
            aiperf_version = "unknown"

        export_data = ServerMetricsExportData(
            aiperf_version=aiperf_version,
            benchmark_id=self._server_metrics_results.benchmark_id,
            summary=summary,
            metrics=metrics,
            input_config=input_config,
        )

        return orjson.dumps(
            export_data.model_dump(mode="json", exclude_none=True),
            option=orjson.OPT_INDENT_2,
        ).decode()

    def _build_hybrid_metrics(
        self,
    ) -> tuple[
        dict[
            str,
            GaugeMetricData | CounterMetricData | HistogramMetricData,
        ],
        dict[str, ServerMetricsEndpointInfo] | None,
    ]:
        """Build hybrid metrics dict from endpoint summaries for JSON export.

        Transforms hierarchical endpoint-first structure into metrics-first structure
        for efficient lookup. Merges series from all endpoints under each metric name,
        with endpoint_url embedded in each series for filtering.

        This hybrid format provides:
        - O(1) metric lookup by name: data["metrics"]["http_requests_total"]
        - Flat stats access: series["stats"]["p99"]
        - Multi-endpoint support: multiple series per metric, each tagged with endpoint
        - Type safety: GaugeMetricData, CounterMetricData, HistogramMetricData

        Series are sorted within each metric by endpoint URL then labels for
        deterministic output and easier navigation.

        Returns:
            Tuple of:
            - metrics dict: Maps metric name to type-specific MetricData with all series
            - endpoint_info dict: Maps endpoint URL to collection metadata (fetch counts, etc.)
        """
        if not self._server_metrics_results:
            return {}, None

        endpoint_summaries = self._server_metrics_results.endpoint_summaries
        if not endpoint_summaries:
            self.debug("No server metrics summaries available.")
            return {}, None

        metrics: dict[
            str,
            GaugeMetricData | CounterMetricData | HistogramMetricData,
        ] = {}
        endpoint_info: dict[str, ServerMetricsEndpointInfo] = {}

        for endpoint_summary in endpoint_summaries.values():
            endpoint_url = endpoint_summary.endpoint_url

            # Collect endpoint metadata for summary
            endpoint_info[endpoint_url] = endpoint_summary.info

            # Process metrics - series contains type-specific stats
            for metric_name, metric_data in endpoint_summary.metrics.items():
                unit = infer_unit(metric_name, metric_data.description)
                unit_display_name = unit.display_name() if unit else None

                # Get or create metric entry with appropriate type
                if metric_name not in metrics:
                    match metric_data.type:
                        case PrometheusMetricType.GAUGE:
                            metric_class = GaugeMetricData
                        case PrometheusMetricType.COUNTER:
                            metric_class = CounterMetricData
                        case PrometheusMetricType.HISTOGRAM:
                            metric_class = HistogramMetricData
                        case _:
                            raise ValueError(f"Unknown metric type: {metric_data.type}")

                    metrics[metric_name] = metric_class(
                        description=metric_data.description,
                        unit=unit_display_name,
                    )

                # Add endpoint info to each series stats and append
                for series_stats in metric_data.series:
                    series_stats.endpoint_url = endpoint_url
                    metrics[metric_name].series.append(series_stats)

        # Sort metrics alphabetically by name for deterministic output and easier lookup
        sorted_metrics = dict(sorted(metrics.items()))

        # Sort series within each metric by endpoint, then by labels
        for metric_data in sorted_metrics.values():
            metric_data.series.sort(
                key=lambda s: (s.endpoint_url or "", str(s.labels) if s.labels else "")
            )

        # Sort endpoint_info for consistency
        sorted_endpoint_info = (
            dict(sorted(endpoint_info.items())) if endpoint_info else None
        )

        return sorted_metrics, sorted_endpoint_info
