# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from rich.console import Console, Group, RenderableType
from rich.table import Table
from rich.text import Text

from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.exporters.utils import normalize_endpoint_display
from aiperf.gpu_telemetry.constants import get_gpu_telemetry_metrics_config


class GPUTelemetryConsoleExporter(AIPerfLoggerMixin):
    """Console exporter for GPU telemetry data.

    Displays GPU metrics in a table format similar to other console exporters.
    Only displays when --gpu-telemetry flag is explicitly provided by the user.
    """

    STAT_COLUMN_KEYS = ["avg", "min", "max", "p99", "p90", "p50", "std"]

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self._results = exporter_config.results
        self._user_config = exporter_config.user_config
        self._service_config = exporter_config.service_config
        self._exporter_config = exporter_config
        self._telemetry_results = exporter_config.telemetry_results

    async def export(self, console: Console) -> None:
        """Export telemetry data to console if --gpu-telemetry flag is present.

        Only displays telemetry data when --gpu-telemetry flag is explicitly provided.
        Skips display if no telemetry data is available.

        Args:
            console: Rich Console instance for formatted output
        """

        if (
            self._user_config.gpu_telemetry is None
            or self._user_config.gpu_telemetry_disabled
        ):
            return

        if not self._telemetry_results:
            return

        self._print_renderable(console, self.get_renderable())

    def _print_renderable(self, console: Console, renderable: RenderableType) -> None:
        """Print the renderable to the console with formatting.

        Adds blank line before output and flushes console buffer after printing.

        Args:
            console: Rich Console instance for formatted output
            renderable: Rich renderable object (Table, Group, Text, etc.) to display
        """
        console.print("\n")
        console.print(renderable)
        console.file.flush()

    def get_renderable(self) -> RenderableType:
        """Create Rich tables showing GPU telemetry metrics with consolidated single-table format.

        Generates formatted output with:
        - Summary header showing endpoint reachability status
        - Per-GPU tables with metrics (power, utilization, temperature, etc.)
        - Statistical summaries (avg, min, max, p99, p90, p75, std) for each metric
        - Error summary if no data was collected

        Returns:
            RenderableType: Rich Group containing multiple Tables, or Text message if no data
        """
        renderables = []
        first_table = True

        # TelemetryExportData uses: endpoints[endpoint_display] -> EndpointData.gpus[gpu_key] -> GpuSummary
        for (
            endpoint_display,
            endpoint_data,
        ) in self._telemetry_results.endpoints.items():
            if not endpoint_data.gpus:
                continue

            for _gpu_key, gpu_summary in endpoint_data.gpus.items():
                gpu_index = gpu_summary.gpu_index
                gpu_name = gpu_summary.gpu_name
                table_title_base = f"{endpoint_display} | GPU {gpu_index} | {gpu_name}"

                if first_table:
                    first_table = False
                    table_title = self._create_summary_header(table_title_base)
                else:
                    renderables.append(Text(""))
                    table_title = table_title_base

                metrics_table = self._create_gpu_metrics_table(
                    table_title, gpu_summary, gpu_index
                )
                renderables.append(metrics_table)

        if not renderables:
            return self._create_no_data_message()

        return Group(*renderables)

    def _create_summary_header(self, table_title_base: str) -> str:
        """Create the summary header with endpoint reachability status.

        Args:
            table_title_base: Base title for the first table

        Returns:
            Formatted title string with endpoint status
        """
        title_lines = ["NVIDIA AIPerf | GPU Telemetry Summary"]

        endpoints_configured = self._telemetry_results.summary.endpoints_configured
        endpoints_successful = self._telemetry_results.summary.endpoints_successful
        total_count = len(endpoints_configured)
        successful_count = len(endpoints_successful)
        failed_count = total_count - successful_count

        if failed_count == 0:
            title_lines.append(
                f"[bold green]{successful_count}/{total_count} DCGM endpoints reachable[/bold green]"
            )
        elif successful_count == 0:
            title_lines.append(
                f"[bold red]{successful_count}/{total_count} DCGM endpoints reachable[/bold red]"
            )
        else:
            title_lines.append(
                f"[bold yellow]{successful_count}/{total_count} DCGM endpoints reachable[/bold yellow]"
            )

        for endpoint in endpoints_configured:
            clean_endpoint = normalize_endpoint_display(endpoint)
            if endpoint in endpoints_successful:
                title_lines.append(f"[green]• {clean_endpoint} \u2714 [/green]")
            else:
                title_lines.append(
                    f"[red]• {clean_endpoint} \u2718 (unreachable)[/red]"
                )

        title_lines.append("")
        title_lines.append(table_title_base)
        return "\n".join(title_lines)

    def _format_number(self, value) -> str:
        """Format a number for console output with adaptive formatting.

        Args:
            value: The value to format

        Returns:
            Formatted string representation of the value
        """
        if value is None:
            return "N/A"

        # Use scientific notation for very large numbers (> 1 million)
        if abs(value) >= 1_000_000:
            return f"{value:.2e}"

        # Use comma-separated format for smaller numbers
        return f"{value:,.2f}"

    def _create_gpu_metrics_table(
        self, table_title: str, gpu_summary, gpu_index: int
    ) -> Table:
        """Create a metrics table for a single GPU.

        Args:
            table_title: Title for the table
            gpu_summary: GpuSummary containing pre-computed metrics
            gpu_index: Index of the GPU

        Returns:
            Rich Table with GPU metrics
        """
        metrics_table = Table(show_header=True, title=table_title, title_style="italic")
        metrics_table.add_column("Metric", justify="right", style="cyan")
        for stat in self.STAT_COLUMN_KEYS:
            metrics_table.add_column(stat, justify="right", style="green")

        metrics_config = get_gpu_telemetry_metrics_config()

        for metric_display, metric_key, unit_enum in metrics_config:
            # Check if metric exists in pre-computed metrics
            if metric_key not in gpu_summary.metrics:
                continue

            try:
                unit = unit_enum.value
                metric_result = gpu_summary.metrics[metric_key]

                row = [f"{metric_display} ({unit})"]
                for stat in self.STAT_COLUMN_KEYS:
                    value = getattr(metric_result, stat, None)
                    row.append(self._format_number(value))

                metrics_table.add_row(*row)
            except Exception as e:
                self.debug(
                    f"Failed to retrieve metric {metric_key} for GPU {gpu_index}: {e}"
                )
                continue

        return metrics_table

    def _create_no_data_message(self) -> Text:
        """Create error message when no GPU telemetry data is available.

        Returns:
            Rich Text with error message and endpoint status
        """
        message_parts = ["No GPU telemetry data collected during the benchmarking run."]

        endpoints_configured = self._telemetry_results.summary.endpoints_configured
        endpoints_successful = self._telemetry_results.summary.endpoints_successful
        failed_endpoints = [
            ep for ep in endpoints_configured if ep not in endpoints_successful
        ]

        if failed_endpoints:
            message_parts.append("\n\nUnreachable endpoints:")
            for endpoint in failed_endpoints:
                clean_endpoint = normalize_endpoint_display(endpoint)
                message_parts.append(f"  • {clean_endpoint}")

        return Text("".join(message_parts), style="dim italic")
