# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from rich.console import Console, Group, RenderableType
from rich.table import Table
from rich.text import Text

from aiperf.common.decorators import implements_protocol
from aiperf.common.enums.data_exporter_enums import ConsoleExporterType
from aiperf.common.factories import ConsoleExporterFactory
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models.telemetry_models import TelemetryResults
from aiperf.common.protocols import ConsoleExporterProtocol
from aiperf.exporters.display_units_utils import normalize_endpoint_display
from aiperf.exporters.exporter_config import ExporterConfig


@implements_protocol(ConsoleExporterProtocol)
@ConsoleExporterFactory.register(ConsoleExporterType.TELEMETRY)
class GPUTelemetryConsoleExporter(AIPerfLoggerMixin):
    """Console exporter for GPU telemetry data.

    Displays GPU metrics in a table format similar to other console exporters.
    Only displays when verbose mode is enabled.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self._results = exporter_config.results
        self._service_config = exporter_config.service_config
        self._exporter_config = exporter_config
        self._telemetry_results = exporter_config.telemetry_results

    async def export(self, console: Console) -> None:
        """Export telemetry data to console if verbose mode is enabled.

        Only displays telemetry data when verbose mode is enabled in service config.
        Skips display if no telemetry data is available.

        Args:
            console: Rich Console instance for formatted output
        """

        if not self._service_config.verbose:
            return

        if not self._telemetry_results:
            return

        self._print_renderable(console, self.get_renderable(self._telemetry_results))

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

    def get_renderable(self, telemetry_results: TelemetryResults) -> RenderableType:
        """Create Rich tables showing GPU telemetry metrics with consolidated single-table format.

        Generates formatted output with:
        - Summary header showing endpoint reachability status
        - Per-GPU tables with metrics (power, utilization, temperature, etc.)
        - Statistical summaries (avg, min, max, p99, p90, p75, std) for each metric
        - Error summary if no data was collected

        Args:
            telemetry_results: TelemetryResults containing GPU data hierarchy and metadata

        Returns:
            RenderableType: Rich Group containing multiple Tables, or Text message if no data
        """

        renderables = []

        telemetry_data = telemetry_results.telemetry_data

        first_table = True
        for dcgm_url, gpus_data in telemetry_data.dcgm_endpoints.items():
            if not gpus_data:
                continue

            endpoint_display = normalize_endpoint_display(dcgm_url)

            for _gpu_uuid, gpu_data in gpus_data.items():
                gpu_index = gpu_data.metadata.gpu_index
                gpu_name = gpu_data.metadata.model_name

                table_title_base = f"{endpoint_display} | GPU {gpu_index} | {gpu_name}"

                if first_table:
                    first_table = False

                    title_lines = []
                    title_lines.append("NVIDIA AIPerf | GPU Telemetry Summary")

                    endpoints_tested = telemetry_results.endpoints_tested
                    endpoints_successful = telemetry_results.endpoints_successful
                    total_count = len(endpoints_tested)
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

                    for endpoint in endpoints_tested:
                        clean_endpoint = normalize_endpoint_display(endpoint)
                        if endpoint in endpoints_successful:
                            title_lines.append(f"[green]• {clean_endpoint} ✅[/green]")
                        else:
                            title_lines.append(
                                f"[red]• {clean_endpoint} ❌ (unreachable)[/red]"
                            )

                    title_lines.append("")
                    title_lines.append(table_title_base)
                    table_title = "\n".join(title_lines)
                else:
                    renderables.append(Text(""))
                    table_title = table_title_base

                metrics_table = Table(
                    show_header=True, title=table_title, title_style="italic"
                )
                metrics_table.add_column("Metric", justify="right", style="cyan")
                metrics_table.add_column("avg", justify="right", style="green")
                metrics_table.add_column("min", justify="right", style="green")
                metrics_table.add_column("max", justify="right", style="green")
                metrics_table.add_column("p99", justify="right", style="green")
                metrics_table.add_column("p90", justify="right", style="green")
                metrics_table.add_column("p75", justify="right", style="green")
                metrics_table.add_column("std", justify="right", style="green")

                metrics_to_display = [
                    ("Power Usage (W)", "gpu_power_usage", "W"),
                    ("Energy Consumption (MJ)", "energy_consumption", "MJ"),
                    ("Utilization (%)", "gpu_utilization", "%"),
                    ("Memory Used (GB)", "gpu_memory_used", "GB"),
                    ("Temperature (°C)", "gpu_temperature", "°C"),
                    ("SM Clock (MHz)", "sm_clock_frequency", "MHz"),
                ]

                for metric_display, metric_key, unit in metrics_to_display:
                    try:
                        metric_result = gpu_data.get_metric_result(
                            metric_key, metric_key, metric_display, unit
                        )

                        metrics_table.add_row(
                            metric_display,
                            f"{metric_result.avg:.2f}"
                            if metric_result.avg is not None
                            else "N/A",
                            f"{metric_result.min:.2f}"
                            if metric_result.min is not None
                            else "N/A",
                            f"{metric_result.max:.2f}"
                            if metric_result.max is not None
                            else "N/A",
                            f"{metric_result.p99:.2f}"
                            if metric_result.p99 is not None
                            else "N/A",
                            f"{metric_result.p90:.2f}"
                            if metric_result.p90 is not None
                            else "N/A",
                            f"{metric_result.p75:.2f}"
                            if metric_result.p75 is not None
                            else "N/A",
                            f"{metric_result.std:.2f}"
                            if metric_result.std is not None
                            else "N/A",
                        )
                    except Exception as e:
                        self.debug(
                            f"Failed to retrieve metric {metric_key} for GPU {gpu_index}: {e}"
                        )
                        continue

                renderables.append(metrics_table)

        if not renderables:
            message_parts = [
                "No GPU telemetry data collected during the benchmarking run."
            ]

            endpoints_tested = telemetry_results.endpoints_tested
            endpoints_successful = telemetry_results.endpoints_successful
            failed_endpoints = [
                ep for ep in endpoints_tested if ep not in endpoints_successful
            ]

            if failed_endpoints:
                message_parts.append("\n\nUnreachable endpoints:")
                for endpoint in failed_endpoints:
                    clean_endpoint = normalize_endpoint_display(endpoint)
                    message_parts.append(f"  • {clean_endpoint}")

            if telemetry_results.error_summary:
                message_parts.append("\n\nErrors encountered:")
                for error_count in telemetry_results.error_summary:
                    error = error_count.error_details
                    count = error_count.count
                    if count > 1:
                        message_parts.append(
                            f"  • {error.message} ({count} occurrences)"
                        )
                    else:
                        message_parts.append(f"  • {error.message}")

            return Text(
                "".join(message_parts),
                style="dim italic",
            )

        return Group(*renderables)
