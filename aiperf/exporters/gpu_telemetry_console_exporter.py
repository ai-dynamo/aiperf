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
from aiperf.exporters.exporter_config import ExporterConfig


@implements_protocol(ConsoleExporterProtocol)
@ConsoleExporterFactory.register(ConsoleExporterType.TELEMETRY)
class GPUTelemetryConsoleExporter(AIPerfLoggerMixin):
    """Console exporter for GPU telemetry data.

    Displays GPU metrics in a table format similar to other console exporters.
    Only displays when verbose mode is enabled.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        """
        Initialize the exporter with the given ExporterConfig and capture runtime data needed for rendering telemetry.
        
        Parameters:
            exporter_config (ExporterConfig): Configuration object for the exporter; used to obtain benchmarking results, service configuration, and optional telemetry results.
        """
        super().__init__(**kwargs)
        self._results = exporter_config.results
        self._service_config = exporter_config.service_config
        self._exporter_config = exporter_config
        self._telemetry_results = getattr(exporter_config, "telemetry_results", None)

    async def export(self, console: Console) -> None:
        """
        Export GPU telemetry to the provided console when verbose mode is enabled and telemetry data is available.
        
        If verbose mode is disabled on the service configuration or no telemetry results are present, the method returns without producing output. When data is available, builds a renderable representation and prints it to the console.
        
        Parameters:
            console (Console): Rich Console instance used to render the telemetry output.
        """

        if not self._service_config.verbose:
            return

        if not self._telemetry_results:
            return

        self._print_renderable(
            console, self.get_renderable(self._telemetry_results, console)
        )

    def _print_renderable(self, console: Console, renderable: RenderableType) -> None:
        """
        Print the given renderable to the provided Rich console and flush the console output.
        
        Parameters:
            console (Console): Rich console to receive the printed renderable.
            renderable (RenderableType): The renderable object (table, text, group, etc.) to print.
        
        Side effects:
            Writes a leading newline, prints the renderable to the console, and flushes the console's output buffer.
        """
        console.print("\n")
        console.print(renderable)
        console.file.flush()

    def get_renderable(
        self, telemetry_results: TelemetryResults, console: Console
    ) -> RenderableType:
        """
        Build a Rich renderable that summarizes GPU telemetry by DCGM endpoint and GPU.
        
        Constructs one or more Rich Tables (grouped into a Group) with per-GPU metrics for each reachable DCGM endpoint; if no metric tables can be produced, returns a dim italic Text explaining that no telemetry was collected and listing unreachable endpoints and any error summary.
        
        Parameters:
            telemetry_results (TelemetryResults): Telemetry data and metadata including telemetry_data.dcgm_endpoints, endpoints_tested, endpoints_successful, and optional error_summary.
            console (Console): Rich Console used for rendering context (passed through for consistent styling/width).
        
        Returns:
            RenderableType: A Group containing per-GPU Tables when telemetry is available, or a Text object with diagnostic information when no tables can be rendered.
        """

        renderables = []

        telemetry_data = telemetry_results.telemetry_data

        first_table = True
        for dcgm_url, gpus_data in telemetry_data.dcgm_endpoints.items():
            if not gpus_data:
                continue

            endpoint_display = dcgm_url.replace("http://", "").replace("/metrics", "")

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
                        clean_endpoint = endpoint.replace("http://", "").replace(
                            "/metrics", ""
                        )
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
                    except Exception:
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
                    clean_endpoint = endpoint.replace("http://", "").replace(
                        "/metrics", ""
                    )
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
