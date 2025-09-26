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
        super().__init__(**kwargs)
        self._results = exporter_config.results
        self._service_config = exporter_config.service_config
        self._exporter_config = exporter_config
        self._telemetry_results = getattr(exporter_config, "telemetry_results", None)

    async def export(self, console: Console) -> None:
        """Export telemetry data to console if verbose mode is enabled."""

        if not self._service_config.verbose:
            return

        # Check if we have telemetry results (using same pattern as CSV exporter)
        if not self._telemetry_results:
            self.error(
                "❌ GPUTelemetryConsoleExporter: No telemetry results available for console export"
            )
            return

        telemetry_data = self._telemetry_results.telemetry_data

        if not telemetry_data or not telemetry_data.dcgm_endpoints:
            self.error(
                "❌ GPUTelemetryConsoleExporter: No telemetry data available for console export"
            )
            self.error(
                f"🔧 GPUTelemetryConsoleExporter DEBUG: telemetry_data exists = {telemetry_data is not None}"
            )
            if telemetry_data:
                self.error(
                    f"🔧 GPUTelemetryConsoleExporter DEBUG: dcgm_endpoints = {telemetry_data.dcgm_endpoints}"
                )
            return

        endpoint_count = len(telemetry_data.dcgm_endpoints)
        self.error(
            f"✅ GPUTelemetryConsoleExporter: About to display telemetry data from {endpoint_count} endpoints (verbose mode)"
        )

        self._print_renderable(
            console, self.get_renderable(self._telemetry_results, console)
        )

    def _print_renderable(self, console: Console, renderable: RenderableType) -> None:
        """Print the renderable to the console."""
        console.print("\n")
        console.print(renderable)
        console.file.flush()

    def get_renderable(
        self, telemetry_results: TelemetryResults, console: Console
    ) -> RenderableType:
        """Create Rich tables showing GPU telemetry metrics with per-metric statistical breakdown."""

        renderables = []

        # Add endpoint reachability summary at the top
        renderables.extend(self._create_endpoint_summary(telemetry_results))

        # Get the actual telemetry data
        telemetry_data = telemetry_results.telemetry_data

        # Create separate table for each DCGM endpoint
        for dcgm_url, gpus_data in telemetry_data.dcgm_endpoints.items():
            if not gpus_data:
                continue

            # Add endpoint header (more prominent)
            endpoint_display = dcgm_url.replace("http://", "").replace("/metrics", "")
            renderables.append(
                Text(f"\nGPU TELEMETRY: {endpoint_display}", style="bold cyan on black")
            )

            # Define metrics to display with full statistics
            metrics_to_display = [
                ("GPU Power Usage", "gpu_power_usage", "W"),
                ("GPU Power Limit", "gpu_power_limit", "W"),
                ("Energy Consumption", "energy_consumption", "MJ"),
                ("GPU Utilization", "gpu_utilization", "%"),
                ("GPU Memory Used", "gpu_memory_used", "GB"),
                ("GPU Temperature", "gpu_temperature", "°C"),
            ]

            # Create table for each metric (ensure left alignment)
            for metric_display, metric_key, unit in metrics_to_display:
                metric_table = Table(
                    title=f"{metric_display} ({unit})",
                    title_justify="left",
                    show_header=True,
                    header_style="bold magenta",
                )

                # Add columns: GPU Index, GPU Name, and statistics
                metric_table.add_column("GPU Index", justify="center", style="cyan")
                metric_table.add_column("GPU Name", justify="left", style="blue")
                metric_table.add_column("avg", justify="right", style="green")
                metric_table.add_column("min", justify="right", style="green")
                metric_table.add_column("max", justify="right", style="green")
                metric_table.add_column("p99", justify="right", style="green")
                metric_table.add_column("p90", justify="right", style="green")
                metric_table.add_column("p75", justify="right", style="green")

                # Add data for each GPU
                has_data = False
                for _gpu_uuid, gpu_data in gpus_data.items():
                    try:
                        metric_result = gpu_data.get_metric_result(
                            metric_key, metric_key, metric_display, unit
                        )

                        # Truncate GPU name if too long
                        gpu_name = gpu_data.metadata.model_name
                        if len(gpu_name) > 40:
                            gpu_name = f"{gpu_name[:37]}..."

                        metric_table.add_row(
                            str(gpu_data.metadata.gpu_index),
                            gpu_name,
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
                        )
                        has_data = True
                    except Exception:
                        # Skip metrics without data
                        continue

                # Only add table if it has data
                if has_data:
                    renderables.append(metric_table)
                    renderables.append(Text(""))  # Spacing between tables

        if not renderables:
            return Text(
                "No GPU telemetry data collected during the benchmarking run.",
                style="dim italic",
            )

        # Return all tables in a vertical layout (each on new line)
        return Group(*renderables)

    def _create_endpoint_summary(
        self, telemetry_results: TelemetryResults
    ) -> list[RenderableType]:
        """Create endpoint reachability summary display."""
        renderables = []

        endpoints_tested = telemetry_results.endpoints_tested
        endpoints_successful = telemetry_results.endpoints_successful

        # Calculate failed endpoints
        endpoints_failed = [
            ep for ep in endpoints_tested if ep not in endpoints_successful
        ]

        total_count = len(endpoints_tested)
        successful_count = len(endpoints_successful)
        failed_count = len(endpoints_failed)

        # Add summary header
        if failed_count == 0:
            status_text = (
                f"✅ {successful_count}/{total_count} DCGM endpoints reachable"
            )
            status_style = "bold green"
        elif successful_count == 0:
            status_text = (
                f"❌ {successful_count}/{total_count} DCGM endpoints reachable"
            )
            status_style = "bold red"
        else:
            status_text = (
                f"⚠️  {successful_count}/{total_count} DCGM endpoints reachable"
            )
            status_style = "bold yellow"

        renderables.append(Text("GPU TELEMETRY SUMMARY", style="bold cyan"))
        renderables.append(Text(status_text, style=status_style))

        # Add detailed endpoint list if there are any tested endpoints
        if total_count > 0:
            for endpoint in endpoints_tested:
                clean_endpoint = endpoint.replace("http://", "").replace("/metrics", "")
                if endpoint in endpoints_successful:
                    renderables.append(Text(f"   • {clean_endpoint} ✅", style="green"))
                else:
                    renderables.append(
                        Text(f"   • {clean_endpoint} ❌ (unreachable)", style="red")
                    )

        # Add spacing after summary
        renderables.append(Text(""))

        return renderables
