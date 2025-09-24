# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from rich.console import Console, RenderableType
from rich.table import Table
from rich.text import Text

from aiperf.common.decorators import implements_protocol
from aiperf.common.enums.data_exporter_enums import ConsoleExporterType
from aiperf.common.factories import ConsoleExporterFactory
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models.telemetry_models import TelemetryHierarchy
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

    async def export(self, console: Console) -> None:
        """Export telemetry data to console if verbose mode is enabled."""
        if not self._service_config.verbose:
            self.debug("Verbose mode not enabled, skipping telemetry console export")
            return

        if (
            not hasattr(self._results, "telemetry_data")
            or not self._results.telemetry_data
        ):
            self.debug("No telemetry data available for console export")
            return

        self._print_renderable(
            console, self.get_renderable(self._results.telemetry_data, console)
        )

    def _print_renderable(self, console: Console, renderable: RenderableType) -> None:
        """Print the renderable to the console."""
        console.print("\n")
        console.print(renderable)
        console.file.flush()

    def get_renderable(
        self, telemetry_data: TelemetryHierarchy, console: Console
    ) -> RenderableType:
        """Create a Rich table showing GPU telemetry metrics."""
        table = Table(title="NVIDIA AIPerf | GPU Telemetry Metrics")

        # Add columns for GPU information and key metrics
        table.add_column("DCGM Endpoint", justify="left", style="cyan")
        table.add_column("GPU", justify="left", style="green")
        table.add_column("Model", justify="left", style="blue")
        table.add_column("Power Usage", justify="right", style="yellow")
        table.add_column("GPU Util", justify="right", style="yellow")
        table.add_column("Memory Used", justify="right", style="yellow")
        table.add_column("Temperature", justify="right", style="red")

        # Process data from hierarchy
        for dcgm_url, gpus_data in telemetry_data.dcgm_endpoints.items():
            for _gpu_uuid, gpu_data in gpus_data.items():
                # Get summary statistics for key metrics
                try:
                    power_result = gpu_data.get_metric_result(
                        "gpu_power_usage", "power", "Power Usage", "W"
                    )
                    power_display = f"{power_result.avg:.1f}W"
                except Exception:
                    power_display = "N/A"

                try:
                    util_result = gpu_data.get_metric_result(
                        "gpu_utilization", "util", "GPU Utilization", "%"
                    )
                    util_display = f"{util_result.avg:.1f}%"
                except Exception:
                    util_display = "N/A"

                try:
                    memory_result = gpu_data.get_metric_result(
                        "gpu_memory_used", "memory", "Memory Used", "GB"
                    )
                    memory_display = f"{memory_result.avg:.1f}GB"
                except Exception:
                    memory_display = "N/A"

                try:
                    temp_result = gpu_data.get_metric_result(
                        "gpu_temperature", "temp", "GPU Temperature", "°C"
                    )
                    temp_display = f"{temp_result.avg:.1f}°C"
                except Exception:
                    temp_display = "N/A"

                # Truncate endpoint URL for display
                endpoint_display = dcgm_url.replace("http://", "").replace(
                    "/metrics", ""
                )
                if len(endpoint_display) > 20:
                    endpoint_display = f"{endpoint_display[:17]}..."

                # Truncate model name for display
                model_display = gpu_data.metadata.model_name
                if len(model_display) > 25:
                    model_display = f"{model_display[:22]}..."

                table.add_row(
                    endpoint_display,
                    f"GPU {gpu_data.metadata.gpu_index}",
                    model_display,
                    power_display,
                    util_display,
                    memory_display,
                    temp_display,
                )

        if table.row_count == 0:
            # Return a message if no data is available
            return Text(
                "No GPU telemetry data collected during the benchmarking run.",
                style="dim italic",
            )

        return table
