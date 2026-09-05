# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.exporters.exporter_config import ExporterConfig


class ConsoleGpuVendorDisclaimerExporter(AIPerfLoggerMixin):
    """Console banner warning that the GPU data below it is vendor-specific.

    Rendered ahead of every GPU section (power efficiency and per-GPU telemetry)
    so it's clear that all following metrics are platform-specific and not
    directly comparable across vendors. Shown only when GPU telemetry is enabled
    and at least one GPU reported.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self._cfg = exporter_config.cfg
        self._telemetry_results = exporter_config.telemetry_results

    async def export(self, console: Console) -> None:
        if self._cfg.gpu_telemetry_disabled:
            return
        if not self._telemetry_results:
            return

        console.print("\n")
        console.print(self._create_platform_disclaimer())
        console.file.flush()

    def _create_platform_disclaimer(self) -> Panel:
        """Create the platform-specific comparability warning box."""
        platforms = sorted(
            {
                gpu_summary.platform
                for endpoint_data in self._telemetry_results.endpoints.values()
                for gpu_summary in endpoint_data.gpus.values()
            }
        )
        platform_text = ", ".join(platforms) if platforms else "unknown"
        body = Text(
            f"Platform: {platform_text}\n"
            "Metric semantics are platform-specific; cross-platform comparisons "
            "require workload and collector validation.",
            style="yellow",
        )
        return Panel(
            body,
            title="GPU Telemetry Platform",
            border_style="bold yellow",
            title_align="center",
            padding=(0, 2),
            expand=False,
        )
