# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Console / panel output helpers for the SystemController.

Extracted to keep ``system_controller.py`` under the ergonomics file-size
limit. Methods are bundled as a mixin because they read a grab-bag of instance
state (results, exit errors, run config, memory tracker) that is expensive to
thread through as function arguments.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel

from aiperf.common.memory_tracker import MemoryPhase, MemoryReading
from aiperf.config.defaults import OutputDefaults
from aiperf.controller.controller_utils import print_exit_errors

if TYPE_CHECKING:
    from aiperf.exporters.exporter_manager import ExporterManager


class SystemControllerOutputMixin:
    """Console / panel output helpers for :class:`SystemController`."""

    def _print_cancel_warning(self) -> None:
        """Print prominent warning panel on first Ctrl+C.

        Informs user that the benchmark is being cancelled gracefully and
        results are being processed. Also instructs how to force quit.

        Uses stderr to ensure visibility even when stdout is redirected or
        captured by the UI.
        """
        console = Console(file=sys.stderr, force_terminal=True)
        console.print()
        console.print(
            Panel(
                "[bold yellow]⚠️  BENCHMARK CANCELLED[/bold yellow]\n\n"
                "Stopping credit issuance and cancelling in-flight requests...\n"
                "Results will be written to files.\n\n"
                "[dim]Press Ctrl+C again to force quit immediately[/dim]\n"
                "[dim](results may be incomplete or not written)[/dim]",
                border_style="yellow",
                padding=(1, 2),
                title="[bold yellow]Cancellation in Progress[/bold yellow]",
            )
        )
        console.print()
        console.file.flush()

    def _print_force_quit_warning(self) -> None:
        """Print warning panel on second Ctrl+C (force quit).

        Warns user that results may be incomplete due to immediate termination.

        Uses stderr to ensure visibility even when stdout is redirected or
        captured by the UI.
        """
        console = Console(file=sys.stderr, force_terminal=True)
        console.print()
        console.print(
            Panel(
                "[bold red]🛑 FORCE QUIT[/bold red]\n\n"
                "Terminating all processes immediately.\n"
                "Results may be incomplete or not written to files.",
                border_style="red",
                padding=(1, 2),
                title="[bold red]Force Quit[/bold red]",
            )
        )
        console.print()
        console.file.flush()

    def _print_exit_errors_and_log_file(self) -> None:
        """Print post exit errors and log file info to the console."""
        console = Console()
        print_exit_errors(self._exit_errors, console=console)
        self._print_log_file_info(console)
        console.print()
        console.file.flush()

    async def _print_post_benchmark_info_and_metrics(self) -> None:
        """Print post benchmark info and metrics to the console."""
        console = Console()
        if console.width < 100:
            console.width = 100

        if not self._results_exported:
            # Non-K8s path or export didn't happen yet — do it now
            from aiperf.exporters.exporter_manager import ExporterManager

            self._exporter_manager = ExporterManager(
                results=self._profile_results.results,
                config=self.run.cfg,
                telemetry_results=self._telemetry_results,
                server_metrics_results=self._server_metrics_results,
            )
            await self._exporter_manager.export_data()
            self._results_exported = True

        await self._exporter_manager.export_console(console=console)

        console.print()
        self._print_cli_command(console)
        self._print_benchmark_duration(console)
        self._print_exported_file_infos(self._exporter_manager, console)
        self._print_log_file_info(console)
        if self._was_cancelled:
            console.print(
                "[italic yellow]The profile run was cancelled early. Results shown may be incomplete or inaccurate.[/italic yellow]"
            )

        console.print()
        console.file.flush()

    def _print_log_file_info(self, console: Console) -> None:
        """Print the log file info."""
        log_file = (
            self.run.cfg.artifacts.dir
            / OutputDefaults.LOG_FOLDER
            / OutputDefaults.LOG_FILE
        )
        console.print(
            f"[bold green]Log File:[/bold green] [cyan]{log_file.resolve()}[/cyan]"
        )

    def _print_exported_file_infos(
        self, exporter_manager: ExporterManager, console: Console
    ) -> None:
        """Print the exported file infos."""
        file_infos = exporter_manager.get_exported_file_infos()
        for file_info in file_infos:
            console.print(
                f"[bold green]{file_info.export_type}[/bold green]: [cyan]{file_info.file_path.resolve()}[/cyan]"
            )

    def _print_cli_command(self, console: Console) -> None:
        """Print the CLI command that was used to run the benchmark."""
        cli_command = self.run.cfg.artifacts.cli_command or "N/A"
        console.print(
            f"[bold green]CLI Command:[/bold green] [italic]{cli_command}[/italic]"
        )

    def _print_benchmark_duration(self, console: Console) -> None:
        """Print the duration of the benchmark."""
        from aiperf.metrics.types.benchmark_duration_metric import (
            BenchmarkDurationMetric,
        )

        # Metrics are already in display units from summarize()
        duration = self._profile_results.get(BenchmarkDurationMetric.tag)
        if duration:
            duration_str = f"[bold green]{BenchmarkDurationMetric.header}[/bold green]: {duration.avg:.2f} {duration.unit}"
            if self._was_cancelled:
                duration_str += " [italic yellow](cancelled early)[/italic yellow]"
            console.print(duration_str)

    def _print_process_memory_summary(self) -> None:
        """Print memory summary for all AIPerf processes."""
        controller_pss_start = getattr(self, "_controller_pss_at_start", None)
        if controller_pss_start is not None:
            self._memory_tracker.record(
                label="SystemController",
                group="controller",
                pid=os.getpid(),
                phase=MemoryPhase.STARTUP,
                reading=MemoryReading(pss=controller_pss_start),
            )
        self._memory_tracker.capture(
            label="SystemController",
            group="controller",
            pid=os.getpid(),
            phase=MemoryPhase.SHUTDOWN,
        )

        self._memory_tracker.print_summary(title="AIPerf Process Memory")
