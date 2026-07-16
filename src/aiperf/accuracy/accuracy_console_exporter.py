# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiperf.common.exceptions import ConsoleExporterDisabled
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.exporters.exporter_config import ExporterConfig

if TYPE_CHECKING:
    from rich.console import Console

    from aiperf.accuracy.models import AccuracySummary


class AccuracyConsoleExporter(AIPerfLoggerMixin):
    """Console exporter for accuracy benchmarking results.

    Renders a Rich table with per-task accuracy breakdown and overall score,
    sourced from the structured ``AccuracySummary`` delivered on the dedicated
    accuracy channel.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs: Any) -> None:
        accuracy_cfg = exporter_config.cfg.accuracy
        if accuracy_cfg is None or not accuracy_cfg.enabled:
            raise ConsoleExporterDisabled(
                "Accuracy console exporter is disabled: accuracy mode is not enabled"
            )

        super().__init__(**kwargs)
        self.exporter_config = exporter_config

    async def export(self, console: Console) -> None:
        """Render accuracy results as a Rich table to the given console.

        Prints a per-task breakdown (passed / total / accuracy%) followed by an
        OVERALL row. Does nothing when no accuracy summary was delivered.
        """
        from rich.table import Table

        summary = self.exporter_config.accuracy_results
        if summary is None:
            return

        table = Table(title="Accuracy Benchmark Results", show_lines=True)
        table.add_column("Task", style="cyan", min_width=30)
        table.add_column("Correct", justify="right")
        table.add_column("Total", justify="right")
        table.add_column("Unparsed", justify="right", style="yellow")
        table.add_column("Accuracy", justify="right", style="bold")

        for task_name, stats in sorted(summary.per_task.items()):
            table.add_row(
                task_name,
                str(stats.passed),
                str(stats.total),
                str(stats.unparsed),
                f"{stats.accuracy_rate:.2%}",
            )

        if summary.total_evaluated:
            table.add_row(
                "[bold]OVERALL[/bold]",
                str(summary.total_passed),
                str(summary.total_evaluated),
                str(summary.overall_unparsed),
                f"[bold green]{summary.accuracy_rate:.2%}[/bold green]",
                style="on dark_green",
            )

        console.print()
        console.print(table)

        self._maybe_warn_all_unparsed(console, summary)

    def _maybe_warn_all_unparsed(
        self,
        console: Console,
        summary: AccuracySummary,
    ) -> None:
        """Loud-but-actionable diagnostic for the "accuracy=0 because the server, not the model" case.

        Triggers when every evaluated response reports unparsed output — almost
        always a mock server or misconfigured endpoint, not an accuracy
        problem. Does not gate on total count so it fires on tiny smoke runs.
        """
        if not (
            summary.total_evaluated
            and summary.overall_unparsed >= summary.total_evaluated
        ):
            return
        # Console-only diagnostic: export() legitimately runs once per target
        # console (fixed-width recording pass + live terminal pass), so it
        # must not carry side effects beyond the passed console.
        console.print(
            "[bold yellow]Warning:[/bold yellow] every accuracy "
            "response was unparsed (accuracy=0). The grader could "
            "not extract an answer from any model output. Verify "
            "the inference server returns valid completions for "
            "this benchmark before trusting the accuracy CSV."
        )
