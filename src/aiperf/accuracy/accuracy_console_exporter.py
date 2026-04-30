# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiperf.accuracy.models import (
    ACCURACY_METRIC_PREFIX,
    ACCURACY_OVERALL_TAG,
    ACCURACY_TASK_TAG_PREFIX,
    ACCURACY_UNPARSED_TAG,
    ACCURACY_UNPARSED_TASK_TAG_PREFIX,
)
from aiperf.common.exceptions import ConsoleExporterDisabled
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.exporters.exporter_config import ExporterConfig

if TYPE_CHECKING:
    from rich.console import Console


class AccuracyConsoleExporter(AIPerfLoggerMixin):
    """Console exporter for accuracy benchmarking results.

    Renders a Rich table with per-task accuracy breakdown and overall score.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs: Any) -> None:
        if not exporter_config.user_config.accuracy.enabled:
            raise ConsoleExporterDisabled(
                "Accuracy console exporter is disabled: accuracy mode is not enabled"
            )

        super().__init__(**kwargs)
        self.exporter_config = exporter_config

    async def export(self, console: Console) -> None:
        """Render accuracy results as a Rich table to the given console.

        Prints a per-task breakdown (correct / total / accuracy%) followed by an
        OVERALL row. Does nothing if no ``accuracy.*`` metrics are present in
        ``exporter_config.results``.
        """
        from rich.table import Table

        results = self.exporter_config.results
        if results is None or results.records is None:
            return

        accuracy_metrics = [
            r for r in results.records if r.tag.startswith(ACCURACY_METRIC_PREFIX)
        ]
        if not accuracy_metrics:
            return

        overall = next(
            (m for m in accuracy_metrics if m.tag == ACCURACY_OVERALL_TAG), None
        )
        task_metrics = [
            m for m in accuracy_metrics if m.tag.startswith(ACCURACY_TASK_TAG_PREFIX)
        ]
        unparsed_overall = next(
            (m for m in accuracy_metrics if m.tag == ACCURACY_UNPARSED_TAG), None
        )
        unparsed_by_task: dict[str, int] = {
            m.tag.removeprefix(ACCURACY_UNPARSED_TASK_TAG_PREFIX): int(m.sum or 0)
            for m in accuracy_metrics
            if m.tag.startswith(ACCURACY_UNPARSED_TASK_TAG_PREFIX)
        }

        table = Table(title="Accuracy Benchmark Results", show_lines=True)
        table.add_column("Task", style="cyan", min_width=30)
        table.add_column("Correct", justify="right")
        table.add_column("Total", justify="right")
        table.add_column("Unparsed", justify="right", style="yellow")
        table.add_column("Accuracy", justify="right", style="bold")

        for m in task_metrics:
            task_name = m.tag.removeprefix(ACCURACY_TASK_TAG_PREFIX)
            acc_str = f"{m.current:.2%}" if m.current is not None else "N/A"
            unparsed_count = str(unparsed_by_task.get(task_name, 0))
            table.add_row(
                task_name,
                str(m.sum or 0),
                str(m.count or 0),
                unparsed_count,
                acc_str,
            )

        if overall:
            acc_str = f"{overall.current:.2%}" if overall.current is not None else "N/A"
            overall_unparsed = str(
                int(unparsed_overall.sum or 0) if unparsed_overall else 0
            )
            table.add_row(
                "[bold]OVERALL[/bold]",
                str(overall.sum or 0),
                str(overall.count or 0),
                overall_unparsed,
                f"[bold green]{acc_str}[/bold green]",
                style="on dark_green",
            )

        console.print()
        console.print(table)
