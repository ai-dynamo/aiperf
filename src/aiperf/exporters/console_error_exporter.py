# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from rich.console import Console
from rich.table import Table
from rich.text import Text

from aiperf.common.models import ErrorDetailsCount
from aiperf.exporters.exporter_config import ExporterConfig


class ConsoleErrorExporter:
    """A class that exports error data to the console"""

    def __init__(self, exporter_config: ExporterConfig, **kwargs):
        self._results = exporter_config.results

    async def export(self, console: Console) -> None:
        if not self._results.error_summary:
            return

        table = Table(title=self._get_title())
        table.add_column("Code", justify="right", style="yellow")
        table.add_column("Type", justify="right", style="yellow")
        table.add_column("Message", justify="left", style="yellow")
        table.add_column("Count", justify="right", style="yellow")
        self._construct_table(table, self._results.error_summary)

        console.print("\n")
        console.print(table)
        console.file.flush()

    def _construct_table(
        self, table: Table, errors_by_type: list[ErrorDetailsCount]
    ) -> None:
        for error_details_count in errors_by_type:
            table.add_row(*self._format_row(error_details_count))

    def _format_row(self, error_details_count: ErrorDetailsCount) -> list[str | Text]:
        details = error_details_count.error_details
        count = error_details_count.count

        # ``type`` and ``message`` carry server-controlled text verbatim (see
        # ``ErrorDetails`` construction in ``transports/aiohttp_client``), so they
        # are wrapped in ``Text`` to keep Rich from parsing them as console markup.
        # Without this a response body containing a stray closing tag raises
        # ``MarkupError``, and one containing an opening tag is silently swallowed.
        # This matches how ``controller_utils._format_field`` renders the same
        # fields in the exit-error panel.
        return [
            str(details.code) if details.code else "[dim]N/A[/dim]",
            Text(str(details.type)) if details.type else "[dim]N/A[/dim]",
            Text(str(details.message)),
            f"{count:,}",
        ]

    def _get_title(self) -> str:
        return "[red]NVIDIA AIPerf | Error Summary[/red]"
