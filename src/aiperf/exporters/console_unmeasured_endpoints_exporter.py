# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from rich.console import Console
from rich.panel import Panel

from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.exporters.exporter_config import ExporterConfig


class ConsoleUnmeasuredEndpointsExporter(AIPerfLoggerMixin):
    """Report requests that per-row endpoint routing sent to another endpoint.

    A dataset row can name its own endpoint (``Turn.endpoint_type``, currently
    authored by the mooncake_trace loader). Those requests are issued and their
    failures counted, but their metrics are not comparable with the run-level
    endpoint's -- an embeddings response has no output tokens, so TTFT/ITL/OSL
    are undefined for it -- so they are excluded from the metric tables.

    Without this notice the request counts in the metrics table would silently
    fail to add up to the number of rows in the trace.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self._results = exporter_config.results
        self._endpoint_type = (
            str(exporter_config.cfg.endpoint.type) if exporter_config.cfg else None
        )

    async def export(self, console: Console) -> None:
        """Print the unmeasured-endpoint summary, if any requests were routed away."""
        counts = getattr(self._results, "unmeasured_request_counts", None)
        if not counts:
            self.debug("No per-row endpoint routing detected, skipping notice")
            return

        console.print()
        console.print(
            Panel(
                self._create_text(counts),
                title="Requests Not Measured",
                border_style="bold cyan",
                title_align="center",
                padding=(0, 2),
                expand=False,
            )
        )
        console.file.flush()

    def _create_text(self, counts: dict[str, int]) -> str:
        """Format the per-endpoint counts and explain why they are excluded."""
        per_endpoint = "\n".join(
            f"  - [cyan]{endpoint}[/cyan]: {count:,} request{'s' if count != 1 else ''}"
            for endpoint, count in sorted(counts.items())
        )
        total = sum(counts.values())
        return f"""\
[bold]{total:,} request(s) were sent to an endpoint other than [cyan]{self._endpoint_type}[/cyan] by per-row dataset routing.[/bold]
{per_endpoint}

[bold]Why:[/bold] Metric applicability depends on the endpoint, so mixing these
records into the tables above would compute token-based metrics over requests
that produce no tokens. They were still issued on schedule, and any failures
are included in the error counts.

[bold]Note:[/bold] The metrics above describe [cyan]{self._endpoint_type}[/cyan] requests only.\
"""
