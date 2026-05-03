# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from rich.console import Console
from rich.panel import Panel

from aiperf.common.environment import Environment
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import MetricResult
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.metrics.types.osl_mismatch_metrics import (
    OSLMismatchCountMetric,
    OSLMismatchDiffMetric,
)
from aiperf.metrics.types.request_count_metric import RequestCountMetric


class ConsoleOSLMismatchExporter(AIPerfLoggerMixin):
    """Display warning panel when actual output length differs significantly from requested OSL.

    This exporter checks if any requests have OSL mismatches exceeding the
    configured threshold and displays a prominent warning panel with:
    - Number and percentage of affected requests
    - Explanation of why this happens
    - Recommended actions to fix the issue
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        if exporter_config.results is None:
            self._metrics_by_tag = {}
        else:
            self._metrics_by_tag = {r.tag: r for r in exporter_config.results.records}
        self._pct_threshold = Environment.METRICS.OSL_MISMATCH_PCT_THRESHOLD
        self._max_token_threshold = Environment.METRICS.OSL_MISMATCH_MAX_TOKEN_THRESHOLD

    async def export(self, console: Console) -> None:
        """Export OSL mismatch warning to console if mismatches detected."""
        metric = self._get_mismatch_metric()
        if not metric or not metric.avg or metric.avg <= 0:
            self.debug(
                "No OSL mismatches detected, skipping output sequence length warning"
            )
            return

        mismatch_count = int(metric.avg)
        total_records = self._get_total_records()
        if not total_records:
            self.debug(
                "No valid records detected, skipping output sequence length warning"
            )
            return
        percentage = (mismatch_count / total_records) * 100
        avg_diff = self._get_avg_diff()

        panel = Panel(
            self._create_warning_text(
                mismatch_count, total_records, percentage, avg_diff
            ),
            title="Output Sequence Length Mismatch Warning",
            border_style="bold yellow",
            title_align="center",
            padding=(0, 2),
            expand=False,
        )

        console.print()
        console.print(panel)
        console.file.flush()

    def _get_mismatch_metric(self) -> MetricResult | None:
        """Extract the OSL mismatch count metric from results."""
        return self._metrics_by_tag.get(OSLMismatchCountMetric.tag)

    def _get_total_records(self) -> int:
        """Get the total number of valid records from results."""
        metric = self._metrics_by_tag.get(RequestCountMetric.tag)
        return int(metric.avg) if metric and metric.avg else 0

    def _get_avg_diff(self) -> float | None:
        """Get the average OSL mismatch diff percentage from results."""
        metric = self._metrics_by_tag.get(OSLMismatchDiffMetric.tag)
        return metric.avg if metric else None

    def _create_warning_text(
        self,
        mismatch_count: int,
        total_records: int,
        percentage: float,
        avg_diff: float | None,
    ) -> str:
        """Create the formatted warning text with details and recommendations."""
        avg_diff_str = f"{avg_diff:.1f}%" if avg_diff is not None else "N/A"
        return f"""\
[bold]{mismatch_count:,} of {total_records:,} requests ({percentage:.1f}%) have output length differing from requested by more than the threshold.[/bold]
[bold]Threshold (tokens):[/bold] min(requested x {self._pct_threshold:g}%, {self._max_token_threshold})
[bold]Average mismatch:[/bold] {avg_diff_str}

[bold]Why:[/bold] Server hit EOS token before reaching requested output length.

[bold]Fix Options:[/bold]
  • [green]--extra-inputs ignore_eos:true[/green] — Generate until max_tokens (vLLM, TensorRT-LLM)
  • [green]--extra-inputs min_tokens:<N>[/green] — Set minimum output length (vLLM, TensorRT-LLM, SGLang)
  • [green]--use-server-token-count[/green] — Use server-reported token counts if tokenizer mismatch suspected

[bold]Diagnostics:[/bold]
  • Review [cyan]profile_export.jsonl[/cyan] → [cyan]osl_mismatch_diff_pct[/cyan] for per-request values
  • Adjust: [green]AIPERF_METRICS_OSL_MISMATCH_PCT_THRESHOLD={self._pct_threshold:g}[/green]
  • Adjust: [green]AIPERF_METRICS_OSL_MISMATCH_MAX_TOKEN_THRESHOLD={self._max_token_threshold}[/green]\
"""
