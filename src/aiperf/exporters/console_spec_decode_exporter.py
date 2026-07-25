# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiperf.common.exceptions import ConsoleExporterDisabled
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.exporters.exporter_config import ExporterConfig

if TYPE_CHECKING:
    from rich.console import Console

# Default cap on how many accepted-draft buckets the one-line console histogram
# shows individually; buckets at or above the cap fold into a trailing ">=CAP"
# bucket so the row stays on one line regardless of the drafter's block size.
SPEC_DECODE_HISTOGRAM_CONSOLE_CAP = 8


def format_acceptance_histogram_line(
    histogram: dict[int, int],
    cap: int = SPEC_DECODE_HISTOGRAM_CONSOLE_CAP,
) -> str | None:
    """Render the pooled accepted-draft histogram as a compact one-line summary.

    Shows the share of verify steps in each accepted-draft bucket ``0 .. min(max
    observed, cap)``. When any bucket ``j >= cap`` has steps, those fold into a
    single trailing ``>=cap`` bucket so the line never grows unbounded. Returns
    None when the histogram is empty (nothing to render).
    """
    total_steps = sum(histogram.values())
    if total_steps <= 0:
        return None

    max_bucket = max(histogram)
    overflow = max_bucket >= cap
    top = cap - 1 if overflow else max_bucket

    parts = [
        f"{bucket}: {histogram.get(bucket, 0) / total_steps * 100:.0f}%"
        for bucket in range(top + 1)
    ]
    if overflow:
        folded = sum(steps for bucket, steps in histogram.items() if bucket >= cap)
        parts.append(f">={cap}: {folded / total_steps * 100:.0f}%")

    return "Accepted-draft histogram (% steps):  " + "   ".join(parts)


class ConsoleSpecDecodeHistogramExporter(AIPerfLoggerMixin):
    """Console exporter for the one-line pooled acceptance histogram.

    Renders the ``pooled_spec_decode_acceptance_histogram`` pooled by the accumulator
    as a single compact row beneath the ``Spec Decode`` scalar table (whose rows
    the main ``ConsoleMetricsExporter`` renders via ``MetricConsoleGroup.SPEC_DECODE``).
    The histogram is a dict, not a ``MetricResult``, so it cannot ride the
    metric-table machinery -- hence this dedicated section, mirroring the
    power-efficiency exporter pattern. Self-disables when no request carried
    spec-decode stats.
    """

    def __init__(self, exporter_config: ExporterConfig, **kwargs: Any) -> None:
        results = exporter_config.results
        if results is None or results.pooled_spec_decode_acceptance_histogram is None:
            raise ConsoleExporterDisabled(
                "Spec-decode histogram console exporter is disabled: no pooled "
                "acceptance histogram in the results."
            )
        super().__init__(**kwargs)
        self._histogram = results.pooled_spec_decode_acceptance_histogram

    async def export(self, console: Console) -> None:
        line = format_acceptance_histogram_line(self._histogram)
        if line is None:
            return
        console.print(f"  [cyan]{line}[/cyan]")
