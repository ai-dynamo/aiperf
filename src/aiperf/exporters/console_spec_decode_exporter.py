# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common.enums import MetricConsoleGroup
from aiperf.exporters.console_metrics_exporter import ConsoleMetricsExporter

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

    # Two spaces after the label and three between buckets visually separate the
    # label from the data (and the buckets from each other) at a glance.
    return "Accepted drafts per step (% of steps):  " + "   ".join(parts)


class ConsoleSpecDecodeExporter(ConsoleMetricsExporter):
    """Dedicated console section for speculative-decoding acceptance.

    Renders the ``Spec Decode`` scalar table (the ``SPEC_DECODE`` metric group)
    immediately followed by the one-line pooled accepted-draft histogram, as a
    single self-contained block -- mirroring the power-efficiency dedicated-
    section pattern. Keeping the table and histogram in one exporter guarantees
    the histogram sits directly beneath its table (the pooled histogram is a
    dict, not a ``MetricResult``, so it cannot ride the metric-table machinery
    and would otherwise be emitted by a separate exporter that prints after the
    main metrics block). Renders nothing when no request carried spec-decode
    stats.
    """

    console_groups = (MetricConsoleGroup.SPEC_DECODE,)

    def _get_group_title(self, group: MetricConsoleGroup) -> str:
        return f"{self._get_title()}: Spec Decode"

    async def export(self, console: Console) -> None:
        if not self._results or not self._results.records:
            return
        renderable = self.get_renderable(self._results.records, console)
        if renderable is not None:
            self._print_renderable(console, renderable)
        histogram = self._results.pooled_spec_decode_acceptance_histogram
        line = format_acceptance_histogram_line(histogram) if histogram else None
        if line is not None:
            console.print(f"  [cyan]{line}[/cyan]")
