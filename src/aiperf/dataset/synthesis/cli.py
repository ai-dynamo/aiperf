# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trace analysis display logic."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from aiperf.dataset.synthesis.models import MetricStats
from aiperf.dataset.synthesis.prefix_analyzer import PrefixAnalyzer

_STAT_COLUMNS = ["Mean", "Std Dev", "Min", "P25", "Median", "P75", "Max"]


def _build_stats_table(metrics: dict[str, MetricStats | None]) -> Table:
    """Build a Rich table with metric statistics."""
    table = Table(title="Trace Statistics")
    table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    for col in _STAT_COLUMNS:
        table.add_column(col, justify="right", style="green", no_wrap=True)

    for name, stats in metrics.items():
        if stats is None:
            table.add_row(name, *["[dim]N/A[/dim]"] * len(_STAT_COLUMNS))
        else:
            table.add_row(
                name,
                f"{stats.mean:,.2f}",
                f"{stats.std_dev:,.2f}",
                f"{stats.min:,.2f}",
                f"{stats.p25:,.2f}",
                f"{stats.median:,.2f}",
                f"{stats.p75:,.2f}",
                f"{stats.max:,.2f}",
            )

    return table


def analyze_trace(
    input_file: Path,
    block_size: int = 512,
    output_file: Path | None = None,
) -> None:
    """Analyze a mooncake trace file for ISL/OSL distributions and cache hit rates."""
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return

    analyzer = PrefixAnalyzer(block_size=block_size)
    stats = analyzer.analyze_file(input_file)

    console = Console(width=120)

    console.print()
    console.print("[bold]Trace Analysis Report[/bold]")
    console.print(f"Total requests:   {stats.total_requests:,}")
    console.print(f"Unique prefixes:  {stats.unique_prefixes:,}")
    console.print(f"Prefix groups:    {stats.num_prefix_groups:,}")
    console.print()

    metrics = {
        "Input Length": stats.isl_stats,
        "Context Length": stats.context_length_stats,
        "Unique Prompt Length": stats.unique_prompt_length_stats,
        "Output Length": stats.osl_stats,
        "Theoretical Hit Rates": stats.hit_rate_stats,
    }

    console.print(_build_stats_table(metrics))
    console.print()

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(stats.model_dump_json(indent=2))
        console.print(f"Analysis report saved to {output_file}")
