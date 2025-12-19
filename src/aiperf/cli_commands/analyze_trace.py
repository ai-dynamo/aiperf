# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for analyzing mooncake traces."""

from pathlib import Path

from cyclopts import App

from aiperf.dataset.synthesis import MetricStats, PrefixAnalyzer

analyze_app = App(
    name="analyze-trace", help="Analyze mooncake trace for prefix statistics"
)


def _format_stats_table(
    metrics: dict[str, MetricStats | None],
) -> str:
    """Format metric statistics as a table.

    Args:
        metrics: Dictionary mapping metric names to MetricStats objects.

    Returns:
        Formatted table string.
    """
    headers = ["", "Mean", "Std Dev", "Min", "P25", "Median", "P75", "Max"]
    col_widths = [max(len(name) for name in metrics) + 1]
    col_widths.extend([10] * 7)

    # Header row
    header_line = (
        "| " + " | ".join(h.rjust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    )
    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    lines = [separator, header_line, separator]

    # Data rows
    for name, stats in metrics.items():
        if stats is None:
            row = [name.ljust(col_widths[0])] + ["N/A".rjust(10)] * 7
        else:
            row = [
                name.ljust(col_widths[0]),
                f"{stats.mean:,.2f}".rjust(10),
                f"{stats.std_dev:,.2f}".rjust(10),
                f"{stats.min:,.2f}".rjust(10),
                f"{stats.p25:,.2f}".rjust(10),
                f"{stats.median:,.2f}".rjust(10),
                f"{stats.p75:,.2f}".rjust(10),
                f"{stats.max:,.2f}".rjust(10),
            ]
        lines.append("| " + " | ".join(row) + " |")

    lines.append(separator)
    return "\n".join(lines)


@analyze_app.default
def analyze_trace(
    input_file: Path,
    block_size: int = 512,
    output_file: Path | None = None,
) -> None:
    """Analyze a mooncake trace file for ISL/OSL distributions and cache hit rates.

    Args:
        input_file: Path to input mooncake trace JSONL file
        block_size: KV cache block size for analysis (default: 512)
        output_file: Optional output path for analysis report (JSON)
    """
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return

    analyzer = PrefixAnalyzer(block_size=block_size)
    stats = analyzer.analyze_file(input_file)

    # Print to console
    print("\nTrace Analysis Report")
    print(f"Total requests:  {stats.total_requests:,}")
    print(f"Unique prefixes: {stats.unique_prefixes:,}")
    print()

    # Build metrics dictionary for table
    metrics = {
        "Input Length": stats.isl_stats,
        "Context Length": stats.context_length_stats,
        "Unique Prompt Length": stats.unique_prompt_length_stats,
        "Output Length": stats.osl_stats,
        "Theoretical Hit Rates": stats.hit_rate_stats,
    }

    print(_format_stats_table(metrics))
    print()

    # Save to file if specified
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(stats.model_dump_json(indent=2))
        print(f"Analysis report saved to {output_file}")
