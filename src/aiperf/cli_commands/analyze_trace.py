# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for analyzing mooncake traces."""

from pathlib import Path

from cyclopts import App

from aiperf.dataset.synthesis import MetricStats, PrefixAnalyzer

analyze_app = App(
    name="analyze-trace", help="Analyze mooncake trace for prefix statistics"
)


def _format_metric_stats(name: str, stats: MetricStats | None, indent: int = 2) -> str:
    """Format metric statistics in a vertical layout.

    Args:
        name: Name of the metric.
        stats: MetricStats object or None.
        indent: Number of spaces for indentation.

    Returns:
        Formatted string with statistics.
    """
    if stats is None:
        return f"{name}:\n{' ' * indent}No data\n"

    prefix = " " * indent
    lines = [
        f"{name}:",
        f"{prefix}Mean:    {stats.mean:,.2f}",
        f"{prefix}Std Dev: {stats.std_dev:,.2f}",
        f"{prefix}Min:     {stats.min:,.2f}",
        f"{prefix}P25:     {stats.p25:,.2f}",
        f"{prefix}Median:  {stats.median:,.2f}",
        f"{prefix}P75:     {stats.p75:,.2f}",
        f"{prefix}Max:     {stats.max:,.2f}",
    ]
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
    print(f"{'=' * 60}")
    print(f"Total requests:        {stats.total_requests:,}")
    print(f"Unique prefixes:       {stats.unique_prefixes:,}")
    print(f"Prefix reuse ratio:    {stats.prefix_reuse_ratio:.2%}")
    print()

    # Print extended statistics in vertical layout
    print(_format_metric_stats("Input Length", stats.isl_stats))
    print()
    print(_format_metric_stats("Context Length", stats.context_length_stats))
    print()
    print(
        _format_metric_stats("Unique Prompt Length", stats.unique_prompt_length_stats)
    )
    print()
    print(_format_metric_stats("Output Length", stats.osl_stats))
    print()
    print(_format_metric_stats("Theoretical Hit Rates", stats.hit_rate_stats))
    print(f"{'=' * 60}\n")

    # Save to file if specified
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(stats.model_dump_json(indent=2))
        print(f"Analysis report saved to {output_file}")
