# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for analyzing mooncake traces."""

from pathlib import Path

from cyclopts import App

from aiperf.dataset.synthesis import PrefixAnalyzer

analyze_app = App(
    name="analyze-trace", help="Analyze mooncake trace for prefix statistics"
)


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
    print(f"Cache hit rate:        {stats.cache_hit_rate:.2%}")
    print(f"Prefix reuse ratio:    {stats.prefix_reuse_ratio:.2%}")
    print()
    print("ISL (Input Sequence Length):")
    print(f"  Min:     {stats.min_isl:,}")
    print(f"  Max:     {stats.max_isl:,}")
    print(f"  Average: {stats.avg_isl:,.1f}")
    print()
    print("OSL (Output Sequence Length):")
    print(f"  Min:     {stats.min_osl:,}")
    print(f"  Max:     {stats.max_osl:,}")
    print(f"  Average: {stats.avg_osl:,.1f}")
    print(f"{'=' * 60}\n")

    # Save to file if specified
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(stats.model_dump_json(indent=2))
        print(f"Analysis report saved to {output_file}")
