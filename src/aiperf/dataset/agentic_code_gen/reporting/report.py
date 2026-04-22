# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report generation for synthesized Agentic Code datasets."""

from __future__ import annotations

from pathlib import Path

import orjson
from rich.console import Console
from rich.table import Table

from aiperf.dataset.agentic_code_gen.models import (
    DatasetManifest,
    PercentileStats,
    SynthesizedSession,
)
from aiperf.dataset.agentic_code_gen.reporting.cache_explorer import (
    render_cache_explorer,
    write_cache_structure,
)
from aiperf.dataset.agentic_code_gen.reporting.comparison import render_comparison_text
from aiperf.dataset.agentic_code_gen.reporting.metrics import (
    ReportData,
    TargetComparison,
    _percentile_stats,
    build_report_data,
    extract_cache_metrics,
    extract_metrics,
)
from aiperf.dataset.agentic_code_gen.reporting.plot_report import render_plot_report
from aiperf.dataset.agentic_code_gen.reporting.trace import (
    ParsedTurn,
    group_sessions,
    load_jsonl,
    synthesized_sessions_to_parsed,
)

__all__ = [
    "ParsedTurn",
    "PercentileStats",
    "ReportData",
    "TargetComparison",
    "build_report_data",
    "extract_cache_metrics",
    "extract_metrics",
    "generate_report",
    "group_sessions",
    "load_jsonl",
    "render_cache_explorer",
    "render_comparison_text",
    "render_plot_report",
    "render_text_report",
    "write_cache_structure",
    "write_generated_reports",
]


def render_text_report(data: ReportData) -> str:
    """Render report tables to plain text."""
    console = Console(width=140, record=True)

    console.print()
    console.print("[bold]Dataset Report[/bold]")
    console.print(f"Sessions: {data.session_count:,}   Turns: {data.total_turns:,}")
    console.print()

    _print_target_table(console, data)
    _print_summary_table(console, data)
    _render_cache_table(console, data)

    return console.export_text()


def write_generated_reports(
    sessions: list[SynthesizedSession],
    manifest: DatasetManifest,
    quality_dict: dict,
    output_dir: Path,
) -> None:
    """Write report artifacts derived from freshly synthesized sessions."""
    parsed_sessions = synthesized_sessions_to_parsed(sessions)

    cache_payload = write_cache_structure(parsed_sessions, manifest, output_dir)
    render_cache_explorer(output_dir, cache_payload)

    metrics = extract_metrics(parsed_sessions)
    metrics.update(
        extract_cache_metrics(
            parsed_sessions,
            block_size=manifest.generation_params.block_size,
        )
    )
    render_plot_report(metrics, parsed_sessions, output_dir)

    sd_stats = _percentile_stats(metrics["session_duration_min"])
    comparison = render_comparison_text(quality_dict, session_duration_stats=sd_stats)
    (output_dir / "comparison.txt").write_text(comparison)


def _print_report_to_console(data: ReportData) -> None:
    """Print report tables directly to the terminal."""
    console = Console(width=140)
    console.print()
    console.print(
        f"[bold]Dataset Report[/bold]  Sessions: {data.session_count:,}  Turns: {data.total_turns:,}"
    )
    console.print()
    _print_target_table(console, data)
    _print_summary_table(console, data)
    _render_cache_table(console, data)


def _print_target_table(console: Console, data: ReportData) -> None:
    table = Table(title="Target vs Observed")
    table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    for col in [
        "Target Mean",
        "Target Median",
        "Obs Mean",
        "Obs Median",
        "p05",
        "p25",
        "p75",
        "p95",
        "p99",
        "% Err",
    ]:
        table.add_column(col, justify="right", style="green", no_wrap=True)

    for comparison in data.comparisons:
        table.add_row(
            comparison.metric_name,
            f"{comparison.target_mean:,.0f}"
            if comparison.target_mean is not None
            else "-",
            f"{comparison.target_median:,.0f}"
            if comparison.target_median is not None
            else "-",
            f"{comparison.observed.mean:,.1f}",
            f"{comparison.observed.median:,.1f}",
            f"{comparison.observed.p05:,.1f}",
            f"{comparison.observed.p25:,.1f}",
            f"{comparison.observed.p75:,.1f}",
            f"{comparison.observed.p95:,.1f}",
            f"{comparison.observed.p99:,.1f}",
            f"{comparison.pct_error_mean:.1f}%"
            if comparison.pct_error_mean is not None
            else "-",
        )
    console.print(table)
    console.print()


def _print_summary_table(console: Console, data: ReportData) -> None:
    table = Table(title="Summary Statistics")
    table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    for col in ["Mean", "Median", "p05", "p25", "p75", "p95", "p99"]:
        table.add_column(col, justify="right", style="green", no_wrap=True)

    for name, stats in [
        ("Hash ID Blocks/Turn", data.hash_id_block_stats),
        ("Request Latency (ms)", data.request_latency_stats),
        ("Session Duration (min)", data.session_duration_min_stats),
    ]:
        _add_stats_row(table, name, stats)
    console.print(table)
    console.print()


def _render_cache_table(console: Console, data: ReportData) -> None:
    """Print cache/prefix statistics table if data is available."""
    cache_rows = [
        ("Prefix Length", data.prefix_length_stats),
        ("Unique Prompt Length", data.unique_prompt_length_stats),
        ("Prefix Ratio", data.prefix_ratio_stats),
        ("Sequential Cache Hit Rate", data.sequential_cache_hit_rate_stats),
        ("Per-Session Cache Hit Rate", data.per_session_cache_hit_rate_stats),
    ]
    if not any(stats for _, stats in cache_rows):
        return

    table = Table(title="Cache / Prefix Statistics")
    table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    for col in ["Mean", "Median", "p05", "p25", "p75", "p95", "p99"]:
        table.add_column(col, justify="right", style="green", no_wrap=True)

    for name, stats in cache_rows:
        if stats is None:
            continue
        _add_stats_row(table, name, stats, compact=stats.mean <= 1.0)
    console.print(table)
    console.print()


def _add_stats_row(
    table: Table, name: str, stats: PercentileStats, compact: bool = False
) -> None:
    fmt = ".4f" if compact else ",.1f"
    table.add_row(
        name,
        f"{stats.mean:{fmt}}",
        f"{stats.median:{fmt}}",
        f"{stats.p05:{fmt}}",
        f"{stats.p25:{fmt}}",
        f"{stats.p75:{fmt}}",
        f"{stats.p95:{fmt}}",
        f"{stats.p99:{fmt}}",
    )


def generate_report(
    run_dir: Path,
    fmt: str = "text",
    prefill_tps: float = 20_000,
    decode_tps: float = 60,
) -> ReportData:
    """Load a run directory and generate the requested report artifacts."""
    allowed_formats = {"text", "plot", "both"}
    if fmt not in allowed_formats:
        allowed = ", ".join(sorted(allowed_formats))
        raise ValueError(f"fmt must be one of: {allowed}")

    jsonl_path = run_dir / "dataset.jsonl"
    manifest_path = run_dir / "manifest.json"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"dataset.jsonl not found in {run_dir}")

    manifest: DatasetManifest | None = None
    if manifest_path.exists():
        manifest = DatasetManifest(**orjson.loads(manifest_path.read_bytes()))

    sessions = group_sessions(load_jsonl(jsonl_path))
    metrics = extract_metrics(sessions, prefill_tps=prefill_tps, decode_tps=decode_tps)

    block_size = manifest.generation_params.block_size if manifest else 512
    metrics.update(extract_cache_metrics(sessions, block_size=block_size))
    report_data = build_report_data(metrics, manifest)

    if fmt in ("text", "both"):
        _print_report_to_console(report_data)
    if fmt in ("plot", "both"):
        render_plot_report(metrics, sessions, run_dir)
        cache_payload = write_cache_structure(sessions, manifest, run_dir)
        render_cache_explorer(run_dir, cache_payload)

    return report_data
