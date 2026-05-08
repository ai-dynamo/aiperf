# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI commands for generating HTML reports from real trace files."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from cyclopts import App, Parameter

if TYPE_CHECKING:
    from rich.console import Console

    from aiperf.dataset.agentic_code_gen.reporting.trace import ParsedTurn

app = App(name="report")


@app.default
def report(
    target: Annotated[
        Literal["weka-trace"],
        Parameter(help="Trace flavor to report on."),
    ],
    path: Path,
    *,
    output: Path = Path("."),
    block_size: int | None = None,
    max_context_length: int | None = None,
    no_subagents: bool = False,
    prefill_tps: float = 20_000,
    decode_tps: float = 60,
) -> None:
    """Render HTML reports (report.html, cache_explorer.html, simulation.html)
    for a real trace file or directory.

    Examples:
        aiperf report weka-trace ./traces/
        aiperf report weka-trace ./traces/ --block-size 64
        aiperf report weka-trace ./traces/ --max-context-length 200000
        aiperf report weka-trace ./traces/ --no-subagents

    Args:
        target: Trace flavor (currently only `weka-trace`).
        path: Path to a trace file or a directory of *.json trace files.
        output: Parent directory for the auto-named run directory.
        block_size: KV cache block size for cache statistics; inferred from weka traces when omitted.
        max_context_length: Drop traces whose peak input_length exceeds this.
        no_subagents: Skip subagent sessions; report only parent traces.
        prefill_tps: Synthetic prefill throughput for latency estimates.
        decode_tps: Synthetic decode throughput for latency estimates.
    """
    match target:
        case "weka-trace":
            report_weka_trace(
                path=path,
                output=output,
                block_size=block_size,
                max_context_length=max_context_length,
                no_subagents=no_subagents,
                prefill_tps=prefill_tps,
                decode_tps=decode_tps,
            )


def report_weka_trace(
    *,
    path: Path,
    output: Path = Path("."),
    block_size: int | None = None,
    max_context_length: int | None = None,
    no_subagents: bool = False,
    prefill_tps: float = 20_000,
    decode_tps: float = 60,
) -> None:
    """Render HTML reports for a weka trace file or directory.

    Writes an auto-named run directory `weka-report_<basename>_<UTC-ts>/`
    containing report.html, cache_explorer.html, simulation.html, and
    cache_structure.json.
    """
    from rich.console import Console

    from aiperf.dataset.agentic_code_gen.reporting.weka_input import (
        infer_weka_block_size,
        load_weka_as_parsed,
    )

    console = Console()
    parsed = load_weka_as_parsed(
        path,
        include_subagents=not no_subagents,
        max_context_length=max_context_length,
    )
    if not parsed:
        console.print(
            "[yellow]No traces matched the input "
            "(empty directory or all dropped by --max-context-length).[/yellow]"
        )
        raise SystemExit(1)

    resolved_block_size = (
        block_size
        if block_size is not None
        else infer_weka_block_size(path, max_context_length=max_context_length)
    )

    basename = path.stem if path.is_file() else path.name
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = output / f"weka-report_{basename}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)

    _render_all(
        parsed=parsed,
        run_dir=run_dir,
        block_size=resolved_block_size,
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
        console=console,
    )


def _render_all(
    *,
    parsed: dict[str, list[ParsedTurn]],
    run_dir: Path,
    block_size: int,
    prefill_tps: float,
    decode_tps: float,
    console: Console,
) -> None:
    """Render all four report artifacts and print summary to console."""
    from aiperf.dataset.agentic_code_gen.reporting.cache_explorer import (
        render_cache_explorer,
        write_cache_structure,
    )
    from aiperf.dataset.agentic_code_gen.reporting.metrics import (
        build_report_data,
        extract_cache_metrics,
        extract_metrics,
    )
    from aiperf.dataset.agentic_code_gen.reporting.plot_report import (
        render_plot_report,
    )
    from aiperf.dataset.agentic_code_gen.reporting.report import (
        _print_report_to_console,
    )
    from aiperf.dataset.agentic_code_gen.reporting.simulation import (
        render_simulation,
    )
    from aiperf.dataset.agentic_code_gen.reporting.weka_input import (
        parsed_to_sim_sessions,
    )

    metrics = extract_metrics(
        parsed,
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
        input_lengths_are_cumulative=True,
    )
    metrics.update(
        extract_cache_metrics(parsed, block_size=block_size, hash_scope="local")
    )
    report_data = build_report_data(metrics, manifest=None)

    render_plot_report(metrics, parsed, run_dir)
    cache_payload = write_cache_structure(
        parsed, manifest=None, output_dir=run_dir, block_size_override=block_size
    )
    render_cache_explorer(run_dir, cache_payload)

    sim_sessions = parsed_to_sim_sessions(parsed)
    render_simulation(sim_sessions, run_dir / "simulation.html", block_size=block_size)

    _print_report_to_console(report_data)
    console.print(f"[green]Run directory: {run_dir}[/green]")
    console.print(f"  Report:          {run_dir / 'report.html'}")
    console.print(f"  Cache explorer:  {run_dir / 'cache_explorer.html'}")
    console.print(f"  Simulation:      {run_dir / 'simulation.html'}")
