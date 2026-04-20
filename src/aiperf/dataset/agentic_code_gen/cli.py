# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI subcommands for Agentic Code dataset generation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from aiperf.dataset.agentic_code_gen.config import load_config
from aiperf.dataset.agentic_code_gen.models import SessionDistributionConfig
from aiperf.dataset.agentic_code_gen.reporting.simulation import (
    load_sessions,
    render_simulation,
)
from aiperf.dataset.agentic_code_gen.reporting.trace import validate_mooncake_trace
from aiperf.dataset.agentic_code_gen.session_synthesizer import SessionSynthesizer
from aiperf.dataset.agentic_code_gen.writer import write_dataset


def synthesize(
    num_sessions: int = 1000,
    output: Path = Path("."),
    config: str | None = None,
    seed: int = 42,
    max_isl: int | None = None,
    max_osl: int | None = None,
) -> None:
    """Synthesize multi-turn session dataset into a unique run directory.

    --config accepts a path to a config JSON or a manifest.json from a previous run.
    If omitted, built-in defaults are used.

    Examples:
        aiperf agentic-code-gen-synthesize --num-sessions 1000 --output .test/
        aiperf agentic-code-gen-synthesize --config custom.json --num-sessions 500
        aiperf agentic-code-gen-synthesize --config .test/prev_run/manifest.json --num-sessions 1000
        aiperf agentic-code-gen-synthesize --max-isl 262144 --num-sessions 1000
        aiperf agentic-code-gen-synthesize --max-osl 10000 --num-sessions 1000

    Args:
        num_sessions: Number of sessions to generate.
        output: Parent directory for the run directory (default: current dir).
        config: Path to config/manifest JSON (default: built-in defaults).
        seed: Random seed for reproducibility.
        max_isl: Maximum input sequence length — overrides max_prompt_tokens to clip context.
        max_osl: Maximum output sequence length — overrides generation_length.max.
    """
    console = Console()

    if config:
        dist_config = load_config(config)
        config_name = Path(config).stem if Path(config).is_file() else config
    else:
        dist_config = SessionDistributionConfig()
        config_name = "default"

    if max_isl is not None:
        dist_config = dist_config.model_copy(update={"max_prompt_tokens": max_isl})

    if max_osl is not None:
        gen = dist_config.generation_length.model_copy(update={"max": float(max_osl)})
        dist_config = dist_config.model_copy(update={"generation_length": gen})

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir_name = f"{config_name}_{num_sessions}s_seed{seed}_{timestamp}"
    run_dir = Path(output) / run_dir_name

    synth = SessionSynthesizer(dist_config, seed=seed)
    console.print(f"Generating {num_sessions} sessions (seed={seed})...")
    sessions = synth.synthesize_sessions(num_sessions)

    jsonl_path, manifest_path, quality_path = write_dataset(
        sessions, run_dir, dist_config, seed=seed, config_name=config_name
    )

    sim_sessions = load_sessions(jsonl_path)
    sim_path = run_dir / "simulation.html"
    render_simulation(
        sim_sessions,
        sim_path,
        block_size=dist_config.block_size,
        l1_tokens=dist_config.cache.layer1_tokens,
        l1_5_tokens=dist_config.cache.layer1_5_tokens,
    )

    total_turns = sum(len(s.turns) for s in sessions)
    console.print(f"[green]Run directory: {run_dir}[/green]")
    console.print(f"  JSONL:           {jsonl_path} ({total_turns} turns)")
    console.print(f"  Manifest:        {manifest_path}")
    console.print(f"  Quality:         {quality_path}")
    console.print(f"  Dashboard:       {run_dir / 'report.html'}")
    console.print(f"  Cache explorer:  {run_dir / 'cache_explorer.html'}")
    console.print(f"  Simulation:      {sim_path}")
    console.print()

    comparison_path = run_dir / "comparison.txt"
    if comparison_path.exists():
        console.print(comparison_path.read_text())

    console.print(f"[dim]View: open {run_dir / 'report.html'} in a browser[/dim]")


def validate(
    input: Path,
) -> None:
    """Validate a generated JSONL dataset for Mooncake compatibility.

    Examples:
        aiperf agentic-code-gen-validate --input dataset.jsonl

    Args:
        input: Path to JSONL dataset file.
    """
    console = Console()
    line_count, errors = validate_mooncake_trace(input)

    if errors:
        console.print(f"[red]Validation failed with {len(errors)} error(s):[/red]")
        for err in errors:
            console.print(f"  {err}")
        raise SystemExit(1)
    else:
        console.print(
            f"[green]Validation passed: {line_count} rows are Mooncake-compatible.[/green]"
        )
