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
        aiperf synthesize agentic-code --num-sessions 1000 --output .test/
        aiperf synthesize agentic-code --config custom.json --num-sessions 500
        aiperf synthesize agentic-code --config .test/prev_run/manifest.json --num-sessions 1000
        aiperf synthesize agentic-code --max-isl 262144 --num-sessions 1000
        aiperf synthesize agentic-code --max-osl 10000 --num-sessions 1000

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

    dist_config = _apply_cli_overrides(dist_config, max_isl=max_isl, max_osl=max_osl)

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir_name = f"{config_name}_{num_sessions}s_seed{seed}_{timestamp}"
    run_dir = Path(output) / run_dir_name

    synth = SessionSynthesizer(dist_config, seed=seed)
    console.print(f"Generating {num_sessions} sessions (seed={seed})...")
    sessions = synth.synthesize_sessions(num_sessions)

    jsonl_path, manifest_path, quality_path = write_dataset(
        sessions, run_dir, dist_config, seed=seed, config_name=config_name
    )
    validated_rows = _validate_mooncake_or_exit(jsonl_path, console)

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
    console.print(f"  Validation:      Mooncake trace ({validated_rows} rows)")
    console.print(f"  Dashboard:       {run_dir / 'report.html'}")
    console.print(f"  Cache explorer:  {run_dir / 'cache_explorer.html'}")
    console.print(f"  Simulation:      {sim_path}")
    console.print()

    comparison_path = run_dir / "comparison.txt"
    if comparison_path.exists():
        console.print(comparison_path.read_text())

    console.print(f"[dim]View: open {run_dir / 'report.html'} in a browser[/dim]")


def validate(
    input_path: Path,
) -> None:
    """Validate a generated JSONL dataset for Mooncake compatibility.

    Examples:
        aiperf validate mooncake-trace --input dataset.jsonl

    Args:
        input_path: Path to JSONL dataset file.
    """
    console = Console()
    if not input_path.is_file():
        console.print(f"[red]Validation failed: {input_path} is not a file.[/red]")
        raise SystemExit(1)
    line_count = _validate_mooncake_or_exit(input_path, console)
    console.print(
        f"[green]Validation passed: {line_count} rows are Mooncake-compatible.[/green]"
    )


def _apply_cli_overrides(
    dist_config: SessionDistributionConfig,
    max_isl: int | None,
    max_osl: int | None,
) -> SessionDistributionConfig:
    """Apply CLI overrides through normal model validation."""
    if max_isl is None and max_osl is None:
        return dist_config

    data = dist_config.model_dump()
    if max_isl is not None:
        data["max_prompt_tokens"] = max_isl
    if max_osl is not None:
        generation_length = dict(data["generation_length"])
        generation_length["max"] = float(max_osl)
        data["generation_length"] = generation_length
    return dist_config.__class__.model_validate(data)


def _validate_mooncake_or_exit(input_path: Path, console: Console) -> int:
    line_count, errors = validate_mooncake_trace(input_path)

    if errors:
        console.print(f"[red]Validation failed with {len(errors)} error(s):[/red]")
        for err in errors:
            console.print(f"  {err}")
        raise SystemExit(1)
    return line_count
