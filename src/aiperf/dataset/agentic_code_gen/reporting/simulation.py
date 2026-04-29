# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simulation dashboard generation for synthesized Agentic Code datasets."""

from __future__ import annotations

import string
from pathlib import Path

from rich.console import Console

from aiperf.dataset.agentic_code_gen.prefix_model import MAX_GROUP_BLOCKS, MAX_GROUPS
from aiperf.dataset.agentic_code_gen.reporting.templates import (
    read_template,
    script_safe_json,
)
from aiperf.dataset.agentic_code_gen.reporting.trace import load_simulation_sessions


def load_sessions(jsonl_path: Path) -> list[dict]:
    """Parse JSONL into grouped sessions with cumulative turn inputs."""
    return load_simulation_sessions(jsonl_path)


def render_simulation(
    sessions: list[dict],
    output_path: Path,
    *,
    block_size: int = 512,
    l1_tokens: int = 32000,
    l1_5_tokens: int = 20000,
) -> None:
    """Inline session data and config into HTML template and write file."""
    console = Console()
    sessions_json = script_safe_json(sessions)
    html = string.Template(read_template("simulation.html")).safe_substitute(
        SESSIONS_JSON=sessions_json,
        BLOCK_SIZE=str(block_size),
        L1_TOKENS=str(l1_tokens),
        L1_5_TOKENS=str(l1_5_tokens),
        MAX_GROUPS=str(MAX_GROUPS),
        MAX_GROUP_BLOCKS=str(MAX_GROUP_BLOCKS),
    )
    output_path.write_text(html, encoding="utf-8")
    console.print(f"[green]Wrote {output_path} ({len(sessions)} sessions)[/green]")
