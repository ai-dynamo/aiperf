# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for synthesizing Agentic Code datasets."""

from __future__ import annotations

from pathlib import Path

from cyclopts import App

app = App(name="agentic-code-gen-synthesize")


@app.default
def synthesize(
    num_sessions: int = 1000,
    output: Path = Path("."),
    config: str | None = None,
    seed: int = 42,
    max_isl: int | None = None,
    max_osl: int | None = None,
) -> None:
    """Synthesize multi-turn Agentic Code session dataset into a unique run directory.

    Args:
        num_sessions: Number of sessions to generate.
        output: Parent directory for the run directory.
        config: Path to config/manifest JSON.
        seed: Random seed for reproducibility.
        max_isl: Maximum input sequence length.
        max_osl: Maximum output sequence length.
    """
    from aiperf.dataset.agentic_code_gen.cli import synthesize as _synthesize

    _synthesize(
        num_sessions=num_sessions,
        output=output,
        config=config,
        seed=seed,
        max_isl=max_isl,
        max_osl=max_osl,
    )
