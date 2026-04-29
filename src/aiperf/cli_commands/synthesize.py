# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for synthesizing datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App, Parameter

app = App(name="synthesize")


@app.default
def synthesize(
    target: Annotated[
        Literal["agentic-code"],
        Parameter(help="Dataset workload to synthesize"),
    ],
    *,
    num_sessions: int = 1000,
    output: Path = Path("."),
    config: str | None = None,
    seed: int = 42,
    max_isl: int | None = None,
    max_osl: int | None = None,
) -> None:
    """Synthesize a dataset workload.

    Args:
        target: Dataset workload to synthesize.
        num_sessions: Number of sessions to generate.
        output: Parent directory for the run directory.
        config: Path to config/manifest JSON.
        seed: Random seed for reproducibility.
        max_isl: Maximum input sequence length.
        max_osl: Maximum output sequence length.
    """
    match target:
        case "agentic-code":
            from aiperf.dataset.agentic_code_gen.cli import synthesize as _synthesize

            _synthesize(
                num_sessions=num_sessions,
                output=output,
                config=config,
                seed=seed,
                max_isl=max_isl,
                max_osl=max_osl,
            )
