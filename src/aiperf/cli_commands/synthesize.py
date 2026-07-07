# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for synthesizing datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App, Parameter
from cyclopts.validators import Number

app = App(name="synthesize")

# Reject non-positive counts/lengths at the CLI boundary so users get a clean
# cyclopts error ("Must be >= 1") instead of a deep pydantic traceback surfaced
# from inside the generator.
_positive_int = Number(gte=1)


@app.default
def synthesize(
    target: Annotated[
        Literal["agentic-code"],
        Parameter(help="Dataset workload to synthesize"),
    ],
    *,
    num_sessions: Annotated[int, Parameter(validator=_positive_int)] = 1000,
    output: Path = Path("."),
    config: str | None = None,
    seed: int = 42,
    max_isl: Annotated[int | None, Parameter(validator=_positive_int)] = None,
    max_osl: Annotated[int | None, Parameter(validator=_positive_int)] = None,
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
    from aiperf.cli_utils import exit_on_error

    match target:
        case "agentic-code":
            from aiperf.dataset.agentic_code_gen.cli import synthesize as _synthesize

            # Broad boundary (mirrors ``aiperf profile``): a malformed --config
            # surfaces orjson.JSONDecodeError / pydantic.ValidationError and a
            # missing config surfaces FileNotFoundError; all render as a clean
            # panel + exit 1 instead of a raw traceback.
            with exit_on_error(
                title="Error Synthesizing Dataset",
                show_traceback=False,
            ):
                _synthesize(
                    num_sessions=num_sessions,
                    output=output,
                    config=config,
                    seed=seed,
                    max_isl=max_isl,
                    max_osl=max_osl,
                )
