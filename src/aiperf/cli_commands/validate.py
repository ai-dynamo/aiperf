# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for validating benchmark artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App, Parameter

app = App(name="validate")


@app.default
def validate(
    target: Annotated[
        Literal["mooncake-trace"],
        Parameter(help="Artifact format to validate"),
    ],
    input: Path,
) -> None:
    """Validate a benchmark artifact.

    Args:
        target: Artifact format to validate.
        input: Path to the artifact file.
    """
    match target:
        case "mooncake-trace":
            from aiperf.dataset.agentic_code_gen.cli import validate as _validate

            _validate(input=input)
