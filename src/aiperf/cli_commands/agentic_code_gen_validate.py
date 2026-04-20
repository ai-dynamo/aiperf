# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for validating Agentic Code datasets."""

from __future__ import annotations

from pathlib import Path

from cyclopts import App

app = App(name="agentic-code-gen-validate")


@app.default
def validate(input: Path) -> None:
    """Validate a generated Agentic Code JSONL dataset.

    Args:
        input: Path to JSONL dataset file.
    """
    from aiperf.dataset.agentic_code_gen.cli import validate as _validate

    _validate(input=input)
