# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube show command: render AIPerfJob CR with Jinja2/env-vars resolved."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

app = App(name="show")


@app.default
def show(
    *,
    path: Annotated[
        Path,
        Parameter(
            name=["-p", "--path"],
            help="Path to an AIPerfJob YAML file.",
        ),
    ],
) -> None:
    """Render an AIPerfJob CR with Jinja2 and env-var templates resolved.

    Reads the CR, expands ``{{ ... }}`` expressions and ``${ENV_VAR}``
    substitutions inside ``spec.benchmark``, validates the result against
    ``AIPerfConfig``, re-wraps it in the original ``metadata`` and
    non-benchmark ``spec.*`` fields, and prints YAML to stdout.

    Examples:
        aiperf kube show --path recipes/qwen3-32b-fp8/trtllm/agg/perf.yaml
    """
    raise NotImplementedError("show command not yet implemented")
