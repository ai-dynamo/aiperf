# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube show command: render AIPerfJob CR with Jinja2/env-vars resolved."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import yaml
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
    from aiperf.cli_utils import exit_on_error
    from aiperf.config import dump_config
    from aiperf.operator.spec_converter import extract_benchmark_config

    with exit_on_error(title="Error Rendering AIPerfJob"):
        doc = yaml.safe_load(path.read_text())

        if not isinstance(doc, dict):
            raise ValueError(f"{path}: document is not a YAML mapping")
        if doc.get("kind") != "AIPerfJob":
            raise ValueError(
                f"{path}: not an AIPerfJob manifest (kind={doc.get('kind')!r})"
            )
        spec = doc.get("spec")
        if not isinstance(spec, dict) or not isinstance(spec.get("benchmark"), dict):
            raise ValueError(
                f"{path}: spec.benchmark is required and must be a mapping"
            )

        # Render + validate the benchmark section. extract_benchmark_config
        # runs expand_config_dict (env vars + Jinja2) then AIPerfConfig
        # validation, and deliberately skips K8s runtime injection.
        config = extract_benchmark_config(spec)
        rendered_benchmark = yaml.safe_load(dump_config(config))

        doc["spec"]["benchmark"] = rendered_benchmark
        print(
            yaml.safe_dump(doc, sort_keys=False, default_flow_style=False),
            end="",
        )
