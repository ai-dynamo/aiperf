# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube show command: render AIPerfJob CR with Jinja2/env-vars resolved."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import yaml
from cyclopts import App, Parameter

from aiperf.common.path_safety import safe_read_template_path

app = App(name="show")


@app.default
def show(
    *,
    path: Annotated[
        Path,
        Parameter(
            name=["-f", "--path"],
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
        aiperf kube show --path benchmarks/qwen3-32b/perf.yaml
        aiperf kube show -f benchmarks/llama-3-70b/perf.yaml
    """
    from aiperf.cli_utils import exit_on_error
    from aiperf.config import dump_config
    from aiperf.kubernetes import console as kube_console
    from aiperf.kubernetes.spec_converter import extract_benchmark_config

    with exit_on_error(title="Error Rendering AIPerfJob"):
        text = safe_read_template_path(str(path))
        if text is None:
            raise ValueError(f"Cannot safely read AIPerfJob manifest: {path}")
        doc = yaml.safe_load(text)

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
        rendered_benchmark = yaml.safe_load(dump_config(config)).get("benchmark", {})

        doc["spec"]["benchmark"] = rendered_benchmark
        # width=inf prevents yaml from soft-wrapping long strings (image refs,
        # URLs) into ambiguous indentation that can confuse `kubectl apply`.
        # markup/highlight disabled and soft_wrap=True keep Rich from mangling
        # YAML when piped; end="" preserves the single trailing newline that
        # yaml.safe_dump already emits.
        output = yaml.safe_dump(
            doc, sort_keys=False, default_flow_style=False, width=float("inf")
        )
        kube_console.console.print(
            output, end="", markup=False, highlight=False, soft_wrap=True
        )
