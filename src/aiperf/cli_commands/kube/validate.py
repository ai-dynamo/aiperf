# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kube validate command: validate AIPerfJob YAML files against the CRD schema."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App, Parameter

app = App(name="validate")


@app.default
async def validate(
    files: Annotated[
        list[Path],
        Parameter(
            negative=(), help="One or more AIPerfJob YAML file paths to validate."
        ),
    ],
    *,
    strict: Annotated[
        bool,
        Parameter(
            name=["-s", "--strict"],
            help="Fail on warnings such as unknown spec fields.",
        ),
    ] = False,
    output: Annotated[
        Literal["text", "json"],
        Parameter(
            name=["-o", "--output"],
            help='Output format: "text" for human-readable, "json" for machine-parseable.',
        ),
    ] = "text",
) -> None:
    """Validate AIPerfJob YAML files against the CRD schema and AIPerfConfig model.

    Performs comprehensive validation including:
    - YAML parsing and structure verification
    - Required fields: apiVersion, kind, metadata.name, spec (with endpoint)
    - Kubernetes resource name validation (RFC 1123)
    - AIPerfConfig model validation via AIPerfJobSpecConverter
    - DeploymentConfig extraction validation
    - Worker count calculation (>= 1)
    - Unknown spec field detection (warning or error with --strict)

    Examples:
        aiperf kube validate aiperfjob.yaml
        aiperf kube validate recipes/llama/*.yaml recipes/qwen/*.yaml
        aiperf kube validate --strict aiperfjob.yaml
        aiperf kube validate -o json aiperfjob.yaml
    """
    from aiperf import cli_utils

    with cli_utils.exit_on_error(title="Error Running Validation"):
        from aiperf.kubernetes import validate as kube_validate

        if output == "json":
            import orjson

            from aiperf.kubernetes.console import console

            results = []
            any_failed = False
            for path in files:
                result = kube_validate.validate_file(path, strict=strict)
                if not result.passed:
                    any_failed = True
                results.append(
                    {
                        "path": str(result.path),
                        "passed": result.passed,
                        "errors": result.errors,
                        "warnings": result.warnings,
                    }
                )

            json_output = orjson.dumps(results, option=orjson.OPT_INDENT_2).decode()
            console.print(json_output, highlight=False)

            if any_failed:
                raise SystemExit(1)
        else:
            passed, failed = await kube_validate.validate_files(files, strict=strict)

            if failed:
                raise SystemExit(1)
