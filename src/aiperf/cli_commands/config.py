# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for AIPerf configuration scaffolding.

aiperf config init --list                                # List bundled templates
aiperf config init --search sweep                        # Filter by keyword
aiperf config init --template goodput_slo                # Print a template to stdout
aiperf config init --template latency_test \\
    --model meta-llama/Llama-3.1-70B-Instruct \\
    --url http://localhost:8000/v1/chat/completions \\
    --output benchmark.yaml                              # Customize and save

aiperf config expand sweep.yaml                          # List sweep variations
aiperf config expand sweep.yaml --full                   # Dump every variation's body
aiperf config expand sweep.yaml --index 2 --full         # Inspect a single variation
aiperf config expand sweep.yaml --format json            # Machine-readable output

aiperf config validate benchmark.yaml                    # Lint a config and surface warnings
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App, Parameter

app = App(name="config", help="Configuration scaffolding commands.")


@app.command(name="init")
def init(
    *,
    template: Annotated[
        str | None,
        Parameter(
            name=["--template", "-t"],
            help="Template name to generate (e.g. 'minimal', 'goodput_slo').",
        ),
    ] = None,
    list_templates: Annotated[
        bool,
        Parameter(name=["--list", "-l"], help="List bundled templates and exit."),
    ] = False,
    search: Annotated[
        str | None,
        Parameter(name=["--search", "-s"], help="Filter templates by keyword."),
    ] = None,
    category: Annotated[
        str | None,
        Parameter(name=["--category", "-c"], help="Filter --list by category."),
    ] = None,
    verbose: Annotated[
        bool,
        Parameter(name=["--verbose", "-v"], help="Show tags and difficulty columns."),
    ] = False,
    model: Annotated[
        str | None,
        Parameter(name=["--model", "-m"], help="Override the template's model name."),
    ] = None,
    url: Annotated[
        str | None,
        Parameter(name=["--url", "-u"], help="Override the template's endpoint URL."),
    ] = None,
    output: Annotated[
        Path | None,
        Parameter(name=["--output", "-o"], help="Write to file instead of stdout."),
    ] = None,
) -> None:
    """Generate, list, or search bundled AIPerf config templates.

    Without ``--output``, selected template YAML is printed to stdout. With
    ``--output``, the customized template is written to that path after applying
    ``--model`` and ``--url`` overrides.
    """
    from aiperf.config.cli_runner import run_init

    run_init(
        template=template,
        list_templates=list_templates,
        search=search,
        category=category,
        verbose=verbose,
        model=model,
        url=url,
        output=output,
    )


@app.command(name="expand")
def expand(
    config_file: Annotated[
        Path,
        Parameter(help="Path to an AIPerf YAML config containing a `sweep:` block."),
    ],
    *,
    full: Annotated[
        bool,
        Parameter(
            name=["--full", "-F"],
            help="Also emit each variation's fully-merged BenchmarkConfig body.",
        ),
    ] = False,
    index: Annotated[
        int | None,
        Parameter(
            name=["--index", "-i"],
            help="Show only the variation at this zero-based index (implies --full).",
        ),
    ] = None,
    fmt: Annotated[
        Literal["text", "yaml", "json"],
        Parameter(
            name=["--format", "-f"],
            help="Output format: text (default human-readable), yaml, or json.",
        ),
    ] = "text",
) -> None:
    """Expand a sweep config and print the resulting variations.

    Drives the same `load_config` -> `build_benchmark_plan` pipeline that
    `aiperf profile` uses, then prints what the orchestrator would have
    iterated over - without launching any benchmarks. Useful for verifying
    sweep paths, dir_name conventions, and per-variation merges before
    spending compute.
    """
    from aiperf.config.cli_runner import run_expand

    run_expand(config_file=config_file, full=full, index=index, fmt=fmt)


@app.command(name="validate")
def validate(
    config_file: Annotated[
        Path,
        Parameter(help="Path to an AIPerf YAML config to validate."),
    ],
) -> None:
    """Validate an AIPerf config file.

    Loads the config through the same pipeline as `aiperf profile`, surfacing
    fatal errors (exit 1) and non-fatal warnings (printed to stderr; exit 0).
    Useful as a pre-flight check or in CI before kicking off a benchmark.
    """
    from aiperf.config.cli_runner import run_validate

    run_validate(config_file=config_file)
