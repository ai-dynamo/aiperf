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

aiperf config show benchmark.yaml                        # Render with defaults expanded
aiperf config show benchmark.yaml --format json
aiperf config schema --output aiperf-schema.json         # Emit JSON Schema for IDEs
aiperf config diff baseline.yaml experiment.yaml         # Compare two configs
aiperf config generate --model llama --url localhost:8000  # CLI flags -> YAML
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App, Parameter

from aiperf.config.flags import CLIConfig

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


@app.command(name="show")
def show(
    config_file: Annotated[
        Path,
        Parameter(help="Path to the YAML configuration file."),
    ],
    *,
    fmt: Annotated[
        Literal["yaml", "json"],
        Parameter(name=["--format", "-f"], help="Output format: yaml or json."),
    ] = "yaml",
    interpolate: Annotated[
        bool,
        Parameter(help="Substitute environment variables before rendering."),
    ] = True,
) -> None:
    """Render a config with all defaults expanded.

    Loads the config through the same pipeline as `aiperf profile` and prints
    the fully-resolved YAML (or JSON) representation.
    """
    from aiperf.cli_utils import exit_on_error

    with exit_on_error(title="Error Loading Configuration"):
        import orjson

        from aiperf.config.loader import dump_config, load_config

        config = load_config(config_file, substitute_env=interpolate)

        if fmt == "json":
            print(
                orjson.dumps(
                    config.model_dump(mode="json"), option=orjson.OPT_INDENT_2
                ).decode()
            )
        else:
            print(dump_config(config))


@app.command(name="schema")
def schema(
    *,
    output: Annotated[
        Path | None,
        Parameter(
            name=["--output", "-o"],
            help="Write the JSON Schema to a file. Prints to stdout when omitted.",
        ),
    ] = None,
) -> None:
    """Emit the JSON Schema for `AIPerfConfig`.

    Useful for IDE integration and external validation tooling.
    """
    from aiperf.cli_utils import exit_on_error

    with exit_on_error(title="Error Generating Schema"):
        import orjson

        from aiperf.config.config import AIPerfConfig

        schema_dict = AIPerfConfig.model_json_schema()
        schema_json = orjson.dumps(schema_dict, option=orjson.OPT_INDENT_2).decode()

        if output is not None:
            output.write_text(schema_json)
            print(f"Schema written to: {output}")
        else:
            print(schema_json)


@app.command(name="diff")
def diff(
    config1: Annotated[Path, Parameter(help="Path to the first YAML config.")],
    config2: Annotated[Path, Parameter(help="Path to the second YAML config.")],
    *,
    fmt: Annotated[
        Literal["text", "json"],
        Parameter(name=["--format", "-f"], help="Output format: text or json."),
    ] = "text",
) -> None:
    """Compare two configs after default-expansion and print the differences."""
    from aiperf.cli_utils import exit_on_error

    with exit_on_error(title="Error Comparing Configurations"):
        import orjson

        from aiperf.config.loader import load_config

        cfg1 = load_config(config1)
        cfg2 = load_config(config2)
        differences = _find_differences(
            cfg1.model_dump(mode="json"), cfg2.model_dump(mode="json")
        )

        if not differences:
            print("Configurations are identical")
            return

        if fmt == "json":
            print(orjson.dumps(differences, option=orjson.OPT_INDENT_2).decode())
            return

        print(f"Comparing: {config1} vs {config2}")
        print(f"Found {len(differences)} difference(s):\n")
        for entry in differences:
            _print_diff_entry(entry, config1.name, config2.name)
            print()


@app.command(name="generate")
def generate(
    *,
    cli_config: CLIConfig,
    output: Annotated[
        Path | None,
        Parameter(
            name=["--output", "-o"],
            help="Write the YAML to a file. Prints to stdout when omitted.",
        ),
    ] = None,
    fmt: Annotated[
        Literal["yaml", "json"],
        Parameter(name=["--format", "-f"], help="Output format: yaml or json."),
    ] = "yaml",
) -> None:
    """Generate a YAML config from `aiperf profile`-style CLI flags.

    Takes the same flag surface as `aiperf profile` and emits the equivalent
    YAML. Useful for migrating from CLI-driven invocations to YAML configs.
    """
    from aiperf.cli_utils import exit_on_error

    with exit_on_error(title="Error Generating Configuration"):
        import orjson

        from aiperf.config.flags.converter import convert_cli_to_aiperf
        from aiperf.config.loader import dump_config

        aiperf_config = convert_cli_to_aiperf(cli_config)

        if fmt == "json":
            rendered = orjson.dumps(
                aiperf_config.model_dump(mode="json"), option=orjson.OPT_INDENT_2
            ).decode()
        else:
            rendered = dump_config(aiperf_config)

        if output is not None:
            output.write_text(rendered)
            print(f"Configuration written to: {output}")
        else:
            print(rendered)


def _find_differences(dict1: dict, dict2: dict, path: str = "") -> list[dict]:
    """Recursively diff two normalized config dicts."""
    differences: list[dict] = []
    all_keys = set(dict1.keys()) | set(dict2.keys())

    for key in sorted(all_keys):
        current_path = f"{path}.{key}" if path else key
        in1 = key in dict1
        in2 = key in dict2

        if in1 and not in2:
            differences.append(
                {"type": "removed", "path": current_path, "value": dict1[key]}
            )
        elif in2 and not in1:
            differences.append(
                {"type": "added", "path": current_path, "value": dict2[key]}
            )
        else:
            differences.extend(_compare_values(dict1[key], dict2[key], current_path))

    return differences


def _compare_values(val1: object, val2: object, current_path: str) -> list[dict]:
    if isinstance(val1, dict) and isinstance(val2, dict):
        return _find_differences(val1, val2, current_path)
    if val1 != val2:
        return [{"type": "changed", "path": current_path, "old": val1, "new": val2}]
    return []


def _print_diff_entry(entry: dict, name1: str, name2: str) -> None:
    path = entry["path"]
    kind = entry["type"]
    if kind == "changed":
        print(f"  {path}:")
        print(f"    - {name1}: {entry['old']}")
        print(f"    + {name2}: {entry['new']}")
    elif kind == "added":
        print(f"  + {path}: {entry['value']} (only in {name2})")
    elif kind == "removed":
        print(f"  - {path}: {entry['value']} (only in {name1})")
