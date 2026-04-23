# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI commands for AIPerf configuration (YAML-based)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from aiperf.cli_utils import exit_on_error
from aiperf.config.cli_model import CLIModel

config_app = App(name="config", help="Manage AIPerf YAML configurations")


def _print_template_table(
    templates: list,
    *,
    verbose: bool = False,
) -> None:
    """Print templates as a Rich table grouped by category."""
    from rich.console import Console
    from rich.table import Table

    from aiperf.config.templates import CATEGORY_ORDER

    console = Console()
    by_category: dict[str, list] = {}
    for t in templates:
        by_category.setdefault(t.category, []).append(t)

    for cat in CATEGORY_ORDER:
        group = by_category.pop(cat, None)
        if not group:
            continue

        table = Table(
            title=cat,
            title_style="bold",
            show_header=True,
            header_style="dim",
            box=None,
            pad_edge=False,
        )
        table.add_column("Name", style="cyan", min_width=25)
        table.add_column("Title")
        table.add_column("Description", style="dim")
        if verbose:
            table.add_column("Tags", style="dim")
            table.add_column("Difficulty", style="dim")

        for t in group:
            row: list[str] = [t.name, t.title, t.description]
            if verbose:
                row.append(", ".join(t.tags) if t.tags else "")
                row.append(t.difficulty)
            table.add_row(*row)

        console.print(table)
        console.print()


@config_app.command(name="init")
def init_config(
    *,
    template: Annotated[
        str | None,
        Parameter(
            name=["-t", "--template"],
            help="Template name to use (e.g. 'minimal', 'goodput_slo'). "
            "Run with --list to see all available templates.",
        ),
    ] = None,
    list_templates: Annotated[
        bool,
        Parameter(
            name=["-l", "--list"],
            help="List all available templates grouped by category.",
        ),
    ] = False,
    search: Annotated[
        str | None,
        Parameter(
            name=["-s", "--search"],
            help="Search templates by keyword (matches name, description, tags, features).",
        ),
    ] = None,
    category: Annotated[
        str | None,
        Parameter(
            name=["-c", "--category"],
            help="Filter templates by category (substring match).",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        Parameter(
            name=["-v", "--verbose"],
            help="Show tags, features, and difficulty in template listings.",
        ),
    ] = False,
    model: Annotated[
        str | None,
        Parameter(
            name=["--model"],
            help="Override model name in the generated config.",
        ),
    ] = None,
    url: Annotated[
        str | None,
        Parameter(
            name=["--url"],
            help="Override endpoint URL in the generated config.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        Parameter(
            name=["-o", "--output"],
            help="Output file path. If not specified, prints to stdout.",
        ),
    ] = None,
) -> None:
    """Generate a starter configuration from bundled templates.

    Without arguments, generates the 'minimal' template. Use --list to browse
    all 19 bundled templates organized by category, or --search to find
    templates by keyword.

    Use --model and --url to pre-fill the two fields every config needs,
    so the generated file is ready to run without editing.

    Examples:
        # List all available templates
        aiperf config init --list

        # List with tags, features, and difficulty
        aiperf config init --list --verbose

        # Search for sweep-related templates
        aiperf config init --search sweep

        # Filter by category
        aiperf config init --list --category "Load Testing"

        # Generate the minimal starter config
        aiperf config init

        # Generate a specific template
        aiperf config init --template goodput_slo

        # Generate with your model and endpoint pre-filled
        aiperf config init --template latency_test \\
            --model meta-llama/Llama-3.1-70B-Instruct \\
            --url http://my-server:8000/v1/chat/completions

        # Save to a file
        aiperf config init --template sweep_distributions --output benchmark.yaml

        # Pipe to a file
        aiperf config init --template latency_test > my_benchmark.yaml
    """
    from aiperf.config.templates import (
        apply_overrides,
        get_template,
        load_template_content,
        search_templates,
        strip_spdx_header,
    )
    from aiperf.config.templates import (
        list_templates as _list_templates,
    )

    with exit_on_error(title="Template Error"):
        # Mode 1: Search
        if search:
            results = search_templates(search)
            if not results:
                print(f"No templates match '{search}'.")
                print("Run 'aiperf config init --list' to see all templates.")
                return
            _print_template_table(results, verbose=verbose)
            return

        # Mode 2: List (optionally filtered by category)
        if list_templates:
            results = _list_templates(category=category)
            if not results:
                print(f"No templates in category '{category}'.")
                return
            _print_template_table(results, verbose=verbose)
            print("Use 'aiperf config init --template <name>' to generate a template.")
            return

        # Mode 3: Generate template
        name = template or "minimal"
        try:
            info = get_template(name)
        except KeyError as e:
            print(str(e))
            raise SystemExit(1) from None

        content = load_template_content(name)

        # Build overrides dict, matching singular/plural form used in template
        import yaml as _yaml

        overrides: dict = {}
        if model or url:
            raw = _yaml.safe_load(content) or {}
            if model:
                if "model" in raw:
                    overrides["model"] = model
                else:
                    overrides["models"] = [model]
            if url:
                ep = raw.get("endpoint", {})
                if "url" in ep:
                    overrides.setdefault("endpoint", {})["url"] = url
                else:
                    overrides.setdefault("endpoint", {})["urls"] = [url]

        content = strip_spdx_header(content)
        if overrides:
            content = apply_overrides(content, overrides)

        if output is None:
            print(content, end="")
            return

        if output.exists():
            response = input(f"File '{output}' already exists. Overwrite? [y/N] ")
            if response.lower() not in ("y", "yes"):
                print("Aborted.")
                return

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content)
        print(f"Created {output} from template '{info.name}' ({info.title})")
        print("\nNext steps:")
        print(f"  1. Edit {output} -- update endpoint URLs and model name")
        print(f"  2. Run:  aiperf profile --config {output}")
        print(f"  3. Or:   aiperf kube profile --config {output} --image <your-image>")


@config_app.command(name="validate")
def validate_config(
    *,
    path: Annotated[Path, Parameter(help="Path to the YAML configuration file.")],
    interpolate: Annotated[
        bool, Parameter(help="Whether to interpolate environment variables.")
    ] = True,
    strict: Annotated[
        bool, Parameter(help="Fail on warnings (e.g., unused fields).")
    ] = False,
) -> None:
    """Validate a YAML configuration file.

    Checks that the configuration is valid YAML, conforms to the AIPerfConfig
    schema, and has no missing required fields.

    Examples:
        # Validate a config file
        aiperf config validate benchmark.yaml

        # Validate without environment variable interpolation
        aiperf config validate benchmark.yaml --no-interpolate

        # Strict validation (fail on warnings)
        aiperf config validate benchmark.yaml --strict
    """
    with exit_on_error(title="Configuration Validation Error"):
        from aiperf.config import load_config, validate_config_file

        # Validate using the config module
        errors = validate_config_file(path)

        if errors:
            print(f"❌ Configuration invalid: {path}")
            for error in errors:
                print(f"  • {error}")
            raise SystemExit(1)

        # Also try to load the config to catch any additional issues
        config = load_config(path, substitute_env=interpolate)

        print(f"✓ Configuration valid: {path}")
        print(f"  Models: {config.get_model_names()}")
        print(f"  Datasets: {list(config.datasets.keys())}")
        print(f"  Phases: {[p.name for p in config.phases.values()]}")


@config_app.command(name="show")
def show_config(
    *,
    path: Annotated[Path, Parameter(help="Path to the YAML configuration file.")],
    format: Annotated[str, Parameter(help="Output format: 'yaml' or 'json'.")] = "yaml",
    interpolate: Annotated[
        bool, Parameter(help="Whether to interpolate environment variables.")
    ] = True,
) -> None:
    """Display a configuration file with all defaults expanded.

    Loads the configuration, applies all defaults, and outputs the complete
    configuration in the specified format.

    Examples:
        # Show config with defaults as YAML
        aiperf config show benchmark.yaml

        # Show config as JSON
        aiperf config show benchmark.yaml --format json

        # Show config without environment interpolation
        aiperf config show benchmark.yaml --no-interpolate
    """
    with exit_on_error(title="Error Loading Configuration"):
        import orjson

        from aiperf.config import dump_config, load_config

        config = load_config(path, substitute_env=interpolate)

        if format == "json":
            output = orjson.dumps(
                config.model_dump(mode="json"), option=orjson.OPT_INDENT_2
            ).decode()
        else:
            output = dump_config(config)
        print(output)


@config_app.command(name="schema")
def show_schema(
    *,
    output: Annotated[
        Path | None,
        Parameter(
            help="Path to write the schema file. If not provided, prints to stdout."
        ),
    ] = None,
) -> None:
    """Output the JSON schema for AIPerfConfig.

    Generates a JSON Schema document that describes the complete AIPerfConfig
    structure. Useful for IDE integration and validation tooling.

    Examples:
        # Print schema to stdout
        aiperf config schema

        # Save schema to file
        aiperf config schema --output aiperf-schema.json
    """
    with exit_on_error(title="Error Generating Schema"):
        import orjson

        from aiperf.config import AIPerfConfig

        schema = AIPerfConfig.model_json_schema()
        schema_json = orjson.dumps(schema, option=orjson.OPT_INDENT_2).decode()

        if output:
            output.write_text(schema_json)
            print(f"✓ Schema written to: {output}")
        else:
            print(schema_json)


@config_app.command(name="diff")
def diff_configs(
    *,
    config1: Annotated[Path, Parameter(help="Path to first YAML configuration file.")],
    config2: Annotated[Path, Parameter(help="Path to second YAML configuration file.")],
    format: Annotated[str, Parameter(help="Output format: 'text' or 'json'.")] = "text",
) -> None:
    """Compare two configuration files and show differences.

    Loads both configurations, normalizes them with defaults, and shows
    the differences between them. Useful for understanding how configs
    differ or verifying changes.

    Examples:
        # Compare two configs (text output)
        aiperf config diff baseline.yaml experiment.yaml

        # Compare with JSON output
        aiperf config diff baseline.yaml experiment.yaml --format json
    """
    import orjson

    from aiperf.config import load_config

    with exit_on_error(title="Error Comparing Configurations"):
        # Load both configs
        cfg1 = load_config(config1)
        cfg2 = load_config(config2)

        # Convert to dicts for comparison
        dict1 = cfg1.model_dump(mode="json")
        dict2 = cfg2.model_dump(mode="json")

        # Find differences
        differences = _find_differences(dict1, dict2)

        if not differences:
            print("✓ Configurations are identical")
            return

        if format == "json":
            print(orjson.dumps(differences, option=orjson.OPT_INDENT_2).decode())
        else:
            print(f"Comparing: {config1} vs {config2}")
            print(f"Found {len(differences)} difference(s):\n")
            for diff in differences:
                _print_diff_entry(diff, config1.name, config2.name)
                print()


def _print_diff_entry(diff: dict, config1_name: str, config2_name: str) -> None:
    """Print a single diff entry in text format."""
    path = diff["path"]
    if diff["type"] == "changed":
        print(f"  {path}:")
        print(f"    - {config1_name}: {diff['old']}")
        print(f"    + {config2_name}: {diff['new']}")
    elif diff["type"] == "added":
        print(f"  + {path}: {diff['value']} (only in {config2_name})")
    elif diff["type"] == "removed":
        print(f"  - {path}: {diff['value']} (only in {config1_name})")


def _compare_values(val1: object, val2: object, current_path: str) -> list[dict]:
    """Compare two same-keyed values and return zero-or-more difference entries."""
    if isinstance(val1, dict) and isinstance(val2, dict):
        return _find_differences(val1, val2, current_path)
    if isinstance(val1, list) and isinstance(val2, list):
        if val1 != val2:
            return [
                {
                    "type": "changed",
                    "path": current_path,
                    "old": val1,
                    "new": val2,
                }
            ]
        return []
    if val1 != val2:
        return [
            {
                "type": "changed",
                "path": current_path,
                "old": val1,
                "new": val2,
            }
        ]
    return []


def _find_differences(dict1: dict, dict2: dict, path: str = "") -> list[dict]:
    """Recursively find differences between two dicts."""
    differences = []

    all_keys = set(dict1.keys()) | set(dict2.keys())

    for key in sorted(all_keys):
        current_path = f"{path}.{key}" if path else key

        in_dict1 = key in dict1
        in_dict2 = key in dict2

        if in_dict1 and not in_dict2:
            differences.append(
                {
                    "type": "removed",
                    "path": current_path,
                    "value": dict1[key],
                }
            )
        elif in_dict2 and not in_dict1:
            differences.append(
                {
                    "type": "added",
                    "path": current_path,
                    "value": dict2[key],
                }
            )
        else:
            differences.extend(_compare_values(dict1[key], dict2[key], current_path))

    return differences


@config_app.command(name="generate")
def generate_config(
    *,
    cli_model: CLIModel,
    output: Annotated[
        Path | None,
        Parameter(help="Path to write the config. If not provided, prints to stdout."),
    ] = None,
    format: Annotated[str, Parameter(help="Output format: 'yaml' or 'json'.")] = "yaml",
) -> None:
    """Generate YAML configuration from CLI options.

    Takes the same CLI flags as 'aiperf profile' and outputs the equivalent
    YAML configuration. Useful for migrating from CLI-based to YAML-based
    configuration.

    Examples:
        # Generate YAML config from CLI options (prints to stdout)
        aiperf config generate --model llama-3.1-8B --url localhost:8000 \\
            --request-rate 10 --request-count 1000

        # Save to file
        aiperf config generate --model llama-3.1-8B --url localhost:8000 \\
            --concurrency 32 --request-count 1000 --output benchmark.yaml

        # Generate as JSON
        aiperf config generate --model llama-3.1-8B --url localhost:8000 \\
            --format json
    """
    import orjson

    from aiperf.config.cli_converter import build_aiperf_config
    from aiperf.config.loader import dump_config

    with exit_on_error(title="Error Generating Configuration"):
        aiperf_config = build_aiperf_config(cli_model)

        if format == "json":
            config_output = orjson.dumps(
                aiperf_config.model_dump(mode="json"), option=orjson.OPT_INDENT_2
            ).decode()
        else:
            config_output = dump_config(aiperf_config)

        if output:
            output.write_text(config_output)
            print(f"✓ Configuration written to: {output}")
        else:
            print(config_output)
