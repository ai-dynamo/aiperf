# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`aiperf config init` command — generate starter configs from bundled templates."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from aiperf.cli_commands.config_cli import config_app
from aiperf.cli_utils import exit_on_error


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


def _handle_search(search: str, *, verbose: bool) -> None:
    from aiperf.config.templates import search_templates

    results = search_templates(search)
    if not results:
        print(f"No templates match '{search}'.")
        print("Run 'aiperf config init --list' to see all templates.")
        return
    _print_template_table(results, verbose=verbose)


def _handle_list(category: str | None, *, verbose: bool) -> None:
    from aiperf.config.templates import list_templates as _list_templates

    results = _list_templates(category=category)
    if not results:
        print(f"No templates in category '{category}'.")
        return
    _print_template_table(results, verbose=verbose)
    print("Use 'aiperf config init --template <name>' to generate a template.")


def _build_overrides(content: str, model: str | None, url: str | None) -> dict:
    """Build overrides dict, matching singular/plural form used in template."""
    import yaml as _yaml

    overrides: dict = {}
    if not (model or url):
        return overrides

    raw = _yaml.safe_load(content) or {}
    if model:
        key = "model" if "model" in raw else "models"
        overrides[key] = model if key == "model" else [model]
    if url:
        ep = raw.get("endpoint", {})
        url_key = "url" if "url" in ep else "urls"
        overrides.setdefault("endpoint", {})[url_key] = (
            url if url_key == "url" else [url]
        )
    return overrides


def _write_template_output(
    content: str, output: Path | None, info_name: str, info_title: str
) -> None:
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
    print(f"Created {output} from template '{info_name}' ({info_title})")
    print("\nNext steps:")
    print(f"  1. Edit {output} -- update endpoint URLs and model name")
    print(f"  2. Run:  aiperf profile --config {output}")
    print(f"  3. Or:   aiperf kube profile --config {output} --image <your-image>")


def _generate_template(
    template: str | None,
    model: str | None,
    url: str | None,
    output: Path | None,
) -> None:
    from aiperf.config.templates import (
        apply_overrides,
        get_template,
        load_template_content,
        strip_spdx_header,
    )

    name = template or "minimal"
    try:
        info = get_template(name)
    except KeyError as e:
        print(str(e))
        raise SystemExit(1) from None

    content = load_template_content(name)
    overrides = _build_overrides(content, model, url)
    content = strip_spdx_header(content)
    if overrides:
        content = apply_overrides(content, overrides)

    _write_template_output(content, output, info.name, info.title)


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
    available templates, --search to find by keyword, --model/--url to
    pre-fill the two fields every config needs.

    Examples:
        aiperf config init --list
        aiperf config init --template goodput_slo --model my-model --url http://host/v1
    """
    with exit_on_error(title="Template Error"):
        if search:
            _handle_search(search, verbose=verbose)
            return
        if list_templates:
            _handle_list(category, verbose=verbose)
            return
        _generate_template(template, model, url, output)
