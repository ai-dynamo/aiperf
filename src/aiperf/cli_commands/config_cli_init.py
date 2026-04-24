# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`aiperf config init` command — generate starter configs from bundled templates."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from aiperf.cli_commands.config_cli import config_app
from aiperf.cli_utils import exit_on_error
from aiperf.config.templates_cli import (
    build_overrides,
    handle_list,
    handle_search,
)

_CMD = "aiperf config init"


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
    overrides = build_overrides(content, model, url)
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
            handle_search(search, verbose=verbose, cmd=_CMD)
            return
        if list_templates:
            handle_list(category, verbose=verbose, cmd=_CMD)
            return
        _generate_template(template, model, url, output)
