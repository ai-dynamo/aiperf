# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`aiperf kube init` — generate starter AIPerfJob CRs from bundled templates.

Wraps the same template library used by `aiperf config init` (see
``src/aiperf/config/templates/``) in an AIPerfJob CR shell, so any bundled
template becomes a deployable Kubernetes manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

app = App(name="init")

_CMD = "aiperf kube init"


def _write_wrapped_template(
    content: str,
    output: Path | None,
    info_name: str,
    info_title: str,
) -> None:
    from aiperf.kubernetes import console as kube_console

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
    kube_console.print_success(
        f"Created {output} from template '{info_name}' ({info_title})"
    )
    kube_console.print_info("Next steps:")
    kube_console.print_action(
        f"1. Edit {output} -- update endpoint URLs, model name, and image"
    )
    kube_console.print_action(
        f"2. Run:  aiperf kube profile --config {output} --image <your-image>"
    )
    kube_console.print_action(f"3. Or:   kubectl apply -f {output}")


def _generate_template(
    template: str | None,
    model: str | None,
    url: str | None,
    output: Path | None,
    job_name: str,
) -> None:
    from aiperf.config.templates import (
        apply_overrides,
        get_template,
        load_template_content,
        strip_spdx_header,
    )
    from aiperf.config.templates_cli import build_overrides
    from aiperf.kubernetes.init_template import wrap_as_aiperf_job

    name = template or "minimal"
    try:
        info = get_template(name)
    except KeyError as e:
        print(str(e))
        raise SystemExit(1) from None

    body = load_template_content(name)
    overrides = build_overrides(body, model, url)
    body = strip_spdx_header(body)
    if overrides:
        body = apply_overrides(body, overrides)

    filename = output.name if output else "benchmark.yaml"
    wrapped = wrap_as_aiperf_job(body, filename=filename, job_name=job_name)
    _write_wrapped_template(wrapped, output, info.name, info.title)


@app.default
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
    job_name: Annotated[
        str,
        Parameter(
            name=["--job-name"],
            help="Value for metadata.name on the generated AIPerfJob.",
        ),
    ] = "my-benchmark",
) -> None:
    """Generate a starter AIPerfJob CR from bundled templates.

    Without arguments, generates the 'minimal' template wrapped in an
    AIPerfJob CR. Use --list to browse available templates, --search to find
    by keyword, --model/--url to pre-fill the two fields every config needs.

    Examples:
        aiperf kube init --list
        aiperf kube init --template minimal -o benchmark.yaml
        aiperf kube init -t goodput_slo --model my-model --url http://svc:8000
    """
    from aiperf.cli_utils import exit_on_error
    from aiperf.config.templates_cli import handle_list, handle_search

    with exit_on_error(title="Error Generating Config Template"):
        if search:
            handle_search(search, verbose=verbose, cmd=_CMD)
            return
        if list_templates:
            handle_list(category, verbose=verbose, cmd=_CMD)
            return
        _generate_template(template, model, url, output, job_name)
