# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared CLI helpers for template-scaffolding commands.

Used by both `aiperf config init` and `aiperf kube init`. The `cmd` parameter
customizes hint text so each command surfaces its own invocation in
"Run '<cmd> --list'" / "Use '<cmd> --template <name>'" messages.
"""

from __future__ import annotations

from typing import Any

from aiperf.config.templates import (
    CATEGORY_ORDER,
    TemplateInfo,
    search_templates,
)
from aiperf.config.templates import (
    list_templates as _list_templates,
)


def print_template_table(
    templates: list[TemplateInfo],
    *,
    verbose: bool = False,
) -> None:
    """Print templates as a Rich table grouped by category."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    by_category: dict[str, list[TemplateInfo]] = {}
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


def handle_search(
    search: str,
    *,
    verbose: bool,
    cmd: str = "aiperf config init",
) -> None:
    """Print templates matching `search`, or a hint if none match."""
    results = search_templates(search)
    if not results:
        print(f"No templates match '{search}'.")
        print(f"Run '{cmd} --list' to see all templates.")
        return
    print_template_table(results, verbose=verbose)


def handle_list(
    category: str | None,
    *,
    verbose: bool,
    cmd: str = "aiperf config init",
) -> None:
    """Print all templates, optionally filtered by category."""
    results = _list_templates(category=category)
    if not results:
        print(f"No templates in category '{category}'.")
        return
    print_template_table(results, verbose=verbose)
    print(f"Use '{cmd} --template <name>' to generate a template.")


def build_overrides(
    content: str,
    model: str | None,
    url: str | None,
) -> dict[str, Any]:
    """Build an overrides dict matching the singular/plural form the template uses.

    AIPerf templates use either ``model:`` / ``models:`` and ``endpoint.url:`` /
    ``endpoint.urls:`` interchangeably. This inspects ``content`` to pick the
    form actually present so the override lands on the same key the template
    declared.

    Envelope shape: ``endpoint`` is a body field. When the template uses the
    envelope shape (has a top-level ``benchmark:`` key), the override is
    nested under ``benchmark.endpoint`` so it merges cleanly with the
    template's body. Flat templates (no ``benchmark:`` key) keep the flat
    top-level placement.

    The ``model`` shorthand may live at envelope top level (``model:``) or
    inside the body (``benchmark.model:``); whichever form is declared (or
    envelope when neither is) wins.
    """
    import yaml as _yaml

    overrides: dict[str, Any] = {}
    if not (model or url):
        return overrides

    raw = _yaml.safe_load(content) or {}
    body = raw.get("benchmark") if isinstance(raw.get("benchmark"), dict) else None
    has_envelope = body is not None
    body = body or {}

    if model:
        if "model" in raw or "models" in raw:
            # Top-level (envelope shorthand or flat shape).
            key = "model" if "model" in raw else "models"
            overrides[key] = model if key == "model" else [model]
        elif "model" in body or "models" in body:
            # Inside the body.
            key = "model" if "model" in body else "models"
            overrides.setdefault("benchmark", {})[key] = (
                model if key == "model" else [model]
            )
        else:
            # Neither form declared. Default to envelope top level (the
            # canonical shorthand entry point for templates).
            overrides["models"] = [model]

    if url:
        # Endpoint is a body field. In envelope-shape templates we look under
        # benchmark.endpoint; in flat templates we fall back to the
        # top-level endpoint key so the snippet round-trips unchanged.
        if has_envelope:
            ep = (
                body.get("endpoint", {})
                if isinstance(body.get("endpoint"), dict)
                else {}
            )
            url_key = "url" if "url" in ep else "urls"
            overrides.setdefault("benchmark", {}).setdefault("endpoint", {})[
                url_key
            ] = url if url_key == "url" else [url]
        else:
            ep = (
                raw.get("endpoint", {}) if isinstance(raw.get("endpoint"), dict) else {}
            )
            url_key = "url" if "url" in ep else "urls"
            overrides.setdefault("endpoint", {})[url_key] = (
                url if url_key == "url" else [url]
            )

    return overrides
