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

    AIPerf templates use either `model:` / `models:` and `endpoint.url:` /
    `endpoint.urls:` interchangeably; this inspects `content` to pick the form
    actually present so the override lands on the same key the template declared.

    Body keys can live either under the `benchmark:` envelope key (the v2
    canonical shape), hoisted at the top level (shortcut), or in both places.
    For each override, prefer wherever the matching key is already present.
    """
    import yaml as _yaml

    overrides: dict[str, Any] = {}
    if not (model or url):
        return overrides

    raw = _yaml.safe_load(content) or {}
    body = raw.get("benchmark") if isinstance(raw.get("benchmark"), dict) else None

    def _set_under(key_path: str, container: str, value: Any) -> None:
        """Place `value` under `key_path` ('a.b' -> nested) within `container`
        (either '' for top-level or 'benchmark')."""
        target = overrides
        if container == "benchmark":
            target = overrides.setdefault("benchmark", {})
        parts = key_path.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value

    if model:
        # Prefer wherever the key already lives.
        if "model" in raw or "models" in raw:
            container = ""
            src = raw
        elif body and ("model" in body or "models" in body):
            container = "benchmark"
            src = body
        else:
            container = "benchmark" if body is not None else ""
            src = body or raw
        key = "model" if "model" in src else "models"
        _set_under(key, container, model if key == "model" else [model])
    if url:
        if "endpoint" in raw and isinstance(raw["endpoint"], dict):
            container = ""
            ep = raw["endpoint"]
        elif body and isinstance(body.get("endpoint"), dict):
            container = "benchmark"
            ep = body["endpoint"]
        else:
            container = "benchmark" if body is not None else ""
            ep = {}
        url_key = "url" if "url" in ep else "urls"
        _set_under(
            f"endpoint.{url_key}", container, url if url_key == "url" else [url]
        )
    return overrides
