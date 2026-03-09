# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Display helpers for plugin CLI commands."""

from __future__ import annotations

from collections import Counter

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from aiperf.plugin import plugins
from aiperf.plugin.types import TypeNotFoundError

console = Console()

_ACRONYMS = frozenset(
    {"ui", "api", "gpu", "cpu", "http", "zmq", "csv", "json", "hf", "tei", "url"}
)


def _title(category: str) -> str:
    """Format category name: snake_case -> Title Case (with acronyms uppercase)."""
    return " ".join(
        w.upper() if w in _ACRONYMS else w.title() for w in category.split("_")
    )


def _hint(msg: str) -> None:
    """Print a dim hint message."""
    console.print(f"\n[dim]{msg}[/dim]")


def show_packages_detailed() -> None:
    """Show installed packages with full details (default view)."""
    pkg_names = plugins.list_packages()
    if not pkg_names:
        console.print("[yellow]No packages found[/yellow]")
        return

    counts = Counter(entry.package for entry in plugins.iter_entries())

    table = Table(title="Installed Packages", show_lines=True, expand=True)
    table.add_column("Package", style="cyan", no_wrap=True)
    table.add_column("Version", no_wrap=True)
    table.add_column("Plugins", no_wrap=True, justify="right")
    table.add_column("Description", ratio=1)

    for pkg in pkg_names:
        meta = plugins.get_package_metadata(pkg)
        table.add_row(
            pkg,
            meta.version or "[dim]-[/dim]",
            str(counts.get(pkg, 0)),
            meta.description or "[dim]-[/dim]",
        )

    console.print(table)
    _hint("Usage: aiperf plugins --all to see all categories and plugins")


def show_categories_overview() -> None:
    """Show all plugin categories and their plugins."""
    categories = plugins.list_categories()
    if not categories:
        console.print("[yellow]No categories found[/yellow]")
        return

    table = Table(title="Plugin Categories", show_lines=True, expand=True)
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Plugins", ratio=1)
    for cat in categories:
        names = ", ".join(f"[italic]{e.name}[/]" for e in plugins.list_entries(cat))
        table.add_row(_title(cat), names)
    console.print(table)

    _hint("Usage: aiperf plugins <category> to see available types")


def show_category_types(category: str) -> None:
    """List all types in a category."""
    entries = plugins.list_entries(category)
    if not entries:
        console.print(f"[yellow]Unknown category: {category}[/yellow]")
        _hint("Available: " + ", ".join(plugins.list_categories()))
        return

    if (meta := plugins.get_category_metadata(category)) and meta.get("description"):
        console.print(f"[dim]{meta['description'].strip()}[/dim]\n")

    table = Table(title=f"{_title(category)} Types", show_lines=True, expand=True)
    table.add_column("Type", style="cyan", no_wrap=True)
    table.add_column("Description", overflow="ellipsis", no_wrap=True, ratio=1)
    for entry in entries:
        desc = (entry.description or "").strip() or "[dim]-[/dim]"
        table.add_row(entry.name, desc)
    console.print(table)
    _hint(f"Usage: aiperf plugins {category} <type> for details")


def show_type_details(category: str, name: str) -> None:
    """Show details about a specific plugin type."""
    try:
        entry = plugins.get_entry(category, name)
    except (KeyError, TypeNotFoundError):
        console.print(f"[red]Not found: {category}:{name}[/red]")
        if entries := plugins.list_entries(category):
            _hint("Available: " + ", ".join(e.name for e in entries))
        return

    console.print(
        Panel(
            f"[bold]Type:[/bold] {entry.name}\n"
            f"[bold]Category:[/bold] {entry.category}\n"
            f"[bold]Package:[/bold] {entry.package}\n"
            f"[bold]Class:[/bold] {entry.class_path}\n\n"
            f"{entry.description or '[dim]No description[/dim]'}",
            title=f"{category}:{name}",
            border_style="cyan",
        )
    )


def run_validate() -> None:
    """Validate all registered plugins."""
    console.print("[bold]Validating plugins...[/bold]\n")

    checks = [
        (
            "Class paths",
            plugins.validate_all(check_class=True),
            lambda cat, errs: [f"{cat}:{n} - {e}" for n, e in errs],
        ),
    ]

    all_passed = True
    for label, errors, fmt in checks:
        if errors:
            console.print(f"[red]✗[/red] {label}")
            for cat, items in errors.items():
                for line in fmt(cat, items):
                    console.print(f"    {line}")
            all_passed = False
        else:
            console.print(f"[green]✓[/green] {label}")

    color = "green" if all_passed else "red"
    msg = "All checks passed" if all_passed else "Validation failed"
    console.print(f"\n[bold {color}]{msg}[/]")
