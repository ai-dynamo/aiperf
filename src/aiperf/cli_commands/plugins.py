# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plugin management CLI commands.

aiperf plugins                       # Show installed packages with details
aiperf plugins --all                 # Show all categories and plugins
aiperf plugins endpoint              # List endpoint types
aiperf plugins endpoint openai       # Details about openai endpoint
aiperf plugins --validate            # Validate plugins.yaml
"""

from __future__ import annotations

from typing import Annotated

from cyclopts import App, Parameter

from aiperf.plugin.enums import PluginType

app = App(name="plugins")


@app.default
def plugins(
    category: Annotated[
        PluginType | None, Parameter(help="Category to explore")
    ] = None,
    name: Annotated[str | None, Parameter(help="Type name for details")] = None,
    *,
    all_plugins: Annotated[
        bool,
        Parameter(name=["--all", "-a"], help="Show all categories and plugins"),
    ] = False,
    validate: Annotated[
        bool,
        Parameter(name=["--validate", "-v"], help="Validate plugins.yaml"),
    ] = False,
) -> None:
    """Explore AIPerf plugins: aiperf plugins [category] [type]"""
    match (all_plugins, validate, category, name):
        case (_, True, _, _):
            from aiperf.plugin.cli import run_validate

            run_validate()
        case (True, _, _, _):
            from aiperf.plugin.cli import show_categories_overview

            show_categories_overview()
        case (_, _, None, _):
            from aiperf.plugin.cli import show_packages_detailed

            show_packages_detailed()
        case (_, _, cat, None):
            from aiperf.plugin.cli import show_category_types

            show_category_types(cat)
        case (_, _, cat, n):
            from aiperf.plugin.cli import show_type_details

            show_type_details(cat, n)
