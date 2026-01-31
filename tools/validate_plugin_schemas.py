#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate categories.yaml and plugins.yaml against their Pydantic schemas.

This script validates:
1. YAML structure against Pydantic models (CategoriesManifest, PluginsManifest)
2. All plugin classes can be loaded

Usage:
    python tools/validate_plugin_schemas.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import ValidationError
from ruamel.yaml import YAML

PLUGIN_DIR = Path(__file__).parent.parent / "src/aiperf/plugin"


def validate_categories() -> list[str]:
    """Validate categories.yaml against CategoriesManifest schema."""
    from aiperf.plugin.schema import CategoriesManifest

    errors: list[str] = []
    yaml_path = PLUGIN_DIR / "categories.yaml"

    if not yaml_path.exists():
        errors.append(f"File not found: {yaml_path}")
        return errors

    yaml = YAML(typ="safe")
    data = yaml.load(yaml_path.read_text())

    try:
        CategoriesManifest.model_validate(data)
    except ValidationError as e:
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"]) if err["loc"] else "(root)"
            errors.append(f"  {loc}: {err['msg']}")

    return errors


def validate_plugins() -> list[str]:
    """Validate plugins.yaml against PluginsManifest schema."""
    from aiperf.plugin.schema import PluginsManifest

    errors: list[str] = []
    yaml_path = PLUGIN_DIR / "plugins.yaml"

    if not yaml_path.exists():
        errors.append(f"File not found: {yaml_path}")
        return errors

    yaml = YAML(typ="safe")
    data = yaml.load(yaml_path.read_text())

    try:
        PluginsManifest.model_validate(data)
    except ValidationError as e:
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"]) if err["loc"] else "(root)"
            errors.append(f"  {loc}: {err['msg']}")

    return errors


def validate_class_loading() -> list[str]:
    """Validate all registered plugin classes can be loaded."""
    from aiperf.plugin import plugins

    errors: list[str] = []

    for category in plugins.list_categories():
        for entry in plugins.list_entries(category):
            try:
                entry.load()
            except Exception as e:
                errors.append(f"  {category}.{entry.name}: {e}")

    return errors


def main() -> int:
    """Validate plugin YAML files and class loading."""
    all_valid = True

    # Validate categories.yaml
    print("Validating categories.yaml...", end=" ")
    errors = validate_categories()
    if errors:
        print("FAILED")
        for error in errors:
            print(error)
        all_valid = False
    else:
        print("OK")

    # Validate plugins.yaml
    print("Validating plugins.yaml...", end=" ")
    errors = validate_plugins()
    if errors:
        print("FAILED")
        for error in errors:
            print(error)
        all_valid = False
    else:
        print("OK")

    # Validate all plugin classes can be loaded
    print("Validating plugin classes can be loaded...", end=" ")
    errors = validate_class_loading()
    if errors:
        print("FAILED")
        for error in errors:
            print(error)
        all_valid = False
    else:
        print("OK")

    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
