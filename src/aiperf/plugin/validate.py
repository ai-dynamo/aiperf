# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plugin registry validation utilities."""

from __future__ import annotations

from aiperf.plugin import plugins


def validate_registry(check_class: bool = False) -> dict[str, list[tuple[str, str]]]:
    """Validate all registered plugins.

    Uses the registry's built-in validation to check that all plugin class paths
    are valid and importable.

    Args:
        check_class: If True, also verify classes exist via AST parsing (slower but thorough).

    Returns:
        Dict mapping category names to lists of (name, error_message) tuples.
        Empty dict means all plugins are valid.
    """
    return plugins.validate_all(check_class=check_class)


def validate_alphabetical_order() -> dict[str, list[str]]:
    """Check that plugin entries within each category are alphabetically ordered.

    Returns:
        Dict mapping category names to lists of error messages.
        Empty dict means all categories are properly sorted.
    """
    errors: dict[str, list[str]] = {}

    for category in plugins.list_categories(include_internal=True):
        entries = plugins.list_entries(category)
        names = [e.name for e in entries]
        sorted_names = sorted(names)

        if names != sorted_names:
            errors[category] = [
                "Not alphabetically sorted",
                f"Current: {names}",
                f"Expected: {sorted_names}",
            ]

    return errors
