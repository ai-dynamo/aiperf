#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate that every template YAML has a valid ``# @template`` metadata block.

Usage:
    python -m tools.validate_template_metadata
    python -m tools.validate_template_metadata --verbose
"""

from __future__ import annotations

import sys
import time

from tools._core import console, print_section, print_step, print_up_to_date


def main() -> int:
    print_section("Template Metadata Validation")
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    start = time.perf_counter()

    try:
        from aiperf.config.templates import _load_all_templates

        # Clear cache so we always re-parse from disk
        _load_all_templates.cache_clear()
        templates = _load_all_templates()
    except (ValueError, ImportError) as e:
        console.print(f"  [red]x[/] {e}")
        return 1

    if verbose:
        for t in templates:
            print_up_to_date(f"{t.name}: {t.title} [{t.category}] ({t.difficulty})")
        print_step(f"Validated {len(templates)} templates")
        console.print()

    elapsed = time.perf_counter() - start
    console.print(
        f"[bold green]v[/] All {len(templates)} templates have valid metadata. "
        f"[dim]({elapsed:.2f}s)[/]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
