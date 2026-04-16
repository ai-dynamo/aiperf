# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaged HTML templates for Agentic Code reporting."""

from __future__ import annotations

from importlib.resources import files

_TEMPLATE_PACKAGE = "aiperf.dataset.agentic_code_gen.reporting.templates"


def read_template(name: str) -> str:
    """Read a packaged template as UTF-8 text."""
    return files(_TEMPLATE_PACKAGE).joinpath(name).read_text(encoding="utf-8")


def render_template(name: str, **replacements: str) -> str:
    """Render a static template by replacing explicit __PLACEHOLDER__ tokens."""
    rendered = read_template(name)
    for key, value in replacements.items():
        rendered = rendered.replace(f"__{key}__", value)
    return rendered
