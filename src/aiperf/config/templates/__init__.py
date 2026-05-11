# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bundled YAML templates and the discovery API.

The bundled `.yaml` files live alongside this `__init__.py`; `discovery.py`
provides the Python API for enumerating, searching, and loading them.
"""

from aiperf.config.templates.discovery import (
    CATEGORY_ORDER,
    TemplateInfo,
    apply_overrides,
    get_template,
    list_templates,
    load_template_content,
    parse_template_meta,
    search_templates,
    strip_spdx_header,
)

__all__ = [
    "CATEGORY_ORDER",
    "TemplateInfo",
    "apply_overrides",
    "get_template",
    "list_templates",
    "load_template_content",
    "parse_template_meta",
    "search_templates",
    "strip_spdx_header",
]
