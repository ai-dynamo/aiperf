# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared file discovery for the docs-e2e guard tests.

``MarkdownParser.parse_directory`` is handed the repository root, not
``docs/``: ``README.md`` and the package-level READMEs are tagged too. The
guards must walk the same set, or a tag outside ``docs/`` is parsed by the
harness and checked by nothing.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

# Directories the harness never ships but a developer checkout may contain.
# Anything dot-prefixed (``.git``, ``.venv``) is skipped wholesale.
_SKIP = {"node_modules", "__pycache__", "site-packages"}


def markdown_files() -> list[Path]:
    """Every markdown file the docs-e2e parser would read, sorted."""
    return sorted(
        path
        for path in REPO.rglob("*.md")
        if not any(
            part in _SKIP or part.startswith(".")
            for part in path.relative_to(REPO).parts
        )
    )
