# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SPDX header generation across every enforced source format."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[3]


def load_tool(name: str) -> ModuleType:
    """Load a standalone tool without importing the dependency-bearing package."""
    path = ROOT / f"tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load tool from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


add_copyright = load_tool("add_copyright")
checker = load_tool("check_spdx_headers")

LICENSE_TEXT = (ROOT / "tools/COPYRIGHT").read_text().strip()


@pytest.mark.parametrize(
    "filename",
    [
        *(f"example{suffix}" for suffix in sorted(checker.SOURCE_SUFFIXES)),
        *sorted(checker.SOURCE_FILENAMES),
        *(f"{prefix}example" for prefix in checker.SOURCE_FILENAME_PREFIXES),
    ],
)
def test_fixer_generates_every_required_header(tmp_path: Path, filename: str) -> None:
    """Every file classified by the checker has a compatible fixer handler."""
    path = tmp_path / filename
    path.write_text("content\n", encoding="utf-8")

    changed, status = add_copyright.process_file(path, LICENSE_TEXT)

    assert (changed, status) == (True, "added copyright")
    assert checker.validate_file(tmp_path, Path(filename)) == []


def test_fixer_preserves_markdown_frontmatter(tmp_path: Path) -> None:
    """Markdown metadata remains the first construct in frontmatter files."""
    path = tmp_path / "rule.md"
    frontmatter = (
        "---\n"
        "description: Example rule\n"
        "alwaysApply: true\n"
        "one: 1\n"
        "two: 2\n"
        "three: 3\n"
        "four: 4\n"
        "five: 5\n"
        "six: 6\n"
        "seven: 7\n"
        "---\n"
    )
    path.write_text(
        frontmatter + "\n# Rule\n",
        encoding="utf-8",
    )

    changed, status = add_copyright.process_file(path, LICENSE_TEXT)

    assert (changed, status) == (True, "added copyright")
    assert path.read_text(encoding="utf-8").startswith(
        frontmatter + "<!--\n" + LICENSE_TEXT + "\n-->\n"
    )
    assert checker.validate_file(tmp_path, Path("rule.md")) == []


@pytest.mark.parametrize(
    "legacy_copyright",
    [
        "SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION",
        "SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION",
        "SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION.",
        "SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES",
        "SPDX-FileCopyrightText: Copyright (c) 2026-2025 NVIDIA CORPORATION "
        "& AFFILIATES. All rights reserved.",
        "SPDX-FileCopyrightText: Copyright (c) 2025 Nvidia Corporation "
        "& Affiliates. All rights reserved.",
    ],
)
def test_fixer_normalizes_legacy_nvidia_headers(
    tmp_path: Path, legacy_copyright: str
) -> None:
    """Legacy NVIDIA SPDX forms are repaired without adding a second header."""
    path = tmp_path / "legacy.py"
    path.write_text(
        f"# {legacy_copyright}\n# SPDX-License-Identifier: Apache-2.0\n",
        encoding="utf-8",
    )

    changed, status = add_copyright.process_file(path, LICENSE_TEXT)

    assert (changed, status) == (True, "updated year")
    assert path.read_text(encoding="utf-8").count("NVIDIA CORPORATION") == 1
    assert checker.validate_file(tmp_path, Path("legacy.py")) == []


@pytest.mark.parametrize(
    ("filename", "content", "critical_prefix"),
    [
        pytest.param(
            "script.py",
            "\ufeff#!/usr/bin/env python3\n# coding: utf-8\nprint('ok')\n",
            "\ufeff#!/usr/bin/env python3\n# coding: utf-8\n",
            id="bom-and-python-preamble",
        ),
        pytest.param(
            "tool.mjs",
            "#!/usr/bin/env node\nconsole.log('ok');\n",
            "#!/usr/bin/env node\n",
            id="javascript-shebang",
        ),
        pytest.param(
            "Dockerfile.custom",
            "# syntax=docker/dockerfile:1\n# check=error=true\nFROM scratch\n",
            "# syntax=docker/dockerfile:1\n# check=error=true\n",
            id="docker-directives",
        ),
        pytest.param(
            "style.css",
            '@charset "UTF-8";\nbody {}\n',
            '@charset "UTF-8";\n',
            id="css-charset",
        ),
        pytest.param(
            "index.html",
            "<!DOCTYPE html>\n<html></html>\n",
            "<!DOCTYPE html>\n",
            id="html-doctype",
        ),
    ],
)
def test_fixer_preserves_critical_preamble(
    tmp_path: Path, filename: str, content: str, critical_prefix: str
) -> None:
    """Syntax-critical leading constructs remain ahead of the SPDX header."""
    path = tmp_path / filename
    path.write_text(content, encoding="utf-8")

    changed, status = add_copyright.process_file(path, LICENSE_TEXT)

    assert (changed, status) == (True, "added copyright")
    assert path.read_text(encoding="utf-8").startswith(critical_prefix)
    assert checker.validate_file(tmp_path, Path(filename)) == []
