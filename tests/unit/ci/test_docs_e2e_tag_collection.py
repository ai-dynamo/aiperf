# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prose that documents the tag syntax must not be run as a benchmark.

``MarkdownParser.parse_directory`` walks every markdown file in the repo and
matches tags line by line, with no notion of a code fence. So a page that
explains the convention by showing a complete tag -- a README, CONTRIBUTING,
a design note -- has that example collected and executed against a real
inference server, silently adding a command nobody meant to run. Writing the
docs for this harness is exactly when that happens.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HARNESS = REPO_ROOT / "tests/ci/test_docs_end_to_end"


@pytest.fixture(autouse=True)
def _contain_import_side_effects():
    saved_path = list(sys.path)
    saved_modules = set(sys.modules)
    sys.path.insert(0, str(HARNESS))
    try:
        yield
    finally:
        sys.path[:] = saved_path
        for name in set(sys.modules) - saved_modules:
            del sys.modules[name]


def _collected_files() -> set[str]:
    from parser import MarkdownParser

    parser = MarkdownParser()
    parser.parse_directory(str(REPO_ROOT))
    return {
        str(Path(command.file_path).relative_to(REPO_ROOT))
        for server in parser.servers.values()
        for command in server.aiperf_commands
    }


def test_only_user_facing_guides_contribute_commands() -> None:
    """Every collected command must come from a guide under ``docs/``."""
    stray = sorted(f for f in _collected_files() if not f.startswith("docs/"))
    assert not stray, (
        f"these files have their example tags collected and run as real "
        f"benchmarks: {stray}. Describe the tag with a placeholder such as "
        f"'{{kind}}-{{server}}-endpoint-server' instead of spelling it out."
    )


@pytest.mark.parametrize(
    "path",
    [
        "CONTRIBUTING.md",
        "tests/ci/test_docs_end_to_end/README.md",
    ],
)  # fmt: skip
def test_the_tag_documentation_is_not_itself_a_test(path: str) -> None:
    """The two pages that explain the convention, named explicitly.

    They are the likeliest to regress, because the natural way to document a
    tag is to show one.
    """
    assert path not in _collected_files()
