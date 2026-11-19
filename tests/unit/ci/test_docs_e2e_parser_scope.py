# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The parser must only see markdown the repository tracks.

``parse_directory`` is handed the repo root, so an rglob also walks ignored
trees: a git worktree kept under the checkout (``AGENTS.md`` encourages
``.worktrees/``, which is gitignored), a local virtualenv, a sibling clone. A
second copy of ``docs/`` in any of them presents the same ``setup-`` tag
twice, and the duplicate-setup branch in ``parser.py`` calls ``sys.exit(1)``
-- turning a green local ``make test`` red while CI, which checks out clean,
stays unaffected.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HARNESS = REPO_ROOT / "tests/ci/test_docs_end_to_end"


@pytest.fixture
def repo_with_untracked_docs_copy(tmp_path: Path) -> Path:
    """A real git checkout holding a tracked doc plus an ignored copy of it."""
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    (root / ".worktrees" / "fake" / "docs").mkdir(parents=True)

    tag = "setup-vllm-default-" + "endpoint-server"
    doc = f"<!-- {tag} -->\n```bash\necho start-server\n```\n<!-- /{tag} -->\n"
    (root / "docs" / "guide.md").write_text(doc)
    (root / ".worktrees" / "fake" / "docs" / "guide.md").write_text(doc)
    (root / ".gitignore").write_text(".worktrees/\n")

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "docs/guide.md", ".gitignore"], cwd=root, check=True)
    return root


def test_ignored_copy_of_a_doc_is_not_parsed(
    repo_with_untracked_docs_copy: Path,
) -> None:
    """The ignored duplicate must not register as a second setup command."""
    from parser import MarkdownParser

    collected = MarkdownParser()._markdown_files(str(repo_with_untracked_docs_copy))

    assert [p.name for p in collected] == ["guide.md"]
    assert not any(".worktrees" in p.parts for p in collected)


def test_duplicate_setup_tag_in_ignored_tree_does_not_exit(
    repo_with_untracked_docs_copy: Path,
) -> None:
    """Parsing a checkout with a worktree under it must not call sys.exit."""
    from parser import MarkdownParser

    servers = MarkdownParser().parse_directory(str(repo_with_untracked_docs_copy))

    assert set(servers) == {"vllm-default"}
    assert servers["vllm-default"].setup_command is not None


def test_falls_back_to_rglob_outside_a_git_checkout(tmp_path: Path) -> None:
    """A plain directory still parses, so vendored copies keep working."""
    from parser import MarkdownParser

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("# not tagged\n")

    collected = MarkdownParser()._markdown_files(str(tmp_path))

    assert [p.name for p in collected] == ["guide.md"]


def test_git_tracked_listing_excludes_an_untracked_sibling_clone(
    repo_with_untracked_docs_copy: Path,
) -> None:
    """An untracked directory that is not gitignored is still excluded."""
    from parser import MarkdownParser

    sibling = repo_with_untracked_docs_copy / "dynamo"
    shutil.copytree(repo_with_untracked_docs_copy / "docs", sibling)

    collected = MarkdownParser()._markdown_files(str(repo_with_untracked_docs_copy))

    assert not any("dynamo" in p.parts for p in collected)
