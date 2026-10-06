# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The new-doc tag gate must not report success without checking anything.

It exists to stop a new guide landing untagged. A gate that returns an empty
candidate list when git fails is worse than no gate: it prints a success line
and a reviewer reasonably believes the docs were inspected.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools/check_docs_e2e_tags.py"

sys.path.insert(0, str(REPO_ROOT / "tools"))


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_an_unusable_base_ref_fails_the_gate() -> None:
    result = _run("--base", "definitely-not-a-ref")
    assert result.returncode != 0, (
        "a failed diff must fail the gate; returning no candidates makes it "
        "print success having inspected nothing"
    )
    assert "Refusing to report success" in (result.stdout + result.stderr)


def test_a_usable_base_ref_still_runs() -> None:
    result = _run("--base", "origin/main")
    assert result.returncode in (0, 1), f"unexpected crash: {result.stderr}"


def test_added_file_paths_survive_whitespace(tmp_path, monkeypatch) -> None:
    """``--name-only`` quotes paths with spaces; splitting on whitespace shears
    them into fragments that resolve to nothing, silently skipping the very
    file being added."""
    import check_docs_e2e_tags as gate

    captured = {}

    class _Result:
        stdout = "docs/a b.md\x00docs/plain.md\x00"

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _Result()

    monkeypatch.setattr(gate.subprocess, "run", _fake_run)
    files = gate.added_markdown_files("origin/main")

    assert "-z" in captured["cmd"], "must request NUL-delimited output"
    assert Path("docs/a b.md") in files, f"lost the spaced path: {files}"
    assert Path("docs/plain.md") in files


def test_tilde_fenced_commands_are_detected(tmp_path) -> None:
    """CommonMark allows ``~~~``; missing it let a guide slip through untagged."""
    import check_docs_e2e_tags as gate

    doc = tmp_path / "tilde.md"
    doc.write_text("# T\n\n~~~bash\naiperf profile --model m\n~~~\n")
    assert gate.has_runnable_command(doc)


def test_a_shorter_fence_does_not_close_a_longer_one(tmp_path) -> None:
    """A backtick run inside a tilde fence must not end the block early."""
    import check_docs_e2e_tags as gate

    doc = tmp_path / "nested.md"
    doc.write_text("# N\n\n~~~~bash\n```\naiperf profile --model m\n```\n~~~~\n")
    assert gate.has_runnable_command(doc)


def test_prose_outside_a_fence_is_not_a_command(tmp_path) -> None:
    import check_docs_e2e_tags as gate

    doc = tmp_path / "prose.md"
    doc.write_text("# P\n\naiperf profile is the command you run.\n")
    assert not gate.has_runnable_command(doc)


def test_tagging_a_non_aiperf_block_does_not_satisfy_the_gate(tmp_path) -> None:
    """The parser categorises a block by its tag without reading the body.

    A doc could therefore tag an ``echo ok`` block and pass while its real
    ``aiperf profile`` command sat untagged beside it.
    """
    import check_docs_e2e_tags as gate

    doc = tmp_path / "loophole.md"
    doc.write_text(
        "# L\n\n"
        "<!-- aiperf-run-vllm-default-openai-endpoint-server -->\n"
        "```bash\necho ok\n```\n"
        "<!-- /aiperf-run-vllm-default-openai-endpoint-server -->\n\n"
        "```bash\naiperf profile --model m\n```\n"
    )
    assert gate.tagged_run_count(doc) == 0, (
        "a tagged block with no aiperf command must not count as coverage"
    )
