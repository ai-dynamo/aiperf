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

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools/check_docs_e2e_tags.py"


@pytest.fixture(autouse=True)
def _contain_import_side_effects():
    """Undo what importing the gate does to this interpreter.

    The gate puts the docs-e2e harness directory on ``sys.path`` so it can
    import the harness's flat modules (``parser``, ``utils``, ``constants``).
    Those names are generic, so once imported they shadow any same-named module
    for the rest of the xdist worker -- a later test importing ``parser`` would
    get the harness's rather than its own.
    """
    saved_path = list(sys.path)
    saved_modules = set(sys.modules)
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        yield
    finally:
        sys.path[:] = saved_path
        for name in set(sys.modules) - saved_modules:
            del sys.modules[name]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _repo_with_one_committed_doc(tmp_path: Path) -> Path:
    """A throwaway repo holding a single committed doc, so a doc added on top
    of it is what ``--diff-filter=A`` reports."""
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    _git(repo.parent, "init", "-q", "-b", "main", str(repo))
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "docs" / "existing.md").write_text("# Existing\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    return repo


def _stage_new_doc(repo: Path, name: str, body: str) -> None:
    (repo / "docs" / name).write_text(body)
    _git(repo, "add", "-A")


UNTAGGED_DOC = "# New\n\n```bash\naiperf profile --model m --url u\n```\n"
TAGGED_DOC = (
    "# New\n\n"
    "<!-- aiperf-run-vllm-default-openai-endpoint-server -->\n"
    "```bash\naiperf profile --model m --url u\n```\n"
    "<!-- /aiperf-run-vllm-default-openai-endpoint-server -->\n"
)


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


def test_a_usable_base_ref_reaches_a_verdict() -> None:
    """Exit 1 alone proves nothing: a failed diff exits 1 too.

    Assert the run actually reached a verdict about the docs, so this test
    cannot pass on the very failure the test above is about.
    """
    result = _run("--base", "origin/main")
    output = result.stdout + result.stderr
    assert result.returncode in (0, 1), f"unexpected crash: {result.stderr}"
    assert "Refusing to report success" not in output, (
        f"the diff failed rather than checking any docs: {output}"
    )
    assert "docs-e2e tags" in output, f"no verdict about the docs was printed: {output}"


def test_main_fails_on_a_newly_added_untagged_doc(tmp_path, monkeypatch) -> None:
    """The gate's whole point, exercised through ``main``.

    Every other test here calls a helper directly, so none of them notice if
    the pieces stop being wired together -- a candidate list that never reached
    the offender check would leave all of them green.
    """
    import check_docs_e2e_tags as gate

    repo = _repo_with_one_committed_doc(tmp_path)
    _stage_new_doc(repo, "untagged.md", UNTAGGED_DOC)
    monkeypatch.setattr(gate, "REPO_ROOT", repo)
    monkeypatch.setattr(sys, "argv", ["check_docs_e2e_tags.py", "--base", "HEAD"])

    with pytest.raises(SystemExit) as excinfo:
        gate.main()
    assert excinfo.value.code == 1


def test_main_passes_once_the_same_doc_is_tagged(tmp_path, monkeypatch, capsys) -> None:
    import check_docs_e2e_tags as gate

    repo = _repo_with_one_committed_doc(tmp_path)
    _stage_new_doc(repo, "untagged.md", TAGGED_DOC)
    monkeypatch.setattr(gate, "REPO_ROOT", repo)
    monkeypatch.setattr(sys, "argv", ["check_docs_e2e_tags.py", "--base", "HEAD"])

    gate.main()  # no SystemExit
    assert "carry docs-e2e tags" in capsys.readouterr().out


def test_a_kubernetes_guide_is_not_demanded_to_carry_a_tag(
    tmp_path, monkeypatch
) -> None:
    """The harness has no cluster, so ``aiperf kube`` is not runnable here.

    Treating every ``aiperf`` line as runnable made the gate block any new page
    under ``docs/kubernetes/`` until it carried a tag the suite could not act
    on.
    """
    import check_docs_e2e_tags as gate

    repo = _repo_with_one_committed_doc(tmp_path)
    _stage_new_doc(
        repo, "kube.md", "# K\n\n```bash\naiperf kube deploy -f job.yaml\n```\n"
    )
    monkeypatch.setattr(gate, "REPO_ROOT", repo)
    monkeypatch.setattr(sys, "argv", ["check_docs_e2e_tags.py", "--base", "HEAD"])

    gate.main()  # no SystemExit


def test_an_env_prefixed_command_still_counts(tmp_path) -> None:
    """``AIPERF_X=1 aiperf profile ...`` is a profile run, not prose."""
    import check_docs_e2e_tags as gate

    doc = tmp_path / "env.md"
    doc.write_text(
        "# E\n\n```bash\n"
        "AIPERF_HTTP_VIDEO_POLL_INTERVAL=0.5 aiperf profile --model m\n```\n"
    )
    assert gate.has_runnable_command(doc)


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
