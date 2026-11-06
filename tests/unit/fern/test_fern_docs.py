# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fern documentation validation tests.

Runs fern CLI commands to validate the documentation configuration,
check for build errors, and verify the dev server starts without errors.

Requires the ``fern`` CLI to be installed globally.
Run with: ``make test-fern-docs`` or ``pytest -m fern``
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

FERN_READY_PATTERN = re.compile(r"Docs preview server ready")
FERN_ERROR_PATTERN = re.compile(r"\[error\]")
FERN_DEV_TIMEOUT_S = 120
# Per-attempt budget for the one-time docs-preview-bundle download (~365 MB).
# Cold downloads on GitHub-hosted runners finish in well under 60 s
# typically; the observed flake (main push, 2026-10-15) was still
# downloading when the old 120 s budget expired, so its true completion time
# is unknown (censored observation). 300 s is 2.5x that previous budget,
# chosen to absorb a several-fold-slower-than-typical download rather than a
# measured tail.
FERN_PREWARM_TIMEOUT_S = 300


def _get_free_port() -> int:
    """Return an available ephemeral port by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


_fern_installed = shutil.which("fern") is not None

pytestmark = [
    pytest.mark.fern,
    pytest.mark.skipif(not _fern_installed, reason="fern CLI not installed"),
]

_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def staged_fern_docs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Stage and convert docs the way CI and ``make fern-preview`` do.

    Copies ``fern/`` and ``docs/`` into a temp tree, runs ``md_to_mdx.py`` to
    convert GitHub Markdown to Fern MDX, then returns the staged ``fern/``
    directory. Fern link validation must run against converted content: raw
    ``docs/`` contains HTML comments that Fern's MDX parser rejects, which
    breaks the link-check rules with a false error.
    """
    staged = tmp_path_factory.mktemp("fern-docs")
    shutil.copytree(
        _REPO_ROOT / "fern",
        staged / "fern",
        ignore=shutil.ignore_patterns(".local-preview", ".preview", ".definition"),
    )
    shutil.copytree(_REPO_ROOT / "docs", staged / "docs")
    subprocess.run(
        [
            sys.executable,
            str(staged / "fern" / "md_to_mdx.py"),
            "--dir",
            str(staged / "docs"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return staged / "fern"


def test_fern_check(staged_fern_docs: Path) -> None:
    """Validate the Fern definition (converted content) has no errors."""
    result = subprocess.run(
        ["fern", "check"],
        cwd=staged_fern_docs,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"fern check failed (exit {result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def _run_fern_docs_dev(staged_fern_docs: Path, timeout_s: int) -> None:
    """Run ``fern docs dev`` until the preview server is ready, then stop it.

    Monitors stdout for the "ready" message or ``[error]`` lines. Fails if an
    error is detected or the server does not become ready within
    ``timeout_s``.
    """
    port = _get_free_port()
    # start_new_session: the CLI spawns setup/runtime children (pnpm, esbuild,
    # the node backend) that can outlive it; a dedicated process group lets the
    # cleanup below reap them instead of leaking them into the pytest session.
    proc = subprocess.Popen(
        ["fern", "docs", "dev", "--port", str(port)],
        cwd=staged_fern_docs,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    output_lines: list[str] = []
    ready = threading.Event()
    error = threading.Event()

    def _read_output() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            output_lines.append(line)
            if FERN_READY_PATTERN.search(line):
                ready.set()
                return
            if FERN_ERROR_PATTERN.search(line):
                error.set()
                return

    reader = threading.Thread(target=_read_output, daemon=True)
    reader.start()

    try:
        reader.join(timeout=timeout_s)
        captured = "".join(output_lines)

        if error.is_set():
            pytest.fail(f"fern docs dev reported errors:\n{captured}")

        if not ready.is_set():
            if proc.poll() is not None:
                pytest.fail(
                    f"fern docs dev exited with code {proc.returncode} "
                    f"before becoming ready:\n{captured}"
                )
            raise TimeoutError(
                f"fern docs dev did not become ready within {timeout_s}s:\n{captured}"
            )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        # Best-effort sweep of orphaned CLI children. With start_new_session
        # the process group id is the CLI's pid; on Windows killpg does not
        # exist and terminate/kill above are all we have.
        killpg = getattr(os, "killpg", None)
        if killpg is not None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                killpg(proc.pid, signal.SIGKILL)


def _invalidate_interrupted_preview_bundle() -> None:
    """Drop bundle state left by an interrupted ``fern docs dev`` setup.

    The pinned fern CLI (5.149.0) extracts the preview bundle straight into
    the live ``~/.fern/app-preview/.next`` directory — there is no atomic
    promotion of a completed bundle — and writes the ``app-preview/etag``
    marker after extraction but *before* its pnpm/esbuild setup step
    finishes. A killed or timed-out attempt can therefore leave either a
    partial ``.next`` with no etag or a matching etag over unfinished setup,
    and the next invocation's ETag fast path ("Using already downloaded
    bundle") would skip the unfinished setup entirely. Removing the whole
    ``app-preview`` subtree — the same unit the CLI itself removes for a
    forced re-download — forces the retry to redo the full download and
    setup. Everything else under ``~/.fern`` (``id``, ``logs/``, other
    bundle kinds) is untouched.

    Only true absence is tolerated: a symlinked ``app-preview`` is refused
    (never delete through a link), and any other deletion failure raises —
    proceeding to the retry with the interrupted bundle still in place would
    risk reusing exactly the half-installed state this invalidation exists
    to discard, so the pre-warm fails loudly instead.
    """
    app_preview = Path.home() / ".fern" / "app-preview"
    if app_preview.is_symlink():
        raise RuntimeError(
            f"refusing to invalidate symlinked fern preview bundle: {app_preview}"
        )
    try:
        shutil.rmtree(app_preview)
    except FileNotFoundError:
        return  # nothing to invalidate
    except OSError as exc:
        raise RuntimeError(
            f"could not invalidate interrupted fern preview bundle at "
            f"{app_preview}: {exc}; retrying would risk reusing unfinished "
            f"setup state, so the pre-warm is aborted"
        ) from exc


@pytest.fixture(scope="session")
def fern_preview_bundle_prewarmed(staged_fern_docs: Path) -> None:
    """Pay the one-time docs-preview-bundle download outside any test budget.

    The first ``fern docs dev`` on a machine downloads and installs a ~365 MB
    preview bundle into the user-level fern cache (``~/.fern``); the CLI
    reuses it whenever the upstream ETag is unchanged. That download is
    network-bound with high variance: observed CI cold starts complete in
    under ~60 s typically, but the 2026-10-15 main-push run was still
    downloading when the test's 120 s budget expired. Warming the cache here
    with a download-sized budget keeps the test's own budget for what it
    actually asserts — a warm server boot.

    One retry on timeout: its value is escaping a wedged or stalled
    connection, and the two attempts bound the total session cost at ~2x
    ``FERN_PREWARM_TIMEOUT_S``. An interrupted attempt can leave partial or
    etag-marked-but-unfinished bundle state (see
    ``_invalidate_interrupted_preview_bundle``), so the bundle is
    invalidated before the retry — the retry never takes the ETag fast path
    on state from an attempt we cut off, and an invalidation failure aborts
    the pre-warm loudly rather than proceeding on a best-effort delete.
    ``[error]`` output and unexpected exits fail immediately without a retry
    and without invalidation: those are deterministic defects, not network
    variance.
    """
    last_error: TimeoutError | None = None
    for _attempt in (1, 2):
        try:
            _run_fern_docs_dev(staged_fern_docs, FERN_PREWARM_TIMEOUT_S)
            return
        except TimeoutError as exc:
            last_error = exc
            _invalidate_interrupted_preview_bundle()
    pytest.fail(
        "fern docs dev failed to become ready during bundle pre-warm after "
        f"2 attempts of {FERN_PREWARM_TIMEOUT_S}s each:\n{last_error}"
    )


def test_fern_docs_dev_starts(
    staged_fern_docs: Path, fern_preview_bundle_prewarmed: None
) -> None:
    """Verify fern docs dev builds and starts without errors.

    The bundle cache is already warm (see ``fern_preview_bundle_prewarmed``),
    so ``FERN_DEV_TIMEOUT_S`` only has to cover a warm boot, which observed
    CI runs complete in well under a minute.
    """
    _run_fern_docs_dev(staged_fern_docs, FERN_DEV_TIMEOUT_S)


def test_fern_check_strict(staged_fern_docs: Path) -> None:
    """Strict validation: broken or relative markdown links must fail.

    ``--strict-broken-links`` promotes broken/relative-link warnings to errors;
    ``--warnings`` just surfaces remaining non-link warnings (auth-skipped
    redirects, accent contrast) in the output without failing the check.
    """
    result = subprocess.run(
        ["fern", "check", "--warnings", "--strict-broken-links"],
        cwd=staged_fern_docs,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"fern check --strict-broken-links failed (exit {result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_fern_broken_links(staged_fern_docs: Path) -> None:
    """Verify Fern finds no broken links in the converted content."""
    result = subprocess.run(
        ["fern", "docs", "broken-links"],
        cwd=staged_fern_docs,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"fern docs broken-links failed (exit {result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
