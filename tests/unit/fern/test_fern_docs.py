# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fern documentation validation tests.

Runs fern CLI commands to validate the documentation configuration,
check for build errors, and verify the dev server starts without errors.

Requires the ``fern`` CLI to be installed globally.
Run with: ``make test-fern-docs`` or ``pytest -m fern``
"""

from __future__ import annotations

import re
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

FERN_READY_PATTERN = re.compile(r"Docs preview server ready")
FERN_ERROR_PATTERN = re.compile(r"\[error\]")
FERN_DEV_TIMEOUT_S = 120


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


def test_fern_docs_dev_starts(staged_fern_docs: Path) -> None:
    """Verify fern docs dev builds and starts without errors.

    Starts ``fern docs dev`` in a subprocess, monitors stdout for the
    "ready" message or ``[error]`` lines, then terminates the server.
    Fails if an error is detected or the server does not become ready
    within the timeout.
    """
    port = _get_free_port()
    proc = subprocess.Popen(
        ["fern", "docs", "dev", "--port", str(port)],
        cwd=staged_fern_docs,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
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
        reader.join(timeout=FERN_DEV_TIMEOUT_S)
        captured = "".join(output_lines)

        if error.is_set():
            pytest.fail(f"fern docs dev reported errors:\n{captured}")

        if not ready.is_set():
            if proc.poll() is not None:
                pytest.fail(
                    f"fern docs dev exited with code {proc.returncode} "
                    f"before becoming ready:\n{captured}"
                )
            pytest.fail(
                f"fern docs dev timed out after {FERN_DEV_TIMEOUT_S}s:\n{captured}"
            )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


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
