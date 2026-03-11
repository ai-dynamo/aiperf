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
import threading

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


def test_fern_check() -> None:
    """Validate the Fern definition has no errors."""
    result = subprocess.run(
        ["fern", "check"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"fern check failed (exit {result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_fern_check_strict() -> None:
    """Validate the Fern definition with strict broken-link checking."""
    result = subprocess.run(
        ["fern", "check", "--warnings", "--strict-broken-links"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"fern check --strict-broken-links failed (exit {result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_fern_docs_dev_starts() -> None:
    """Verify fern docs dev builds and starts without errors.

    Starts ``fern docs dev`` in a subprocess, monitors stdout for the
    "ready" message or ``[error]`` lines, then terminates the server.
    Fails if an error is detected or the server does not become ready
    within the timeout.
    """
    port = _get_free_port()
    proc = subprocess.Popen(
        ["fern", "docs", "dev", "--port", str(port)],
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
