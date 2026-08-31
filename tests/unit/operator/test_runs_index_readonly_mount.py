# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only-mount behavior of ``runs_index.open_readonly``.

The results-server sidecar used to mount the results PVC read-only. SQLite in
WAL mode requires every reader to create/attach the ``-wal``/``-shm`` sidecars,
so the open failed and the runs index silently degraded to filesystem scans.
These tests pin the diagnosis: a read-only filesystem raises the dedicated
``ReadOnlyMountError`` rather than a generic OperationalError, and a normal
writable mount still opens cleanly.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path

import pytest

from aiperf.operator import runs_index


@pytest.fixture
async def created_index(tmp_path: Path) -> Path:
    """Create + cleanly close a real WAL-mode index, leaving no sidecars."""
    path = tmp_path / ".aiperf_index.sqlite"
    await runs_index.open(path)
    await runs_index.close()
    yield path
    await runs_index.close()


def _freeze_directory(directory: Path) -> None:
    """Make ``directory`` and its files read-only, emulating a read-only mount."""
    for entry in directory.iterdir():
        os.chmod(entry, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    os.chmod(directory, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP)


def _thaw_directory(directory: Path) -> None:
    os.chmod(directory, 0o755)
    for entry in directory.iterdir():
        os.chmod(entry, 0o644)


@pytest.mark.asyncio
async def test_open_readonly_writable_mount_succeeds(created_index: Path) -> None:
    await runs_index.open_readonly(created_index)
    assert runs_index.is_open()
    assert await runs_index.get_meta("schema_version") == "1"


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="requires a non-root POSIX permission model",
)
@pytest.mark.asyncio
async def test_open_readonly_read_only_mount_raises_readonly_mount_error(
    created_index: Path,
) -> None:
    _freeze_directory(created_index.parent)
    try:
        with pytest.raises(runs_index.ReadOnlyMountError) as excinfo:
            await runs_index.open_readonly(created_index)
    finally:
        _thaw_directory(created_index.parent)

    assert "read-write" in str(excinfo.value)
    assert not runs_index.is_open(), "failed open must not leave a live connection"


@pytest.mark.asyncio
async def test_open_readonly_missing_file_does_not_claim_readonly_mount(
    tmp_path: Path,
) -> None:
    with pytest.raises(Exception) as excinfo:
        await runs_index.open_readonly(tmp_path / "absent.sqlite")
    assert not isinstance(excinfo.value, runs_index.ReadOnlyMountError)
    await runs_index.close()


@pytest.mark.asyncio
async def test_results_server_lifespan_logs_error_when_existing_index_unopenable(
    created_index: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from aiperf.operator import results_server

    async def _boom(path: Path) -> None:
        raise runs_index.ReadOnlyMountError(f"cannot open {path}")

    holder: list = [None]
    lifespan = results_server._build_lifespan(created_index.parent, [None], holder)
    with caplog.at_level(logging.INFO):
        monkey = runs_index.open_readonly
        runs_index.open_readonly = _boom  # type: ignore[assignment]
        try:
            async with lifespan(None):  # type: ignore[arg-type]
                pass
        finally:
            runs_index.open_readonly = monkey  # type: ignore[assignment]

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "an existing but unopenable index must be logged loudly"
    assert "could not be opened read-only" in errors[0].getMessage()
