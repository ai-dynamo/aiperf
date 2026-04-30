# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for runs_index.py — the SQLite-backed runs + sweep variation index."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.operator import runs_index


@pytest.fixture
async def index_path(tmp_path: Path) -> Path:
    """Open a fresh runs_index DB rooted at tmp_path; close on teardown."""
    path = tmp_path / ".aiperf_index.sqlite"
    await runs_index.open(path)
    yield path
    await runs_index.close()


@pytest.mark.asyncio
async def test_open_creates_schema_idempotently(tmp_path: Path) -> None:
    path = tmp_path / ".aiperf_index.sqlite"
    await runs_index.open(path)
    await runs_index.close()
    assert path.exists()

    # Re-opening must not raise and must not duplicate rows in meta
    await runs_index.open(path)
    schema_version = await runs_index.get_meta("schema_version")
    assert schema_version == "1"
    await runs_index.close()


@pytest.mark.asyncio
async def test_integrity_check_detects_corruption(tmp_path: Path) -> None:
    path = tmp_path / ".aiperf_index.sqlite"
    await runs_index.open(path)
    await runs_index.close()

    # Stomp the file with garbage
    path.write_bytes(b"not a sqlite db")
    ok = await runs_index.integrity_check(path)
    assert ok is False
