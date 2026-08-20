# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component-integration tests for runs_index lazy fallback + handler wiring."""

from __future__ import annotations

import asyncio
from pathlib import Path

import orjson
import pytest
from aiperf.operator.results_layout import list_runs

from aiperf.operator import runs_index


@pytest.fixture
async def open_index(tmp_path: Path):
    db = tmp_path / ".aiperf_index.sqlite"
    await runs_index.open(db)
    yield tmp_path
    await runs_index.close()


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_list_runs_falls_back_to_disk_when_index_empty(
    open_index: Path,
) -> None:
    base = open_index
    run = base / "ns" / "j" / "1714069323"
    run.mkdir(parents=True)
    (run / "profile_export_aiperf.json").write_bytes(orjson.dumps({}))
    (run / ".aiperf_results_ready.json").write_text("{}")
    (base / "ns" / "j" / "latest.txt").write_text("1714069323")

    # Index is empty; list_runs must fall back to disk.
    rows = list_runs(base, "ns", "j")
    assert len(rows) == 1
    assert rows[0].epoch == "1714069323"

    # Within ~1s the lazy backfill must populate the index.
    for _ in range(20):
        await asyncio.sleep(0.05)
        if await runs_index.get_run("ns", "j", "1714069323") is not None:
            break
    assert await runs_index.get_run("ns", "j", "1714069323") is not None
