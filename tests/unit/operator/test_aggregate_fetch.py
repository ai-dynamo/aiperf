# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the sweep-aggregate harvest handler.

Covers ``fetch_sweep_aggregate_to_disk`` — the operator-side helper that pulls
the sweep-controller's parent aggregate + children.json + per-strategy
confidence payload off the sweep-controller pod's results-sidecar and onto
the operator's PVC, before the JobSet (and pod) is deleted on success. The
data lives only on the sweep-controller's emptyDir, so the harvest is the
last chance to capture it.

Key invariant: the latest-pointer write is advisory. A transient PVC error
on the pointer write must NOT downgrade the count returned to the caller —
the caller uses the count to decide "did we get the artifacts?", not "did
we update the latest-pointer?".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.operator.handlers.sweep._aggregate_fetch import (
    fetch_sweep_aggregate_to_disk,
)


@pytest.mark.asyncio
async def test_fetch_returns_count_when_pointer_write_raises_oserror(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """If ``_write_sweep_latest_pointer`` raises ``OSError`` (e.g. PVC full,
    permission denied), the harvest must still return the file count — the
    pointer is only an advisory hint for ``aiperf kube sweeps``. Logging an
    advisory warning is fine; downgrading the count to 0 is not.
    """
    fake_progress_client = MagicMock()
    fake_progress_client.__aenter__ = AsyncMock(return_value=fake_progress_client)
    fake_progress_client.__aexit__ = AsyncMock(return_value=None)
    fake_progress_client.download_all_results = AsyncMock(
        return_value=[
            "aggregate.json",
            "children.json",
            "profile_export_aiperf_aggregate.json",
        ]
    )

    from aiperf.operator.handlers.sweep import _aggregate_fetch as mod

    monkeypatch.setattr(mod, "ProgressClient", lambda *a, **kw: fake_progress_client)
    monkeypatch.setattr(
        mod,
        "_write_sweep_latest_pointer",
        MagicMock(side_effect=OSError("disk full")),
    )

    with caplog.at_level("WARNING", logger=mod.logger.name):
        count = await fetch_sweep_aggregate_to_disk(
            sweep_name="sweep-conc-demo",
            namespace="aiperf-benchmarks",
            epoch="1778027124",
            base_dir=tmp_path,
        )

    assert count == 3, "fetch must report success even when pointer write fails"
    assert any(
        "latest-pointer write" in rec.message and "disk full" in rec.message
        for rec in caplog.records
    ), (
        "expected an advisory warning naming the pointer write + the OSError; "
        f"got: {[r.message for r in caplog.records]}"
    )


@pytest.mark.asyncio
async def test_fetch_returns_count_on_happy_path(tmp_path: Path, monkeypatch) -> None:
    """Straight-line happy path: the harvest returns the listed file count
    and the pointer write completes (no warning logged).
    """
    fake_progress_client = MagicMock()
    fake_progress_client.__aenter__ = AsyncMock(return_value=fake_progress_client)
    fake_progress_client.__aexit__ = AsyncMock(return_value=None)
    fake_progress_client.download_all_results = AsyncMock(
        return_value=["aggregate.json", "children.json"]
    )

    from aiperf.operator.handlers.sweep import _aggregate_fetch as mod

    monkeypatch.setattr(mod, "ProgressClient", lambda *a, **kw: fake_progress_client)

    count = await fetch_sweep_aggregate_to_disk(
        sweep_name="sweep-conc-demo",
        namespace="aiperf-benchmarks",
        epoch="1778027124",
        base_dir=tmp_path,
    )
    assert count == 2

    pointer = (
        tmp_path / "aiperf-benchmarks" / "sweeps" / "sweep-conc-demo" / "latest.txt"
    )
    assert pointer.exists()
    assert pointer.read_text() == "1778027124"


@pytest.mark.asyncio
async def test_fetch_returns_zero_when_sidecar_unreachable(
    tmp_path: Path, monkeypatch
) -> None:
    """A transport error fetching from the sweep-controller's sidecar returns
    0 (the caller retries on the next reconcile) and does NOT raise.
    """
    import aiohttp

    fake_progress_client = MagicMock()
    fake_progress_client.__aenter__ = AsyncMock(return_value=fake_progress_client)
    fake_progress_client.__aexit__ = AsyncMock(return_value=None)
    fake_progress_client.download_all_results = AsyncMock(
        side_effect=aiohttp.ClientConnectionError("sweep-controller pod gone")
    )

    from aiperf.operator.handlers.sweep import _aggregate_fetch as mod

    monkeypatch.setattr(mod, "ProgressClient", lambda *a, **kw: fake_progress_client)

    count = await fetch_sweep_aggregate_to_disk(
        sweep_name="sweep-conc-demo",
        namespace="aiperf-benchmarks",
        epoch="1778027124",
        base_dir=tmp_path,
    )
    assert count == 0


@pytest.mark.asyncio
async def test_fetch_returns_zero_when_no_files_listed(
    tmp_path: Path, monkeypatch
) -> None:
    """An empty download list (sidecar reachable but pre-marker / pre-aggregate)
    returns 0 and does NOT write the latest-pointer.
    """
    fake_progress_client = MagicMock()
    fake_progress_client.__aenter__ = AsyncMock(return_value=fake_progress_client)
    fake_progress_client.__aexit__ = AsyncMock(return_value=None)
    fake_progress_client.download_all_results = AsyncMock(return_value=[])

    pointer_writes: list[Any] = []

    from aiperf.operator.handlers.sweep import _aggregate_fetch as mod

    monkeypatch.setattr(mod, "ProgressClient", lambda *a, **kw: fake_progress_client)
    monkeypatch.setattr(
        mod,
        "_write_sweep_latest_pointer",
        lambda *a, **kw: pointer_writes.append(("called", a, kw)),
    )

    count = await fetch_sweep_aggregate_to_disk(
        sweep_name="sweep-conc-demo",
        namespace="aiperf-benchmarks",
        epoch="1778027124",
        base_dir=tmp_path,
    )
    assert count == 0
    assert pointer_writes == [], (
        "latest-pointer must NOT be written when no files were harvested"
    )
