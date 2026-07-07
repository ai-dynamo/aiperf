# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the sweep-aggregate harvest handler.

Covers ``fetch_sweep_aggregate_to_disk`` — the operator-side helper that pulls
the sweep-controller's parent aggregate + children.json + per-strategy
confidence payload off the sweep-controller pod's results-sidecar and onto
the operator's PVC, before the JobSet (and pod) is deleted on success. The
data lives only on the sweep-controller's emptyDir, so the harvest is the
last chance to capture it.

Key invariants:
- The latest-pointer write is advisory. A transient PVC error on the pointer
  write must NOT downgrade the counts returned to the caller — the caller
  uses them to decide "did we get the artifacts?", not "did we update the
  latest-pointer?".
- The result reports both ``downloaded`` and ``listed`` so the caller can
  detect a PARTIAL harvest (some advertised files never landed) and keep the
  JobSet alive for a re-harvest instead of destroying the only other copy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.operator.handlers.sweep._aggregate_fetch import (
    SweepAggregateFetchResult,
    fetch_sweep_aggregate_to_disk,
)


def _file_infos(names: list[str]) -> list[dict[str, Any]]:
    """Build ``/api/results/list``-shaped file-info dicts."""
    return [{"name": name, "size": 128} for name in names]


def _fake_sidecar(
    listed: list[str], downloaded: list[str] | BaseException
) -> MagicMock:
    """Build a ProgressClient double with list + download behaviors."""
    fake = MagicMock()
    fake.__aenter__ = AsyncMock(return_value=fake)
    fake.__aexit__ = AsyncMock(return_value=None)
    fake.get_results_list = AsyncMock(return_value=_file_infos(listed))
    if isinstance(downloaded, BaseException):
        fake.download_all_results = AsyncMock(side_effect=downloaded)
    else:
        fake.download_all_results = AsyncMock(return_value=downloaded)
    return fake


@pytest.mark.asyncio
async def test_fetch_returns_counts_when_pointer_write_raises_oserror(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """If ``_write_sweep_latest_pointer`` raises ``OSError`` (e.g. PVC full,
    permission denied), the harvest must still return the file counts — the
    pointer is only an advisory hint for ``aiperf kube sweeps``. Logging an
    advisory warning is fine; downgrading the counts is not.
    """
    files = [
        "aggregate.json",
        "children.json",
        "profile_export_aiperf_aggregate.json",
    ]
    fake_progress_client = _fake_sidecar(listed=files, downloaded=files)

    from aiperf.operator.handlers.sweep import _aggregate_fetch as mod

    monkeypatch.setattr(mod, "ProgressClient", lambda *a, **kw: fake_progress_client)
    monkeypatch.setattr(
        mod,
        "_write_sweep_latest_pointer",
        MagicMock(side_effect=OSError("disk full")),
    )

    with caplog.at_level("WARNING", logger=mod.logger.name):
        result = await fetch_sweep_aggregate_to_disk(
            sweep_name="sweep-conc-demo",
            namespace="aiperf-benchmarks",
            epoch="1778027124",
            base_dir=tmp_path,
        )

    assert result == SweepAggregateFetchResult(downloaded=3, listed=3), (
        "fetch must report success even when pointer write fails"
    )
    assert not result.is_partial
    assert any(
        "latest-pointer write" in rec.message and "disk full" in rec.message
        for rec in caplog.records
    ), (
        "expected an advisory warning naming the pointer write + the OSError; "
        f"got: {[r.message for r in caplog.records]}"
    )


@pytest.mark.asyncio
async def test_fetch_returns_counts_on_happy_path(tmp_path: Path, monkeypatch) -> None:
    """Straight-line happy path: the harvest returns the downloaded/listed
    counts and the pointer write completes (no warning logged).
    """
    files = ["aggregate.json", "children.json"]
    fake_progress_client = _fake_sidecar(listed=files, downloaded=files)

    from aiperf.operator.handlers.sweep import _aggregate_fetch as mod

    monkeypatch.setattr(mod, "ProgressClient", lambda *a, **kw: fake_progress_client)

    result = await fetch_sweep_aggregate_to_disk(
        sweep_name="sweep-conc-demo",
        namespace="aiperf-benchmarks",
        epoch="1778027124",
        base_dir=tmp_path,
    )
    assert result == SweepAggregateFetchResult(downloaded=2, listed=2)

    pointer = (
        tmp_path / "aiperf-benchmarks" / "sweeps" / "sweep-conc-demo" / "latest.txt"
    )
    assert pointer.exists()
    assert pointer.read_text() == "1778027124"


@pytest.mark.asyncio
async def test_fetch_reports_partial_when_some_downloads_fail(
    tmp_path: Path, monkeypatch
) -> None:
    """A harvest where the sidecar advertised 3 files but only 2 landed must
    surface ``downloaded < listed`` so the caller keeps the JobSet alive."""
    fake_progress_client = _fake_sidecar(
        listed=["aggregate.json", "children.json", "conditions.json"],
        downloaded=["aggregate.json", "children.json"],
    )

    from aiperf.operator.handlers.sweep import _aggregate_fetch as mod

    monkeypatch.setattr(mod, "ProgressClient", lambda *a, **kw: fake_progress_client)

    result = await fetch_sweep_aggregate_to_disk(
        sweep_name="sweep-conc-demo",
        namespace="aiperf-benchmarks",
        epoch="1778027124",
        base_dir=tmp_path,
    )

    assert result == SweepAggregateFetchResult(downloaded=2, listed=3)
    assert result.is_partial


@pytest.mark.asyncio
async def test_fetch_returns_zero_when_sidecar_unreachable(
    tmp_path: Path, monkeypatch
) -> None:
    """A transport error fetching from the sweep-controller's sidecar returns
    zero counts (the caller retries on the next reconcile) and does NOT raise.
    """
    import aiohttp

    fake_progress_client = _fake_sidecar(
        listed=["aggregate.json"],
        downloaded=aiohttp.ClientConnectionError("sweep-controller pod gone"),
    )

    from aiperf.operator.handlers.sweep import _aggregate_fetch as mod

    monkeypatch.setattr(mod, "ProgressClient", lambda *a, **kw: fake_progress_client)

    result = await fetch_sweep_aggregate_to_disk(
        sweep_name="sweep-conc-demo",
        namespace="aiperf-benchmarks",
        epoch="1778027124",
        base_dir=tmp_path,
    )
    assert result == SweepAggregateFetchResult(downloaded=0, listed=0)


@pytest.mark.asyncio
async def test_fetch_returns_zero_when_no_files_listed(
    tmp_path: Path, monkeypatch
) -> None:
    """An empty listing (sidecar reachable but pre-marker / pre-aggregate)
    returns zero counts, skips the download, and does NOT write the
    latest-pointer.
    """
    fake_progress_client = _fake_sidecar(listed=[], downloaded=[])

    pointer_writes: list[Any] = []

    from aiperf.operator.handlers.sweep import _aggregate_fetch as mod

    monkeypatch.setattr(mod, "ProgressClient", lambda *a, **kw: fake_progress_client)
    monkeypatch.setattr(
        mod,
        "_write_sweep_latest_pointer",
        lambda *a, **kw: pointer_writes.append(("called", a, kw)),
    )

    result = await fetch_sweep_aggregate_to_disk(
        sweep_name="sweep-conc-demo",
        namespace="aiperf-benchmarks",
        epoch="1778027124",
        base_dir=tmp_path,
    )
    assert result == SweepAggregateFetchResult(downloaded=0, listed=0)
    fake_progress_client.download_all_results.assert_not_awaited()
    assert pointer_writes == [], (
        "latest-pointer must NOT be written when no files were harvested"
    )
