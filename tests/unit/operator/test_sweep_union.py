# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_DEFAULT_EPOCH = "1714069323"


def _write_aggregate(
    base: Path, ns: str, name: str, body: dict, *, epoch: str = _DEFAULT_EPOCH
) -> Path:
    from aiperf.operator.results_layout import write_sweep_latest

    d = base / ns / "sweeps" / name / epoch
    d.mkdir(parents=True)
    (d / "aggregate.json").write_text(json.dumps(body))
    write_sweep_latest(base, ns, name, epoch)
    return d


def _live_cr(
    ns: str,
    name: str,
    *,
    phase: str = "Running",
    total: int = 4,
    completed: int = 1,
    failed: int = 0,
    model: str = "m",
    creation: str = "2026-04-01T00:00:00Z",
) -> dict:
    return {
        "metadata": {"namespace": ns, "name": name, "creationTimestamp": creation},
        "spec": {
            "template": {"spec": {"models": [{"name": model}]}},
            "sweep": {
                "type": "grid",
                "axes": [{"name": "concurrency", "values": [1, 2, 4, 8]}],
            },
        },
        "status": {
            "phase": phase,
            "totalVariations": total,
            "completedRuns": completed,
            "failedRuns": failed,
        },
    }


@pytest.mark.asyncio
async def test_list_all_sweeps_live_only(tmp_path: Path) -> None:
    from aiperf.operator import sweep_union

    api = MagicMock()
    with patch.object(
        sweep_union,
        "list_aiperfsweeps",
        AsyncMock(return_value=[_live_cr("bench", "s1")]),
    ):
        records = await sweep_union.list_all_sweeps(api, tmp_path, all_namespaces=True)
    assert len(records) == 1
    r = records[0]
    assert r.source == "live"
    assert r.phase == "Running"
    assert r.total_variations == 4
    assert r.aggregate_path is None


@pytest.mark.asyncio
async def test_list_all_sweeps_archived_only(tmp_path: Path) -> None:
    from aiperf.operator import sweep_union

    _write_aggregate(
        tmp_path,
        "bench",
        "s1",
        {
            "phase": "Succeeded",
            "totalVariations": 4,
            "completedRuns": 4,
            "failedRuns": 0,
            "completedAt": "2026-04-25T01:00:00Z",
            "specSummary": {
                "sweep_type": "grid",
                "dimensions": [{"name": "concurrency", "values": [1, 2, 4, 8]}],
            },
            "model": "m",
        },
    )
    api = MagicMock()
    with patch(
        "aiperf.operator.sweep_union.list_aiperfsweeps",
        AsyncMock(return_value=[]),
    ):
        records = await sweep_union.list_all_sweeps(api, tmp_path, all_namespaces=True)
    assert len(records) == 1
    r = records[0]
    assert r.source == "archived"
    assert r.phase == "Succeeded"
    assert r.total_variations == 4
    assert r.aggregate_path is not None


@pytest.mark.asyncio
async def test_list_all_sweeps_archived_records_use_epoch_dir_when_timestamps_missing(
    tmp_path: Path,
) -> None:
    from aiperf.operator import sweep_union

    _write_aggregate(
        tmp_path,
        "bench",
        "old-sweep",
        {
            "phase": "Succeeded",
            "totalVariations": 1,
            "completedRuns": 1,
            "failedRuns": 0,
        },
        epoch="1714069100",
    )
    _write_aggregate(
        tmp_path,
        "bench",
        "new-sweep",
        {
            "phase": "Succeeded",
            "totalVariations": 1,
            "completedRuns": 1,
            "failedRuns": 0,
        },
        epoch="1714069200",
    )
    api = MagicMock()
    with (
        patch(
            "aiperf.operator.sweep_union.list_aiperfsweeps",
            AsyncMock(return_value=[]),
        ),
        patch.object(
            sweep_union,
            "_now_utc",
            return_value=sweep_union.datetime.fromtimestamp(
                1714069300, tz=sweep_union.UTC
            ),
        ),
    ):
        records = await sweep_union.list_all_sweeps(api, tmp_path, all_namespaces=True)

    assert [r.name for r in records] == ["new-sweep", "old-sweep"]
    assert {r.name: r.age_seconds for r in records} == {
        "new-sweep": 100,
        "old-sweep": 200,
    }


@pytest.mark.asyncio
async def test_list_all_sweeps_both(tmp_path: Path) -> None:
    from aiperf.operator import sweep_union

    _write_aggregate(
        tmp_path,
        "bench",
        "s1",
        {
            "phase": "Succeeded",
            "totalVariations": 4,
            "completedRuns": 4,
            "failedRuns": 0,
            "completedAt": "2026-04-25T01:00:00Z",
            "specSummary": {
                "sweep_type": "grid",
                "dimensions": [{"name": "concurrency", "values": [1, 2, 4, 8]}],
            },
            "model": "m",
        },
    )
    api = MagicMock()
    with patch(
        "aiperf.operator.sweep_union.list_aiperfsweeps",
        AsyncMock(
            return_value=[
                _live_cr("bench", "s1", phase="Aggregating", completed=4, total=4)
            ]
        ),
    ):
        records = await sweep_union.list_all_sweeps(api, tmp_path, all_namespaces=True)
    assert len(records) == 1
    r = records[0]
    assert r.source == "both"
    # Live phase wins on overlap.
    assert r.phase == "Aggregating"
    assert r.aggregate_path is not None


@pytest.mark.asyncio
async def test_find_any_sweep_archived_corrupt_aggregate(tmp_path: Path) -> None:
    from aiperf.operator.results_layout import write_sweep_latest

    from aiperf.operator import sweep_union

    d = tmp_path / "bench" / "sweeps" / "s1" / _DEFAULT_EPOCH
    d.mkdir(parents=True)
    (d / "aggregate.json").write_text("not json")
    write_sweep_latest(tmp_path, "bench", "s1", _DEFAULT_EPOCH)
    api = MagicMock()
    with patch(
        "aiperf.operator.sweep_union.find_aiperfsweep",
        AsyncMock(return_value=None),
    ):
        rec = await sweep_union.find_any_sweep(api, tmp_path, "bench", "s1")
    # Corrupt aggregate still surfaces a record so the list page is not blank;
    # phase is Unknown to mark the broken state.
    assert rec is not None
    assert rec.phase == "Unknown"
    assert rec.source == "archived"


@pytest.mark.asyncio
async def test_find_any_sweep_neither_returns_none(tmp_path: Path) -> None:
    from aiperf.operator import sweep_union

    api = MagicMock()
    with patch(
        "aiperf.operator.sweep_union.find_aiperfsweep",
        AsyncMock(return_value=None),
    ):
        rec = await sweep_union.find_any_sweep(api, tmp_path, "bench", "s1")
    assert rec is None


def test_synthesize_status_from_aggregate_terminal() -> None:
    from aiperf.operator.sweep_union import synthesize_sweep_status_from_aggregate

    out = synthesize_sweep_status_from_aggregate(
        "bench",
        "s1",
        {
            "phase": "Succeeded",
            "totalVariations": 4,
            "completedRuns": 4,
            "failedRuns": 0,
            "completedAt": "2026-04-25T01:00:00Z",
        },
        conditions=[{"type": "Done", "status": "True"}],
    )
    assert out["phase"] == "Succeeded"
    assert out["totalVariations"] == 4
    assert out["completedRuns"] == 4
    assert out["completedAt"] == "2026-04-25T01:00:00Z"
    assert out["conditions"] == [{"type": "Done", "status": "True"}]
