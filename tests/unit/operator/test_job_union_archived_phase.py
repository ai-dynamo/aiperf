# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Archived jobs recover their terminal phase from the durable runs index."""

from pathlib import Path
from typing import Any

import orjson
import pytest

from aiperf.operator import job_union
from aiperf.operator.results_layout import write_latest
from aiperf.operator.runs_index_models import RunIndexRow

_EPOCH = "1714064523"
_NS = "bench"
_NAME = "failed-child"


def _row(*, phase: str, error: str | None = None) -> RunIndexRow:
    return RunIndexRow(
        namespace=_NS,
        job_id=_NAME,
        epoch=_EPOCH,
        phase=phase,
        is_latest=True,
        start_time=None,
        end_time=None,
        created_unix=0,
        mtime_epoch=0,
        error=error,
        model=None,
        endpoint=None,
        gpu_count=0,
        gpu_name=None,
        file_count=0,
        total_size_bytes=0,
        sweep_namespace=None,
        sweep_name=None,
        sweep_epoch=None,
        sweep_variation_idx=None,
    )


def _write_summary(base: Path) -> None:
    run = base / _NS / _NAME / _EPOCH
    run.mkdir(parents=True)
    (run / "profile_export_aiperf.json").write_bytes(
        orjson.dumps(
            {
                "start_time": "2026-08-01T10:00:00Z",
                "end_time": "2026-08-01T10:45:00Z",
                "request_throughput": {"avg": 42.1},
            }
        )
    )
    write_latest(base, _NS, _NAME, _EPOCH)


def _stub_index(
    monkeypatch: pytest.MonkeyPatch,
    row: RunIndexRow | None,
) -> None:
    async def _list_all_latest() -> list[RunIndexRow]:
        return [row] if row is not None else []

    async def _get_latest_run(_namespace: str, _job_id: str) -> RunIndexRow | None:
        return row

    monkeypatch.setattr(job_union.runs_index, "is_open", lambda: True)
    monkeypatch.setattr(job_union.runs_index, "list_all_latest", _list_all_latest)
    monkeypatch.setattr(job_union.runs_index, "get_latest_run", _get_latest_run)


async def _no_live_jobs(_api: Any, **_: Any) -> list[Any]:
    return []


@pytest.mark.asyncio
async def test_archived_failure_uses_indexed_phase_and_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(job_union, "list_aiperf_jobs", _no_live_jobs)
    _write_summary(tmp_path)
    _stub_index(monkeypatch, _row(phase="Failed", error="worker crashed"))

    [job] = await job_union.list_all_jobs(None, tmp_path)

    assert job.phase == "Failed"
    assert job.error == "worker crashed"


@pytest.mark.asyncio
async def test_missing_index_phase_remains_neutral(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(job_union, "list_aiperf_jobs", _no_live_jobs)
    _write_summary(tmp_path)
    _stub_index(monkeypatch, None)

    [job] = await job_union.list_all_jobs(None, tmp_path)

    assert job.phase == "Archived"
