# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``find_any_job`` epoch-specific historical lookup.

Covers the multi-epoch contract: when ``epoch=`` is supplied, the union
resolver pins the archived half to ``<base>/<ns>/<name>/<epoch>/`` instead
of the ``latest.txt`` pointer; when omitted, the legacy "latest" behavior
is preserved. An unknown epoch returns ``None`` even if other runs exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import zstandard as zstd


def _write_summary(base: Path, ns: str, name: str, epoch: str, body: dict) -> None:
    """Drop a ``profile_export_aiperf.json`` under an epoch-specific run dir."""
    d = base / ns / name / epoch
    d.mkdir(parents=True)
    (d / "profile_export_aiperf.json").write_text(json.dumps(body))


def _write_compressed_summary(
    base: Path, ns: str, name: str, epoch: str, body: dict
) -> None:
    d = base / ns / name / epoch
    d.mkdir(parents=True)
    payload = json.dumps(body).encode()
    (d / "profile_export_aiperf.json.zst").write_bytes(
        zstd.ZstdCompressor().compress(payload)
    )


@pytest.mark.asyncio
async def test_find_any_job_epoch_specific_returns_old_epoch(tmp_path: Path) -> None:
    from aiperf.operator import job_union
    from aiperf.operator.results_layout import write_latest

    _write_summary(
        tmp_path,
        "bench",
        "j1",
        "1714069323",
        {
            "status": "Succeeded",
            "request_throughput": {"avg": 100.0},
            "request_latency": {"p99": 5.0},
            "input_config": {
                "models": {"items": [{"name": "m"}]},
                "endpoint": {"urls": ["x"]},
            },
        },
    )
    _write_summary(
        tmp_path,
        "bench",
        "j1",
        "1714069400",
        {
            "status": "Succeeded",
            "request_throughput": {"avg": 200.0},
            "request_latency": {"p99": 7.0},
            "input_config": {
                "models": {"items": [{"name": "m"}]},
                "endpoint": {"urls": ["x"]},
            },
        },
    )
    write_latest(tmp_path, "bench", "j1", "1714069400")
    with patch.object(job_union, "find_aiperf_job", AsyncMock(return_value=None)):
        rec = await job_union.find_any_job(
            None,
            tmp_path,
            "bench",
            "j1",
            epoch="1714069323",
        )
    assert rec is not None
    assert rec.throughput_rps == 100.0
    assert rec.source == "archived"


@pytest.mark.asyncio
async def test_find_any_job_epoch_reads_compressed_profile_export(
    tmp_path: Path,
) -> None:
    from aiperf.operator import job_union
    from aiperf.operator.results_layout import write_latest

    _write_compressed_summary(
        tmp_path,
        "bench",
        "j1",
        "1714069323",
        {
            "status": "Succeeded",
            "request_throughput": {"avg": 123.0},
            "request_latency": {"p99": 9.0},
            "input_config": {
                "models": {"items": [{"name": "m"}]},
                "endpoint": {"urls": ["x"]},
            },
        },
    )
    write_latest(tmp_path, "bench", "j1", "1714069323")
    with patch.object(job_union, "find_aiperf_job", AsyncMock(return_value=None)):
        rec = await job_union.find_any_job(
            None,
            tmp_path,
            "bench",
            "j1",
            epoch="1714069323",
        )
    assert rec is not None
    assert rec.source == "archived"
    assert rec.throughput_rps == 123.0
    assert rec.latency_p99_ms == 9.0


@pytest.mark.asyncio
async def test_find_any_job_no_epoch_uses_latest(tmp_path: Path) -> None:
    from aiperf.operator import job_union
    from aiperf.operator.results_layout import write_latest

    _write_summary(
        tmp_path,
        "bench",
        "j1",
        "1714069323",
        {
            "status": "Succeeded",
            "request_throughput": {"avg": 100.0},
            "input_config": {
                "models": {"items": [{"name": "m"}]},
                "endpoint": {"urls": ["x"]},
            },
        },
    )
    _write_summary(
        tmp_path,
        "bench",
        "j1",
        "1714069400",
        {
            "status": "Succeeded",
            "request_throughput": {"avg": 200.0},
            "input_config": {
                "models": {"items": [{"name": "m"}]},
                "endpoint": {"urls": ["x"]},
            },
        },
    )
    write_latest(tmp_path, "bench", "j1", "1714069400")
    with patch.object(job_union, "find_aiperf_job", AsyncMock(return_value=None)):
        rec = await job_union.find_any_job(
            None,
            tmp_path,
            "bench",
            "j1",
        )
    assert rec is not None
    assert rec.throughput_rps == 200.0


@pytest.mark.asyncio
async def test_find_any_job_unknown_epoch_returns_none(tmp_path: Path) -> None:
    from aiperf.operator import job_union
    from aiperf.operator.results_layout import write_latest

    _write_summary(
        tmp_path,
        "bench",
        "j1",
        "1714069323",
        {
            "status": "Succeeded",
            "input_config": {
                "models": {"items": [{"name": "m"}]},
                "endpoint": {"urls": ["x"]},
            },
        },
    )
    write_latest(tmp_path, "bench", "j1", "1714069323")
    with patch.object(job_union, "find_aiperf_job", AsyncMock(return_value=None)):
        rec = await job_union.find_any_job(
            None,
            tmp_path,
            "bench",
            "j1",
            epoch="9999999999",
        )
    assert rec is None


@pytest.mark.asyncio
async def test_find_any_job_epoch_drops_live_half(tmp_path: Path) -> None:
    """When epoch= is given the user wants the historical record specifically.

    The live CR is dropped because it always reflects the *current* (latest)
    run — merging it into a historical record would conflate epochs.
    """
    from aiperf.kubernetes.models import AIPerfJobInfo
    from aiperf.operator import job_union
    from aiperf.operator.results_layout import write_latest

    _write_summary(
        tmp_path,
        "bench",
        "j1",
        "1714069323",
        {
            "status": "Succeeded",
            "request_throughput": {"avg": 100.0},
            "input_config": {
                "models": {"items": [{"name": "m"}]},
                "endpoint": {"urls": ["x"]},
            },
        },
    )
    write_latest(tmp_path, "bench", "j1", "1714069323")
    live = AIPerfJobInfo(
        name="j1",
        namespace="bench",
        phase="Running",
        job_id="j1",
        throughput_rps=999.0,
    )
    with patch.object(job_union, "find_aiperf_job", AsyncMock(return_value=live)):
        rec = await job_union.find_any_job(
            None,
            tmp_path,
            "bench",
            "j1",
            epoch="1714069323",
        )
    assert rec is not None
    assert rec.source == "archived"
    assert rec.throughput_rps == 100.0


@pytest.mark.asyncio
async def test_find_any_job_epoch_without_summary_returns_stub(
    tmp_path: Path,
) -> None:
    """A pinned epoch dir that exists on disk but has no profile_export
    summary (e.g. a run that failed before producing artifacts) must yield
    a stub archived ``AIPerfJobInfo`` — not ``None`` — so the run-detail
    route does not 404 on epochs that ``/epochs`` happily enumerates.
    """
    from aiperf.operator import job_union

    # No summary, just an empty epoch dir.
    (tmp_path / "bench" / "j1" / "1714069323").mkdir(parents=True)

    with patch.object(job_union, "find_aiperf_job", AsyncMock(return_value=None)):
        rec = await job_union.find_any_job(
            None,
            tmp_path,
            "bench",
            "j1",
            epoch="1714069323",
        )
    assert rec is not None
    assert rec.source == "archived"
    assert rec.phase == "Unknown"
    assert rec.throughput_rps is None
    assert rec.total_requests is None


@pytest.mark.asyncio
async def test_find_any_job_latest_pointer_without_summary_still_none(
    tmp_path: Path,
) -> None:
    """The stub fallback only applies to explicit historical epochs.

    When the caller asks for "latest" (no epoch / ``"latest"``) and the
    latest run dir has no summary yet, the live CR should still win —
    falling back to a stub here would mask in-flight runs as archived.
    """
    from aiperf.operator import job_union
    from aiperf.operator.results_layout import write_latest

    (tmp_path / "bench" / "j1" / "1714069323").mkdir(parents=True)
    write_latest(tmp_path, "bench", "j1", "1714069323")

    with patch.object(job_union, "find_aiperf_job", AsyncMock(return_value=None)):
        rec = await job_union.find_any_job(
            None,
            tmp_path,
            "bench",
            "j1",
            epoch=None,
        )
    assert rec is None
