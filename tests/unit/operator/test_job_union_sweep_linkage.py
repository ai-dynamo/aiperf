# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sweep linkage propagation in the unified-jobs view.

Live sweep children carry ``aiperf.nvidia.com/{sweep,variation-index,variation-label}``
labels on the AIPerfJob CR; ``AIPerfJobCR.to_info()`` lifts those into the flat
``AIPerfJobInfo``. Archived sweep children read the same linkage from the
``sweep.json`` marker the sweep-controller drops at child-create time, so the
parent pointer survives CR TTL reaping. Standalone (non-sweep) jobs keep all
three fields as ``None`` on both planes.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import orjson
import pytest

from aiperf.kubernetes.models import AIPerfJobCR, AIPerfJobInfo
from aiperf.operator import job_union
from aiperf.operator.results_layout import write_latest

_TEST_EPOCH = "1714064523"


def _write_summary(base: Path, ns: str, name: str) -> None:
    """Write a minimal ``profile_export_aiperf.json`` for an archived job."""
    d = base / ns / name / _TEST_EPOCH
    d.mkdir(parents=True, exist_ok=True)
    body = {
        "status": "Succeeded",
        "input_config": {
            "models": {"items": [{"name": "m"}]},
            "endpoint": {"urls": ["http://x"]},
        },
        "request_throughput": {"avg": 100.0},
        "request_latency": {"p99": 5.0},
    }
    (d / "profile_export_aiperf.json").write_bytes(orjson.dumps(body))
    write_latest(base, ns, name, _TEST_EPOCH)


def _write_sweep_marker(
    base: Path,
    ns: str,
    name: str,
    *,
    sweep_name: str,
    variation_index: int,
    variation_label: str,
) -> None:
    """Drop a ``sweep.json`` marker into ``<base>/<ns>/<name>/``."""
    d = base / ns / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "sweep.json").write_bytes(
        orjson.dumps(
            {
                "sweep_name": sweep_name,
                "variation_index": variation_index,
                "variation_label": variation_label,
                "trial_index": 0,
            }
        )
    )


def test_to_info_extracts_sweep_linkage_from_labels():
    """Live CR with sweep labels lifts them onto the flat AIPerfJobInfo."""
    raw = {
        "metadata": {
            "namespace": "bench",
            "name": "ch-0-0",
            "labels": {
                "aiperf.nvidia.com/sweep": "saturation-sweep",
                "aiperf.nvidia.com/variation-index": "7",
                "aiperf.nvidia.com/variation-label": "concurrency-128-rate-50",
            },
            "creationTimestamp": "2026-04-25T00:00:00Z",
        },
        "status": {"phase": "Running"},
        "spec": {"benchmark": {"models": [{"name": "m"}]}},
    }
    info = AIPerfJobCR.model_validate(raw).to_info()
    assert info.sweep_name == "saturation-sweep"
    assert info.variation_index == 7
    assert info.variation_label == "concurrency-128-rate-50"


def test_to_info_extracts_trial_index_from_label():
    """Multi-trial sweep children carry the trial-index label onto the info."""
    raw = {
        "metadata": {
            "namespace": "bench",
            "name": "satsweep-v07-t2",
            "labels": {
                "aiperf.nvidia.com/sweep": "satsweep",
                "aiperf.nvidia.com/variation-index": "07",
                "aiperf.nvidia.com/variation-label": "concurrency-128",
                "aiperf.nvidia.com/trial-index": "2",
            },
            "creationTimestamp": "2026-04-25T00:00:00Z",
        },
        "status": {"phase": "Running"},
        "spec": {"benchmark": {"models": [{"name": "m"}]}},
    }
    info = AIPerfJobCR.model_validate(raw).to_info()
    assert info.trial_index == 2
    # The operator UI reads job.trialIndex, so the camelCase alias must hold.
    assert info.model_dump(by_alias=True)["trialIndex"] == 2


def test_to_info_no_labels_returns_none_linkage():
    """Standalone job with no sweep labels keeps all linkage fields as None."""
    raw = {
        "metadata": {
            "namespace": "bench",
            "name": "one-shot",
            "creationTimestamp": "2026-04-25T00:00:00Z",
        },
        "status": {"phase": "Running"},
        "spec": {"benchmark": {"models": [{"name": "m"}]}},
    }
    info = AIPerfJobCR.model_validate(raw).to_info()
    assert info.sweep_name is None
    assert info.variation_index is None
    assert info.variation_label is None
    assert info.trial_index is None


def test_to_info_invalid_variation_index_falls_back_to_none():
    """Non-integer variation-index / trial-index labels tolerated as None."""
    raw = {
        "metadata": {
            "namespace": "bench",
            "name": "ch-x",
            "labels": {
                "aiperf.nvidia.com/sweep": "s",
                "aiperf.nvidia.com/variation-index": "not-an-int",
                "aiperf.nvidia.com/trial-index": "not-an-int",
            },
            "creationTimestamp": "2026-04-25T00:00:00Z",
        },
        "status": {"phase": "Running"},
        "spec": {"benchmark": {"models": [{"name": "m"}]}},
    }
    info = AIPerfJobCR.model_validate(raw).to_info()
    assert info.sweep_name == "s"
    assert info.variation_index is None
    assert info.trial_index is None


@pytest.mark.asyncio
async def test_list_all_jobs_live_child_carries_sweep_labels(tmp_path: Path) -> None:
    """A live CR-only sweep child surfaces its labels through list_all_jobs."""
    live = AIPerfJobInfo(
        name="ch-0-0",
        namespace="bench",
        phase="Running",
        job_id="ch-0-0",
        sweep_name="saturation-sweep",
        variation_index=7,
        variation_label="concurrency-128-rate-50",
    )
    with patch.object(job_union, "list_aiperf_jobs", AsyncMock(return_value=[live])):
        results = await job_union.list_all_jobs(
            api=None, results_dir=tmp_path, all_namespaces=True
        )
    matches = [r for r in results if r.name == "ch-0-0"]
    assert len(matches) == 1
    j = matches[0]
    assert j.sweep_name == "saturation-sweep"
    assert j.variation_index == 7
    assert j.variation_label == "concurrency-128-rate-50"


@pytest.mark.asyncio
async def test_list_all_jobs_archived_child_reads_sweep_marker(
    tmp_path: Path,
) -> None:
    """An archived (CR-gone) sweep child reads linkage from sweep.json."""
    _write_summary(tmp_path, "bench", "ch-0-0")
    _write_sweep_marker(
        tmp_path,
        "bench",
        "ch-0-0",
        sweep_name="saturation-sweep",
        variation_index=7,
        variation_label="concurrency-128-rate-50",
    )
    with patch.object(job_union, "list_aiperf_jobs", AsyncMock(return_value=[])):
        results = await job_union.list_all_jobs(
            api=None, results_dir=tmp_path, all_namespaces=True
        )
    matches = [r for r in results if r.name == "ch-0-0"]
    assert len(matches) == 1
    j = matches[0]
    assert j.source == "archived"
    assert j.sweep_name == "saturation-sweep"
    assert j.variation_index == 7
    assert j.variation_label == "concurrency-128-rate-50"


@pytest.mark.asyncio
async def test_list_all_jobs_no_sweep_linkage_returns_none(tmp_path: Path) -> None:
    """A standalone (non-sweep) job leaves all three linkage fields as None."""
    live = AIPerfJobInfo(
        name="one-shot",
        namespace="bench",
        phase="Running",
        job_id="one-shot",
    )
    with patch.object(job_union, "list_aiperf_jobs", AsyncMock(return_value=[live])):
        results = await job_union.list_all_jobs(
            api=None, results_dir=tmp_path, all_namespaces=True
        )
    matches = [r for r in results if r.name == "one-shot"]
    assert len(matches) == 1
    assert matches[0].sweep_name is None
    assert matches[0].variation_index is None
    assert matches[0].variation_label is None


@pytest.mark.asyncio
async def test_list_all_jobs_archived_without_marker_keeps_none(
    tmp_path: Path,
) -> None:
    """Archived job without a sweep.json marker keeps linkage None (not raised)."""
    _write_summary(tmp_path, "bench", "legacy-job")
    with patch.object(job_union, "list_aiperf_jobs", AsyncMock(return_value=[])):
        results = await job_union.list_all_jobs(
            api=None, results_dir=tmp_path, all_namespaces=True
        )
    matches = [r for r in results if r.name == "legacy-job"]
    assert len(matches) == 1
    assert matches[0].sweep_name is None
    assert matches[0].variation_index is None
    assert matches[0].variation_label is None


@pytest.mark.asyncio
async def test_list_all_jobs_overlap_lets_live_linkage_win(tmp_path: Path) -> None:
    """When live + archived overlap, live linkage wins; archived fills only gaps."""
    live = AIPerfJobInfo(
        name="ch-0-0",
        namespace="bench",
        phase="Running",
        job_id="ch-0-0",
        sweep_name="live-sweep",
        variation_index=42,
        variation_label="live-label",
    )
    _write_summary(tmp_path, "bench", "ch-0-0")
    _write_sweep_marker(
        tmp_path,
        "bench",
        "ch-0-0",
        sweep_name="archived-sweep",
        variation_index=7,
        variation_label="archived-label",
    )
    with patch.object(job_union, "list_aiperf_jobs", AsyncMock(return_value=[live])):
        results = await job_union.list_all_jobs(
            api=None, results_dir=tmp_path, all_namespaces=True
        )
    j = next(r for r in results if r.name == "ch-0-0")
    assert j.source == "both"
    assert j.sweep_name == "live-sweep"
    assert j.variation_index == 42
    assert j.variation_label == "live-label"


@pytest.mark.asyncio
async def test_list_all_jobs_overlap_archived_fills_when_live_missing(
    tmp_path: Path,
) -> None:
    """Live without linkage + archived with linkage: archived backfills."""
    live = AIPerfJobInfo(
        name="ch-0-0",
        namespace="bench",
        phase="Running",
        job_id="ch-0-0",
    )
    _write_summary(tmp_path, "bench", "ch-0-0")
    _write_sweep_marker(
        tmp_path,
        "bench",
        "ch-0-0",
        sweep_name="archived-sweep",
        variation_index=7,
        variation_label="archived-label",
    )
    with patch.object(job_union, "list_aiperf_jobs", AsyncMock(return_value=[live])):
        results = await job_union.list_all_jobs(
            api=None, results_dir=tmp_path, all_namespaces=True
        )
    j = next(r for r in results if r.name == "ch-0-0")
    assert j.source == "both"
    assert j.sweep_name == "archived-sweep"
    assert j.variation_index == 7
    assert j.variation_label == "archived-label"


@pytest.mark.asyncio
async def test_find_any_job_archived_reads_sweep_marker(tmp_path: Path) -> None:
    """find_any_job's archived path also reads sweep.json from the name_dir."""
    _write_summary(tmp_path, "bench", "ch-0-0")
    _write_sweep_marker(
        tmp_path,
        "bench",
        "ch-0-0",
        sweep_name="saturation-sweep",
        variation_index=7,
        variation_label="concurrency-128-rate-50",
    )
    with patch.object(job_union, "find_aiperf_job", AsyncMock(return_value=None)):
        info = await job_union.find_any_job(
            api=None, results_dir=tmp_path, namespace="bench", name="ch-0-0"
        )
    assert info is not None
    assert info.source == "archived"
    assert info.sweep_name == "saturation-sweep"
    assert info.variation_index == 7
    assert info.variation_label == "concurrency-128-rate-50"


def test_sweep_linkage_from_marker_unreadable_returns_none(tmp_path: Path) -> None:
    """Corrupt sweep.json is logged + tolerated as no-linkage."""
    job_dir = tmp_path / "bench" / "ch-bad"
    job_dir.mkdir(parents=True)
    (job_dir / "sweep.json").write_bytes(b"{not valid json")
    sweep_name, variation_index, variation_label = job_union._sweep_linkage_from_marker(
        job_dir
    )
    assert sweep_name is None
    assert variation_index is None
    assert variation_label is None
