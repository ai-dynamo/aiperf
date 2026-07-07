# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for latest.txt pointer update ordering.

Focuses on:
- atomic job-run and sweep latest-pointer writes preserving the previous pointer
- bad epoch inputs rejected before they can poison latest.txt
- stale older epochs not clobbering newer latest pointers
- job completion keeping disk latest.txt and runs_index.is_latest consistent
- partial job/sweep result bundles not advancing the latest pointer

Out of scope (covered elsewhere):
- retention policy and run enumeration: tests/unit/operator/test_results_layout.py
- runs_index row upsert internals: tests/unit/operator/test_runs_index.py
- sweep variation payload parsing: tests/unit/operator/test_runs_index_adversarial.py
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from pytest import param

from aiperf.operator import results_layout, runs_index
from aiperf.operator.handlers import completion
from aiperf.operator.handlers.sweep import _aggregate_fetch
from aiperf.operator.models import ControllerFetchResult
from aiperf.operator.results_layout import LATEST_POINTER, epoch_key_from_body
from aiperf.operator.status import StatusBuilder

# ============================================================================
# Helpers
# ============================================================================


_NAMESPACE = "bench-prod"
_JOB_ID = "llama-results-7f2a"
_SWEEP_NAME = "latency-sweep-5d1c"
_EPOCH_OLD = "1779105601"
_EPOCH_NEW = "1779109201"


@dataclass(slots=True)
class _Patch:
    """Minimal kopf.Patch stand-in carrying only the mutable status mapping."""

    status: dict[str, object] = field(default_factory=dict)


def _job_body(epoch_timestamp: str = "2026-05-18T12:00:01Z") -> dict[str, object]:
    """Build a realistic AIPerfJob body with a deterministic epoch key."""
    return {
        "metadata": {
            "name": _JOB_ID,
            "namespace": _NAMESPACE,
            "creationTimestamp": epoch_timestamp,
        },
        "spec": {
            "benchmark": {
                "models": {"items": [{"name": "meta-llama/Llama-3-8B"}]},
                "endpoint": {"urls": ["http://inference.local:8000"]},
            }
        },
    }


def _metrics_payload(throughput: float = 212.5) -> dict[str, object]:
    """Return a controller /api/metrics-shaped payload with one compare metric."""
    return {
        "metrics": {
            "request_throughput": {
                "avg": throughput,
                "p50": throughput - 5.0,
                "p99": throughput + 10.0,
                "unit": "rps",
            }
        },
        "end_time": "2026-05-18T12:03:21Z",
    }


def _write_profile_export(
    run_path: Path, payload: dict[str, object] | None = None
) -> None:
    """Materialize the authoritative profile export for a completed run."""
    run_path.mkdir(parents=True)
    (run_path / "profile_export_aiperf.json").write_bytes(
        orjson.dumps(payload or _metrics_payload())
    )


@pytest.fixture
async def opened_index(tmp_path: Path) -> AsyncGenerator[Path, None]:
    """Open a fresh writable runs_index DB and close it after the test."""
    path = tmp_path / ".aiperf_index.sqlite"
    await runs_index.open(path)
    try:
        yield path
    finally:
        await runs_index.close()


@pytest.fixture
def completion_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point completion-side disk writes at tmp_path and neutralize external effects."""
    base = tmp_path / "results"
    monkeypatch.setattr(completion.OperatorEnvironment.RESULTS, "DIR", base)
    monkeypatch.setattr(completion, "_delete_backing_jobset", AsyncMock())
    monkeypatch.setattr(completion.events, "completed", MagicMock())
    monkeypatch.setattr(completion.events, "results_stored", MagicMock())
    monkeypatch.setattr(completion.events, "results_failed", MagicMock())
    monkeypatch.setattr(completion.events, "index_update_failed", MagicMock())
    return base


def _status_builder() -> tuple[_Patch, StatusBuilder]:
    """Create a StatusBuilder backed by a minimal mutable patch object."""
    patch = _Patch()
    return patch, StatusBuilder(patch, existing_status={})


# ============================================================================
# Atomicity and bad epoch validation
# ============================================================================


def test_write_latest_replace_failure_preserves_existing_job_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Atomic replace failure must leave readers on the previous complete run."""
    base = tmp_path / "results"
    results_layout.write_latest(base, _NAMESPACE, _JOB_ID, _EPOCH_OLD)

    def fail_replace(src: Path | str, dst: Path | str) -> None:
        if os.fspath(dst).endswith(LATEST_POINTER):
            raise OSError("simulated PVC rename failure for latest.txt")
        os.replace(src, dst)

    monkeypatch.setattr(results_layout.os, "replace", fail_replace)

    with pytest.raises(OSError, match=r"PVC rename failure.*latest\.txt"):
        results_layout.write_latest(base, _NAMESPACE, _JOB_ID, _EPOCH_NEW)

    assert results_layout.resolve_latest(base, _NAMESPACE, _JOB_ID) == _EPOCH_OLD
    assert (
        results_layout.job_dir(base, _NAMESPACE, _JOB_ID) / f"{LATEST_POINTER}.tmp"
    ).read_text() == _EPOCH_NEW


def test_write_sweep_latest_replace_failure_preserves_existing_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sweep aggregate pointer replacement must be atomic for concurrent readers."""
    base = tmp_path / "results"
    _aggregate_fetch._write_sweep_latest_pointer(
        base, _NAMESPACE, _SWEEP_NAME, _EPOCH_OLD
    )
    original_replace = Path.replace

    def fail_sweep_replace(path: Path, target: Path | str) -> Path:
        if path.name == "latest.tmp" and os.fspath(target).endswith(LATEST_POINTER):
            raise OSError("simulated sweep latest rename failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_sweep_replace)

    with pytest.raises(OSError, match=r"sweep latest rename failure"):
        _aggregate_fetch._write_sweep_latest_pointer(
            base, _NAMESPACE, _SWEEP_NAME, _EPOCH_NEW
        )

    assert (
        results_layout.resolve_sweep_latest(base, _NAMESPACE, _SWEEP_NAME) == _EPOCH_OLD
    )
    assert (
        base / _NAMESPACE / "sweeps" / _SWEEP_NAME / "latest.tmp"
    ).read_text() == _EPOCH_NEW


@pytest.mark.parametrize(
    "bad_epoch",
    [
        param("latest", id="symbolic-latest-rejected"),
        param("../escaped-results-9d2c", id="path-traversal-rejected"),
        param("171606", id="too-short-epoch-rejected"),
        param("171606100112345678901", id="too-long-epoch-rejected"),
    ],
)  # fmt: skip
def test_write_latest_bad_epoch_rejected_before_pointer_update(
    tmp_path: Path, bad_epoch: str
) -> None:
    """Bad job epochs must fail before latest.txt can store unresolvable garbage."""
    base = tmp_path / "results"
    results_layout.write_latest(base, _NAMESPACE, _JOB_ID, _EPOCH_OLD)

    with pytest.raises(ValueError, match=rf"epoch.*{bad_epoch!r}"):
        results_layout.write_latest(base, _NAMESPACE, _JOB_ID, bad_epoch)

    assert results_layout.resolve_latest(base, _NAMESPACE, _JOB_ID) == _EPOCH_OLD


@pytest.mark.parametrize(
    "bad_epoch",
    [
        param("latest", id="symbolic-latest-rejected"),
        param("../../job-results", id="path-traversal-rejected"),
        param("171606", id="too-short-epoch-rejected"),
        param("171606100112345678901", id="too-long-epoch-rejected"),
    ],
)  # fmt: skip
def test_write_sweep_latest_bad_epoch_rejected_before_pointer_update(
    tmp_path: Path, bad_epoch: str
) -> None:
    """Bad sweep epochs must fail before latest.txt can point at junk."""
    base = tmp_path / "results"
    results_layout.write_sweep_latest(base, _NAMESPACE, _SWEEP_NAME, _EPOCH_OLD)

    with pytest.raises(ValueError, match=rf"epoch.*{bad_epoch!r}"):
        results_layout.write_sweep_latest(base, _NAMESPACE, _SWEEP_NAME, bad_epoch)

    assert (
        results_layout.resolve_sweep_latest(base, _NAMESPACE, _SWEEP_NAME) == _EPOCH_OLD
    )


# ============================================================================
# Ordering and disk/index consistency
# ============================================================================


def test_write_latest_stale_older_epoch_does_not_clobber_newer_pointer(
    tmp_path: Path,
) -> None:
    """A delayed older completion must not roll latest.txt back from a newer epoch."""
    base = tmp_path / "results"
    results_layout.write_latest(base, _NAMESPACE, _JOB_ID, _EPOCH_NEW)

    results_layout.write_latest(base, _NAMESPACE, _JOB_ID, _EPOCH_OLD)

    assert results_layout.resolve_latest(base, _NAMESPACE, _JOB_ID) == _EPOCH_NEW


@pytest.mark.asyncio
async def test_handle_completion_completed_run_keeps_disk_and_index_latest_consistent(
    completion_harness: Path,
    opened_index: Path,
) -> None:
    """Successful completion must advance latest.txt and runs_index to the same epoch."""
    body = _job_body()
    epoch = epoch_key_from_body(body)
    run_path = results_layout.run_dir(completion_harness, _NAMESPACE, _JOB_ID, epoch)
    _write_profile_export(run_path)
    patch, sb = _status_builder()

    await completion.handle_completion(
        body,
        _NAMESPACE,
        "aiperf-llama-results-7f2a",
        _JOB_ID,
        status={},
        sb=sb,
        result=ControllerFetchResult(
            metrics=_metrics_payload(),
            downloaded=["profile_export_aiperf.json"],
        ),
    )

    assert (
        results_layout.resolve_latest(completion_harness, _NAMESPACE, _JOB_ID) == epoch
    )
    row = await runs_index.get_latest_run(_NAMESPACE, _JOB_ID)
    assert row is not None
    assert row.epoch == epoch
    assert patch.status["runEpoch"] == int(epoch)


# ============================================================================
# Partial bundle guards
# ============================================================================


@pytest.mark.asyncio
async def test_handle_completion_missing_materialized_key_file_does_not_update_latest(
    completion_harness: Path,
    opened_index: Path,
) -> None:
    """A downloaded-list key file without the on-disk export must not become latest."""
    body = _job_body()
    epoch = epoch_key_from_body(body)
    patch, sb = _status_builder()

    await completion.handle_completion(
        body,
        _NAMESPACE,
        "aiperf-llama-results-7f2a",
        _JOB_ID,
        status={},
        sb=sb,
        result=ControllerFetchResult(
            metrics=_metrics_payload(),
            downloaded=["profile_export_aiperf.json"],
        ),
    )

    assert (
        results_layout.resolve_latest(completion_harness, _NAMESPACE, _JOB_ID) is None
    )
    assert (
        results_layout.resolve_run_dir(completion_harness, _NAMESPACE, _JOB_ID) is None
    )
    assert await runs_index.get_latest_run(_NAMESPACE, _JOB_ID) is None
    assert "runEpoch" not in patch.status
    assert not (completion_harness / _NAMESPACE / _JOB_ID / epoch).exists()


@pytest.mark.asyncio
async def test_fetch_sweep_aggregate_empty_download_list_does_not_update_latest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty sidecar listing must not advance the sweep latest pointer.

    ``fetch_sweep_aggregate_to_disk`` derives its harvest count from the
    sidecar's reported file list (the sidecar's ``download_all_results``
    materializes the bytes onto the PVC as a side effect — see
    ``test_aggregate_fetch.py`` for the authoritative count contract). When the
    sidecar is reachable but pre-marker, it returns an empty list: the count is
    0 and the advisory ``latest.txt`` pointer must stay unwritten so readers
    never default to an epoch with no aggregate.
    """
    base = tmp_path / "results"
    fake_progress_client = MagicMock()
    fake_progress_client.__aenter__ = AsyncMock(return_value=fake_progress_client)
    fake_progress_client.__aexit__ = AsyncMock(return_value=None)
    fake_progress_client.get_results_list = AsyncMock(return_value=[])
    fake_progress_client.download_all_results = AsyncMock(return_value=[])
    monkeypatch.setattr(
        _aggregate_fetch, "ProgressClient", lambda *args, **kwargs: fake_progress_client
    )

    result = await _aggregate_fetch.fetch_sweep_aggregate_to_disk(
        sweep_name=_SWEEP_NAME,
        namespace=_NAMESPACE,
        epoch=_EPOCH_NEW,
        base_dir=base,
    )

    assert result.downloaded == 0
    assert results_layout.resolve_sweep_latest(base, _NAMESPACE, _SWEEP_NAME) is None
