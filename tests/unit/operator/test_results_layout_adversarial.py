# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for Kubernetes results layout fallback behavior.

Focuses on disk/index trust-boundary failures:
- missing runs_index connections falling back to the PVC layout
- stale index rows merging with newer disk epochs without hiding disk truth
- malformed run-directory shapes and corrupt latest pointers
- URL-encoded SQLite paths used by read-only index sidecars
- cache failures that must degrade to slower reads, not wrong reads

Out of scope (covered elsewhere):
- retention policy and epoch-key generation: tests/unit/operator/test_results_layout.py
- index row upsert semantics: tests/unit/operator/test_runs_index.py
- summary decoding and analytics fallbacks: tests/unit/operator/test_runs_index_adversarial.py
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from aiperf.operator import results_layout, runs_index
from aiperf.operator.results_layout import RunEntry
from aiperf.operator.runs_index_models import RunIndexRow

# ============================================================================
# Helpers
# ============================================================================


_EPOCH_OLD = "1716061001"
_EPOCH_NEW = "1716061101"
_EPOCH_STALE = "1716060901"


def _make_run_dir(
    base: Path,
    namespace: str = "bench-prod",
    job_id: str = "llama-results-7f2a",
    epoch: str = _EPOCH_NEW,
    *,
    latest: bool = False,
    filename: str = "profile_export_aiperf.json",
    contents: bytes = b"{}",
) -> Path:
    """Create one realistic run directory with a small result artifact."""
    path = results_layout.run_dir(base, namespace, job_id, epoch)
    path.mkdir(parents=True)
    (path / filename).write_bytes(contents)
    if latest:
        results_layout.write_latest(base, namespace, job_id, epoch)
    return path


def _index_row(
    *,
    namespace: str = "bench-prod",
    job_id: str = "llama-results-7f2a",
    epoch: str = _EPOCH_STALE,
    is_latest: bool = False,
    mtime_epoch: int | None = None,
    file_count: int = 1,
    total_size_bytes: int = 2,
) -> RunIndexRow:
    """Build a runs_index row shaped like SQLite would hydrate it."""
    return RunIndexRow(
        namespace=namespace,
        job_id=job_id,
        epoch=epoch,
        phase="Succeeded",
        is_latest=is_latest,
        start_time="2026-05-18T11:00:00Z",
        end_time="2026-05-18T11:05:00Z",
        created_unix=int(epoch) if epoch.isdigit() else 0,
        mtime_epoch=mtime_epoch,
        error=None,
        model="meta-llama/Llama-3-8B",
        endpoint="http://inference.local:8000",
        gpu_count=0,
        gpu_name=None,
        file_count=file_count,
        total_size_bytes=total_size_bytes,
        sweep_namespace=None,
        sweep_name=None,
        sweep_epoch=None,
        sweep_variation_idx=None,
    )


@pytest.fixture
async def opened_index(tmp_path: Path) -> AsyncGenerator[Path, None]:
    """Open a fresh writable runs_index DB and close it after the test."""
    path = tmp_path / "index" / ".aiperf_index.sqlite"
    await runs_index.open(path)
    try:
        yield path
    finally:
        await runs_index.close()


# ============================================================================
# Missing and stale index fallbacks
# ============================================================================


@pytest.mark.asyncio
async def test_list_runs_async_unopened_index_falls_back_to_disk_truth(
    tmp_path: Path,
) -> None:
    """Missing sidecar index state must degrade to the PVC walk, not empty results."""
    base = tmp_path / "results"
    _make_run_dir(base, epoch=_EPOCH_OLD)
    _make_run_dir(base, epoch=_EPOCH_NEW, latest=True)

    runs = await results_layout.list_runs_async(
        base, "bench-prod", "llama-results-7f2a"
    )

    assert {run.epoch for run in runs} == {_EPOCH_OLD, _EPOCH_NEW}
    assert {run.epoch for run in runs if run.is_latest} == {_EPOCH_NEW}


@pytest.mark.asyncio
async def test_list_runs_async_stale_index_row_missing_on_disk_is_ignored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pruned cache row must not resurrect a run directory absent from disk."""
    base = tmp_path / "results"
    _make_run_dir(base, epoch=_EPOCH_NEW, latest=True)

    async def fake_index_rows(namespace: str, job_id: str) -> list[RunIndexRow]:
        return [
            _index_row(
                namespace=namespace,
                job_id=job_id,
                epoch=_EPOCH_STALE,
                is_latest=True,
                mtime_epoch=9999999999,
            )
        ]

    monkeypatch.setattr(runs_index, "list_runs_for_job", fake_index_rows)

    runs = await results_layout.list_runs_async(
        base, "bench-prod", "llama-results-7f2a"
    )

    assert [run.epoch for run in runs] == [_EPOCH_NEW]
    assert runs[0].is_latest is True


@pytest.mark.asyncio
async def test_list_runs_async_real_index_stale_latest_yields_disk_latest(
    tmp_path: Path,
    opened_index: Path,
) -> None:
    """Disk latest.txt must override an older is_latest row in the cache merge."""
    base = tmp_path / "results"
    _make_run_dir(base, epoch=_EPOCH_OLD)
    _make_run_dir(base, epoch=_EPOCH_NEW, latest=True)
    await runs_index.upsert_run_created(
        "bench-prod", "llama-results-7f2a", _EPOCH_OLD, spec={"benchmark": {}}
    )
    await runs_index.upsert_run_completed(
        "bench-prod",
        "llama-results-7f2a",
        _EPOCH_OLD,
        summary_blob=b"{}",
        metrics={},
        files=["profile_export_aiperf.json"],
        mtime_epoch=int(_EPOCH_OLD),
        total_size_bytes=2,
    )
    await runs_index.set_latest("bench-prod", "llama-results-7f2a", _EPOCH_OLD)

    runs = await results_layout.list_runs_async(
        base, "bench-prod", "llama-results-7f2a"
    )

    assert {run.epoch for run in runs} == {_EPOCH_OLD, _EPOCH_NEW}
    assert {run.epoch for run in runs if run.is_latest} == {_EPOCH_NEW}


@pytest.mark.asyncio
async def test_list_runs_async_index_read_error_falls_back_to_disk_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A corrupt cache SELECT should not turn an available run directory into a 500."""
    base = tmp_path / "results"
    _make_run_dir(base, epoch=_EPOCH_NEW, latest=True)

    async def failing_index_rows(namespace: str, job_id: str) -> list[RunIndexRow]:
        raise sqlite3.DatabaseError(
            f"malformed runs_index cache while reading {namespace}/{job_id}"
        )

    monkeypatch.setattr(runs_index, "list_runs_for_job", failing_index_rows)

    runs = await results_layout.list_runs_async(
        base, "bench-prod", "llama-results-7f2a"
    )

    assert [run.epoch for run in runs] == [_EPOCH_NEW]


@pytest.mark.asyncio
async def test_list_runs_async_populated_current_index_avoids_disk_walk(
    tmp_path: Path,
    opened_index: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A current indexed run history must not enumerate every artifact on disk."""
    base = tmp_path / "results"
    _make_run_dir(base, epoch=_EPOCH_OLD)
    _make_run_dir(base, epoch=_EPOCH_NEW, latest=True)
    for epoch, latest in ((_EPOCH_OLD, False), (_EPOCH_NEW, True)):
        await runs_index.upsert_run_created(
            "bench-prod", "llama-results-7f2a", epoch, spec={"benchmark": {}}
        )
        await runs_index.upsert_run_completed(
            "bench-prod",
            "llama-results-7f2a",
            epoch,
            summary_blob=b"{}",
            metrics={},
            files=["profile_export_aiperf.json"],
            mtime_epoch=int(epoch),
            total_size_bytes=2,
        )
        if latest:
            await runs_index.set_latest("bench-prod", "llama-results-7f2a", epoch)
    runs_index.mark_catalog_complete(base)

    def fail_disk_walk(*args: object, **kwargs: object) -> list[RunEntry]:
        raise AssertionError("unexpected recursive run-directory walk")

    monkeypatch.setattr(results_layout, "_walk_runs", fail_disk_walk)

    runs = await results_layout.list_runs_async(
        base, "bench-prod", "llama-results-7f2a"
    )

    assert [run.epoch for run in runs] == [_EPOCH_NEW, _EPOCH_OLD]
    assert {run.epoch for run in runs if run.is_latest} == {_EPOCH_NEW}


@pytest.mark.asyncio
async def test_list_runs_async_incomplete_index_preserves_and_backfills_disk_epoch(
    tmp_path: Path,
    opened_index: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A current latest row does not prove older disk epochs are indexed."""
    base = tmp_path / "results"
    _make_run_dir(base, epoch=_EPOCH_OLD)
    _make_run_dir(base, epoch=_EPOCH_NEW, latest=True)
    await runs_index.upsert_run_created(
        "bench-prod", "llama-results-7f2a", _EPOCH_NEW, spec={"benchmark": {}}
    )
    await runs_index.upsert_run_completed(
        "bench-prod",
        "llama-results-7f2a",
        _EPOCH_NEW,
        summary_blob=b"{}",
        metrics={},
        files=["profile_export_aiperf.json"],
        mtime_epoch=int(_EPOCH_NEW),
        total_size_bytes=2,
    )
    await runs_index.set_latest("bench-prod", "llama-results-7f2a", _EPOCH_NEW)
    backfilled_epochs: list[str] = []

    async def capture_backfill(
        base: Path, namespace: str, job_id: str, epoch: str
    ) -> bool:
        backfilled_epochs.append(epoch)
        return True

    monkeypatch.setattr(runs_index, "lazy_backfill_run", capture_backfill)

    runs = await results_layout.list_runs_async(
        base, "bench-prod", "llama-results-7f2a"
    )
    await asyncio.sleep(0)

    assert {run.epoch for run in runs} == {_EPOCH_OLD, _EPOCH_NEW}
    assert _EPOCH_OLD in backfilled_epochs


# ============================================================================
# Malformed disk layout and latest-pointer corruption
# ============================================================================


def test_list_runs_malformed_entries_and_bad_epochs_are_ignored(
    tmp_path: Path,
) -> None:
    """Noise under a job root must not be presented as benchmark run history."""
    base = tmp_path / "results"
    _make_run_dir(base, epoch=_EPOCH_OLD)
    _make_run_dir(base, epoch=_EPOCH_NEW, latest=True)
    root = results_layout.job_dir(base, "bench-prod", "llama-results-7f2a")
    (root / "latest.txt.tmp").write_text(_EPOCH_OLD)
    (root / "171606").mkdir()
    (root / "171606100112345678901").mkdir()
    (root / "1716061O01").mkdir()
    (root / "profile_export_aiperf.json").write_text("{}")

    runs = results_layout.list_runs(base, "bench-prod", "llama-results-7f2a")

    assert {run.epoch for run in runs} == {_EPOCH_OLD, _EPOCH_NEW}
    assert all(run.file_count == 1 for run in runs)


def _poison_latest_pointer(base: Path, namespace: str, job_id: str, value: str) -> None:
    """Write a garbage value straight into latest.txt, bypassing write_latest.

    ``write_latest`` now rejects non-epoch values (the write-time guard tested
    in ``test_latest_pointer_adversarial.py``). These read-side tests need to
    simulate an already-corrupted pointer on disk — e.g. left by an older
    operator build or a manual edit — so they write the file directly and then
    assert the READ path (``list_runs`` / ``resolve_run_dir``) refuses to trust
    it. Defense in depth: both the writer and the reader reject junk.
    """
    pointer = results_layout.job_dir(base, namespace, job_id) / "latest.txt"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(value)


def test_list_runs_corrupt_latest_pointer_lists_disk_runs_without_false_latest(
    tmp_path: Path,
) -> None:
    """A garbage latest.txt value must not fabricate a latest entry."""
    base = tmp_path / "results"
    _make_run_dir(base, epoch=_EPOCH_OLD)
    _make_run_dir(base, epoch=_EPOCH_NEW)
    _poison_latest_pointer(
        base, "bench-prod", "llama-results-7f2a", "not-an-epoch-anymore"
    )

    runs = results_layout.list_runs(base, "bench-prod", "llama-results-7f2a")

    assert {run.epoch for run in runs} == {_EPOCH_OLD, _EPOCH_NEW}
    assert [run for run in runs if run.is_latest] == []


def test_resolve_run_dir_corrupt_latest_path_traversal_returns_none(
    tmp_path: Path,
) -> None:
    """A poisoned latest.txt pointer must not escape <base>/<namespace>/<job>."""
    base = tmp_path / "results"
    _make_run_dir(base, epoch=_EPOCH_NEW)
    escape = base / "bench-prod" / "escaped-results-9d2c"
    escape.mkdir(parents=True)
    _poison_latest_pointer(
        base, "bench-prod", "llama-results-7f2a", "../escaped-results-9d2c"
    )

    assert (
        results_layout.resolve_run_dir(base, "bench-prod", "llama-results-7f2a") is None
    )


def test_resolve_run_dir_explicit_bad_epoch_path_traversal_returns_none(
    tmp_path: Path,
) -> None:
    """An explicit URL epoch parameter must be epoch-shaped before path joining."""
    base = tmp_path / "results"
    _make_run_dir(base, epoch=_EPOCH_NEW)
    escape = base / "bench-prod" / "escaped-results-9d2c"
    escape.mkdir(parents=True)

    assert (
        results_layout.resolve_run_dir(
            base, "bench-prod", "llama-results-7f2a", epoch="../escaped-results-9d2c"
        )
        is None
    )


# ============================================================================
# URL/path encoding and read-only sidecar cache
# ============================================================================


@pytest.mark.asyncio
async def test_runs_index_open_readonly_encoded_db_path_returns_cached_row(
    tmp_path: Path,
) -> None:
    """Sidecar read-only URI quoting must survive spaces, percent, and hash chars."""
    base = tmp_path / "results dir with spaces # blue%25"
    db_path = base / ".aiperf_index.sqlite"
    await runs_index.open(db_path)
    try:
        await runs_index.upsert_run_created(
            "bench-prod", "encoded-path-bench-5c1e", _EPOCH_NEW, spec={"benchmark": {}}
        )
        await runs_index.set_latest("bench-prod", "encoded-path-bench-5c1e", _EPOCH_NEW)
    finally:
        await runs_index.close()

    await runs_index.open_readonly(db_path)
    try:
        row = await runs_index.get_run(
            "bench-prod", "encoded-path-bench-5c1e", _EPOCH_NEW
        )
    finally:
        await runs_index.close()

    assert row is not None
    assert row.is_latest is True


@pytest.mark.asyncio
async def test_list_runs_async_preserves_disk_file_stats_over_stale_cache_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the same epoch exists in both places, disk stats are the truthful row."""
    base = tmp_path / "results"
    _make_run_dir(
        base,
        epoch=_EPOCH_NEW,
        latest=True,
        filename="profile_export_aiperf.json",
        contents=b'{"request_throughput":{"avg":123.0}}',
    )
    run_path = results_layout.run_dir(
        base, "bench-prod", "llama-results-7f2a", _EPOCH_NEW
    )
    (run_path / "server_metrics.parquet").write_bytes(b"metrics")

    async def fake_index_rows(namespace: str, job_id: str) -> list[RunIndexRow]:
        return [
            _index_row(
                namespace=namespace,
                job_id=job_id,
                epoch=_EPOCH_NEW,
                is_latest=False,
                mtime_epoch=1,
                file_count=99,
                total_size_bytes=999999,
            )
        ]

    monkeypatch.setattr(runs_index, "list_runs_for_job", fake_index_rows)

    runs = await results_layout.list_runs_async(
        base, "bench-prod", "llama-results-7f2a"
    )

    assert runs == [
        RunEntry(
            epoch=_EPOCH_NEW,
            mtime_epoch=runs[0].mtime_epoch,
            file_count=2,
            total_size_bytes=len(b'{"request_throughput":{"avg":123.0}}')
            + len(b"metrics"),
            is_latest=True,
        )
    ]
