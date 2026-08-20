# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``aiperf.operator.handlers.cleanup`` not covered by ``test_main.py``.

``test_main.py`` exercises the phase-gating, TTL expiry, and shutil failure
paths. This file covers the path guard, default-TTL fallback, and multi-epoch
filesystem/index reconciliation.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch as mock_patch

import orjson
import pytest

from aiperf.kubernetes.phase import Phase
from aiperf.operator import runs_index
from aiperf.operator.environment import OperatorEnvironment
from aiperf.operator.handlers import cleanup as cleanup_handler
from aiperf.operator.handlers.cleanup import cleanup_old_results
from aiperf.operator.results_layout import (
    LATEST_POINTER,
    resolve_latest,
    resolve_run_dir,
    resolve_sweep_latest,
    run_dir,
    write_latest,
    write_sweep_latest,
)


class TestCleanupPathTraversalGuard:
    """Tests that cleanup refuses to act on paths outside ``RESULTS.DIR``."""

    @pytest.mark.asyncio
    async def test_refuses_path_outside_results_dir(self, tmp_path: Path) -> None:
        """Verify a results_path escaping RESULTS.DIR is skipped and NOT deleted."""
        results_root = tmp_path / "aiperf-results"
        results_root.mkdir()

        # A sibling dir outside results_root — this must NEVER be deleted.
        outside = tmp_path / "not-aiperf" / "stuff"
        outside.mkdir(parents=True)
        victim = outside / "important.txt"
        victim.write_text("do not delete me")

        # Age it past TTL so only the guard can save it.
        old_ts = datetime.now(UTC).timestamp() - (99 * 86400)
        os.utime(outside, (old_ts, old_ts))

        with mock_patch.object(OperatorEnvironment.RESULTS, "DIR", results_root):
            await cleanup_old_results(
                body={},
                status={
                    "phase": Phase.COMPLETED,
                    "jobId": "job-evil",
                    "resultsPath": str(outside),
                    "resultsTtlDays": 1,
                },
                name="test-job",
            )

        assert outside.exists()
        assert victim.read_text() == "do not delete me"

    @pytest.mark.asyncio
    async def test_cleans_up_failed_phase(self, tmp_path: Path) -> None:
        """Verify FAILED phase jobs are cleaned up (they can leak partial artifacts)."""
        results_dir = tmp_path / "job-failed"
        results_dir.mkdir()

        old_ts = datetime.now(UTC).timestamp() - (40 * 86400)
        os.utime(results_dir, (old_ts, old_ts))

        with (
            mock_patch("aiperf.operator.events.results_cleaned"),
            mock_patch.object(OperatorEnvironment.RESULTS, "DIR", tmp_path),
        ):
            await cleanup_old_results(
                body={},
                status={
                    "phase": Phase.FAILED,
                    "jobId": "job-failed",
                    "resultsPath": str(results_dir),
                    "resultsTtlDays": 30,
                },
                name="test-job",
            )

        assert not results_dir.exists()


class TestCleanupEpochReconciliation:
    """TTL cleanup keeps sibling epochs, pointer, and index in agreement."""

    @pytest.mark.asyncio
    async def test_cancelled_job_reaps_every_expired_epoch_and_repoints_latest(
        self, tmp_path: Path
    ) -> None:
        """Cancelled results obey TTL across siblings and retain a usable default."""
        base = tmp_path / "results"
        namespace = "bench"
        job_id = "cancelled-job"
        expired_old = "1710000000"
        retained_older = "1720000000"
        retained_newer = "1730000000"
        expired_latest = "1740000000"
        now = datetime.now(UTC).timestamp()

        for epoch, age_days in (
            (expired_old, 60),
            (retained_older, 1),
            (retained_newer, 2),
            (expired_latest, 40),
        ):
            path = run_dir(base, namespace, job_id, epoch)
            path.mkdir(parents=True)
            os.utime(path, (now - age_days * 86400, now - age_days * 86400))
        write_latest(base, namespace, job_id, expired_latest)

        await runs_index.open(base / ".aiperf_index.sqlite")
        try:
            for epoch in (
                expired_old,
                retained_older,
                retained_newer,
                expired_latest,
            ):
                await runs_index.upsert_run_created(namespace, job_id, epoch, spec={})
            await runs_index.set_latest(namespace, job_id, expired_latest)

            with (
                mock_patch("aiperf.operator.events.results_cleaned"),
                mock_patch.object(OperatorEnvironment.RESULTS, "DIR", base),
            ):
                await cleanup_old_results(
                    body={"metadata": {"namespace": namespace}},
                    status={
                        "phase": Phase.CANCELLED,
                        "jobId": job_id,
                        "resultsPath": str(
                            run_dir(base, namespace, job_id, expired_latest)
                        ),
                        "resultsTtlDays": 30,
                    },
                    name=job_id,
                )

            assert not run_dir(base, namespace, job_id, expired_old).exists()
            assert not run_dir(base, namespace, job_id, expired_latest).exists()
            assert run_dir(base, namespace, job_id, retained_older).is_dir()
            assert run_dir(base, namespace, job_id, retained_newer).is_dir()
            assert resolve_latest(base, namespace, job_id) == retained_newer
            assert resolve_run_dir(base, namespace, job_id) == run_dir(
                base, namespace, job_id, retained_newer
            )

            rows = await runs_index.list_runs_for_job(namespace, job_id)
            assert {row.epoch: row.is_latest for row in rows} == {
                retained_older: False,
                retained_newer: True,
            }
        finally:
            await runs_index.close()

    @pytest.mark.asyncio
    async def test_cleanup_with_no_expired_epochs_preserves_valid_latest(
        self, tmp_path: Path
    ) -> None:
        """A no-op TTL pass cannot replace a valid latest epoch by directory mtime."""
        base = tmp_path / "results"
        namespace = "bench"
        job_id = "recent-job"
        older_epoch = "1720000000"
        latest_epoch = "1730000000"
        now = datetime.now(UTC).timestamp()

        for epoch, age_days in ((older_epoch, 1), (latest_epoch, 2)):
            path = run_dir(base, namespace, job_id, epoch)
            path.mkdir(parents=True)
            os.utime(path, (now - age_days * 86400, now - age_days * 86400))
        write_latest(base, namespace, job_id, latest_epoch)

        await runs_index.open(base / ".aiperf_index.sqlite")
        try:
            for epoch in (older_epoch, latest_epoch):
                await runs_index.upsert_run_created(namespace, job_id, epoch, spec={})
            await runs_index.set_latest(namespace, job_id, latest_epoch)

            with mock_patch.object(OperatorEnvironment.RESULTS, "DIR", base):
                await cleanup_old_results(
                    body={"metadata": {"namespace": namespace}},
                    status={
                        "phase": Phase.COMPLETED,
                        "jobId": job_id,
                        "resultsPath": str(
                            run_dir(base, namespace, job_id, latest_epoch)
                        ),
                        "resultsTtlDays": 30,
                    },
                    name=job_id,
                )

            assert resolve_latest(base, namespace, job_id) == latest_epoch
            rows = await runs_index.list_runs_for_job(namespace, job_id)
            assert {row.epoch: row.is_latest for row in rows} == {
                older_epoch: False,
                latest_epoch: True,
            }
        finally:
            await runs_index.close()

    @pytest.mark.asyncio
    async def test_cleanup_syncs_valid_disk_latest_after_stale_index_latest_deleted(
        self, tmp_path: Path
    ) -> None:
        """Deleting index-latest B must promote surviving disk-latest A in SQLite."""
        base = tmp_path / "results"
        namespace = "bench"
        job_id = "split-latest-job"
        disk_latest = "1730000000"
        stale_index_latest = "1720000000"
        now = datetime.now(UTC).timestamp()

        for epoch, age_days in ((disk_latest, 2), (stale_index_latest, 40)):
            path = run_dir(base, namespace, job_id, epoch)
            path.mkdir(parents=True)
            os.utime(path, (now - age_days * 86400, now - age_days * 86400))
        write_latest(base, namespace, job_id, disk_latest)

        await runs_index.open(base / ".aiperf_index.sqlite")
        try:
            for epoch in (disk_latest, stale_index_latest):
                await runs_index.upsert_run_created(namespace, job_id, epoch, spec={})
            await runs_index.set_latest(namespace, job_id, stale_index_latest)

            with (
                mock_patch("aiperf.operator.events.results_cleaned"),
                mock_patch.object(OperatorEnvironment.RESULTS, "DIR", base),
            ):
                await cleanup_old_results(
                    body={"metadata": {"namespace": namespace}},
                    status={
                        "phase": Phase.COMPLETED,
                        "jobId": job_id,
                        "resultsPath": str(
                            run_dir(base, namespace, job_id, disk_latest)
                        ),
                        "resultsTtlDays": 30,
                    },
                    name=job_id,
                )

            assert resolve_latest(base, namespace, job_id) == disk_latest
            assert not run_dir(base, namespace, job_id, stale_index_latest).exists()
            rows = await runs_index.list_runs_for_job(namespace, job_id)
            assert [(row.epoch, row.is_latest) for row in rows] == [(disk_latest, True)]
        finally:
            await runs_index.close()

    @pytest.mark.asyncio
    async def test_cleanup_last_expired_epoch_removes_latest_pointer(
        self, tmp_path: Path
    ) -> None:
        """Deleting the final epoch clears filesystem and index latest state."""
        base = tmp_path / "results"
        namespace = "bench"
        job_id = "expired-job"
        epoch = "1710000000"
        index_only_epoch = "1740000000"
        path = run_dir(base, namespace, job_id, epoch)
        path.mkdir(parents=True)
        old_time = datetime.now(UTC).timestamp() - 40 * 86400
        os.utime(path, (old_time, old_time))
        write_latest(base, namespace, job_id, epoch)

        await runs_index.open(base / ".aiperf_index.sqlite")
        try:
            await runs_index.upsert_run_created(namespace, job_id, epoch, spec={})
            await runs_index.upsert_run_created(
                namespace, job_id, index_only_epoch, spec={}
            )
            await runs_index.set_latest(namespace, job_id, index_only_epoch)
            with (
                mock_patch("aiperf.operator.events.results_cleaned"),
                mock_patch.object(OperatorEnvironment.RESULTS, "DIR", base),
            ):
                await cleanup_old_results(
                    body={"metadata": {"namespace": namespace}},
                    status={
                        "phase": Phase.COMPLETED,
                        "jobId": job_id,
                        "resultsPath": str(path),
                        "resultsTtlDays": 30,
                    },
                    name=job_id,
                )

            assert not path.exists()
            assert resolve_latest(base, namespace, job_id) is None
            assert not (path.parent / LATEST_POINTER).exists()
            rows = await runs_index.list_runs_for_job(namespace, job_id)
            assert [(row.epoch, row.is_latest) for row in rows] == [
                (index_only_epoch, False)
            ]
        finally:
            await runs_index.close()

    @pytest.mark.asyncio
    async def test_pointer_reconcile_failure_still_removes_deleted_index_row(
        self, tmp_path: Path
    ) -> None:
        """A pointer I/O error cannot strand an index row for a deleted epoch."""
        base = tmp_path / "results"
        namespace = "bench"
        job_id = "pointer-error-job"
        epoch = "1710000000"
        path = run_dir(base, namespace, job_id, epoch)
        path.mkdir(parents=True)
        old_time = datetime.now(UTC).timestamp() - 40 * 86400
        os.utime(path, (old_time, old_time))
        write_latest(base, namespace, job_id, epoch)

        await runs_index.open(base / ".aiperf_index.sqlite")
        try:
            await runs_index.upsert_run_created(namespace, job_id, epoch, spec={})
            await runs_index.set_latest(namespace, job_id, epoch)
            with (
                mock_patch("aiperf.operator.events.results_cleaned"),
                mock_patch.object(OperatorEnvironment.RESULTS, "DIR", base),
                mock_patch(
                    "aiperf.operator.handlers.cleanup.reconcile_latest",
                    side_effect=OSError("pointer write failed"),
                ),
            ):
                await cleanup_old_results(
                    body={"metadata": {"namespace": namespace}},
                    status={
                        "phase": Phase.COMPLETED,
                        "jobId": job_id,
                        "resultsPath": str(path),
                        "resultsTtlDays": 30,
                    },
                    name=job_id,
                )

            assert not path.exists()
            assert await runs_index.get_run(namespace, job_id, epoch) is None
        finally:
            await runs_index.close()

    @pytest.mark.asyncio
    async def test_uses_default_ttl_when_missing(self, tmp_path: Path) -> None:
        """Verify the environment default TTL is used when ``resultsTtlDays`` absent."""
        results_dir = tmp_path / "job-default"
        results_dir.mkdir()

        # Age it just past the env default.
        default_ttl = OperatorEnvironment.RESULTS.TTL_DAYS
        age_days = default_ttl + 1
        old_ts = datetime.now(UTC).timestamp() - (age_days * 86400)
        os.utime(results_dir, (old_ts, old_ts))

        with (
            mock_patch("aiperf.operator.events.results_cleaned"),
            mock_patch.object(OperatorEnvironment.RESULTS, "DIR", tmp_path),
        ):
            await cleanup_old_results(
                body={},
                status={
                    "phase": Phase.COMPLETED,
                    "jobId": "job-default",
                    "resultsPath": str(results_dir),
                    # Note: no resultsTtlDays.
                },
                name="test-job",
            )

        assert not results_dir.exists()


class TestSweepArchiveCleanup:
    """Sweep results TTL is driven from durable archives, not parent CRs."""

    def test_reads_legacy_snake_case_snapshot_ttl(self, tmp_path: Path) -> None:
        """Archives written before canonical alias serialization retain their TTL."""
        epoch_dir = tmp_path / "benchmarks" / "sweeps" / "legacy" / "1714000000"
        epoch_dir.mkdir(parents=True)
        (epoch_dir / "aggregate.json").write_bytes(
            orjson.dumps({"specSnapshot": {"results_ttl_days": 7}})
        )

        with mock_patch.object(OperatorEnvironment.RESULTS, "TTL_DAYS", 30):
            assert cleanup_handler._sweep_results_ttl_days(epoch_dir) == 7

    @pytest.mark.asyncio
    async def test_reaps_expired_epochs_and_reconciles_index_and_latest(
        self, tmp_path: Path
    ) -> None:
        """Each epoch's snapshot TTL governs it after the parent CR is gone."""
        namespace = "benchmarks"
        sweep_name = "latency-sweep"
        retained_epoch = "1730000000"
        expired_latest = "1740000000"
        defaulted_sweep = "default-retention-sweep"
        defaulted_expired_epoch = "1720000000"
        now = datetime.now(UTC).timestamp()

        def seed_epoch(
            name: str,
            epoch: str,
            *,
            age_days: int,
            results_ttl_days: int | None,
        ) -> Path:
            epoch_dir = tmp_path / namespace / "sweeps" / name / epoch
            epoch_dir.mkdir(parents=True)
            snapshot: dict[str, int] = {}
            if results_ttl_days is not None:
                snapshot["resultsTtlDays"] = results_ttl_days
            (epoch_dir / "aggregate.json").write_bytes(
                orjson.dumps({"phase": "Succeeded", "specSnapshot": snapshot})
            )
            mtime = now - age_days * 86400
            os.utime(epoch_dir, (mtime, mtime))
            return epoch_dir

        retained_dir = seed_epoch(
            sweep_name,
            retained_epoch,
            age_days=1,
            results_ttl_days=30,
        )
        expired_dir = seed_epoch(
            sweep_name,
            expired_latest,
            age_days=40,
            results_ttl_days=30,
        )
        defaulted_expired_dir = seed_epoch(
            defaulted_sweep,
            defaulted_expired_epoch,
            age_days=40,
            results_ttl_days=None,
        )
        write_sweep_latest(tmp_path, namespace, sweep_name, expired_latest)
        write_sweep_latest(
            tmp_path,
            namespace,
            defaulted_sweep,
            defaulted_expired_epoch,
        )

        deleted_rows: list[tuple[str, str, str]] = []

        async def delete_sweep_epoch(
            row_namespace: str,
            row_sweep_name: str,
            row_epoch: str,
        ) -> None:
            deleted_rows.append((row_namespace, row_sweep_name, row_epoch))

        async def run_sync(function: Any, *args: Any) -> Any:
            return function(*args)

        with (
            mock_patch.object(OperatorEnvironment.RESULTS, "TTL_DAYS", 30),
            mock_patch("asyncio.to_thread", side_effect=run_sync),
            mock_patch.object(
                runs_index,
                "delete_sweep_epoch",
                delete_sweep_epoch,
                create=True,
            ),
        ):
            reconcile = getattr(
                cleanup_handler,
                "reconcile_sweep_results",
                None,
            )
            if reconcile is not None:
                await reconcile(base_dir=tmp_path)

        assert retained_dir.is_dir()
        assert not expired_dir.exists()
        assert not defaulted_expired_dir.exists()
        assert resolve_sweep_latest(tmp_path, namespace, sweep_name) == retained_epoch
        assert resolve_sweep_latest(tmp_path, namespace, defaulted_sweep) is None
        assert deleted_rows == [
            (namespace, defaulted_sweep, defaulted_expired_epoch),
            (namespace, sweep_name, expired_latest),
        ]

    @pytest.mark.asyncio
    async def test_reconciles_stale_latest_even_when_no_epoch_expires(
        self, tmp_path: Path
    ) -> None:
        """A crash after deletion cannot strand a pointer to an absent epoch."""
        namespace = "benchmarks"
        sweep_name = "latency-sweep"
        retained_epoch = "1730000000"
        stale_epoch = "1740000000"
        epoch_dir = tmp_path / namespace / "sweeps" / sweep_name / retained_epoch
        epoch_dir.mkdir(parents=True)
        (epoch_dir / "aggregate.json").write_bytes(
            orjson.dumps({"specSnapshot": {"resultsTtlDays": 30}})
        )
        write_sweep_latest(tmp_path, namespace, sweep_name, stale_epoch)

        async def run_sync(function: Any, *args: Any) -> Any:
            return function(*args)

        with mock_patch("asyncio.to_thread", side_effect=run_sync):
            await cleanup_handler.reconcile_sweep_results(base_dir=tmp_path)

        assert resolve_sweep_latest(tmp_path, namespace, sweep_name) == retained_epoch
