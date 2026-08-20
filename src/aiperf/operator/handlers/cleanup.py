# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Results TTL cleanup timer handler logic.

This module contains the business logic only — no kopf decorators.
Decorators live in ``aiperf.operator.main``.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson

from aiperf.kubernetes.phase import Phase
from aiperf.operator import events, runs_index
from aiperf.operator.environment import OperatorEnvironment
from aiperf.operator.results_layout import (
    EPOCH_RE,
    enforce_retention,
    job_dir,
    list_run_epochs,
    reconcile_latest,
    reconcile_sweep_latest,
    resolve_latest,
    run_dir,
)

logger = logging.getLogger(__name__)

SWEEP_RESULTS_CLEANUP_INTERVAL_SECONDS = 86400.0


async def cleanup_old_results(
    body: dict[str, Any],
    status: dict[str, Any],
    name: str,
    **_: Any,
) -> None:
    """Clean up old results based on TTL."""
    # Run cleanup for terminal phases. Failed jobs can still leave partial
    # artifacts on disk (see monitor._recover_from_partial_checkpoints
    # writing results before setting Phase.FAILED) — without this they leak
    # forever.
    if status.get("phase") not in (Phase.COMPLETED, Phase.FAILED, Phase.CANCELLED):
        return

    ttl_days = status.get("resultsTtlDays", OperatorEnvironment.RESULTS.TTL_DAYS)
    if ttl_days <= 0:
        return  # 0 = never clean per RESULTS.TTL_DAYS contract

    job_id = status.get("jobId", name)
    results_path = status.get("resultsPath")

    if not results_path:
        return

    results_dir = Path(results_path)
    # Validate that results_dir is under RESULTS_DIR to prevent path traversal
    try:
        results_dir.resolve().relative_to(OperatorEnvironment.RESULTS.DIR.resolve())
    except ValueError:
        logger.error(
            f"Results path {results_dir} is outside RESULTS_DIR "
            f"{OperatorEnvironment.RESULTS.DIR}, "
            "skipping cleanup"
        )
        return

    namespace = (body.get("metadata") or {}).get("namespace")
    if namespace and _is_epoch_result_path(results_dir, namespace, job_id):
        await _cleanup_expired_epochs(
            body=body,
            namespace=namespace,
            job_id=job_id,
            ttl_days=ttl_days,
        )
        return

    await _cleanup_legacy_result(
        body=body,
        results_dir=results_dir,
        namespace=namespace,
        job_id=job_id,
        ttl_days=ttl_days,
    )


async def reconcile_sweep_results(*, base_dir: Path) -> None:
    """Reap expired durable sweep epochs without relying on a parent CR.

    Each archive carries its own ``specSnapshot.resultsTtlDays``. Missing or
    legacy snapshots use the operator default, so the policy remains available
    after Kubernetes has deleted the short-lived parent resource.
    """
    epochs = _sweep_epoch_dirs(base_dir)
    sweep_roots = {(namespace, sweep_name) for namespace, sweep_name, _ in epochs}
    for namespace, sweep_name, epoch_dir in epochs:
        ttl_days = _sweep_results_ttl_days(epoch_dir)
        if ttl_days <= 0 or not _sweep_epoch_expired(epoch_dir, ttl_days):
            continue
        try:
            await asyncio.to_thread(shutil.rmtree, epoch_dir)
        except (OSError, shutil.Error) as exc:
            logger.warning(
                "Failed to clean up sweep results for %s/%s/%s: %s",
                namespace,
                sweep_name,
                epoch_dir.name,
                exc,
            )
            continue

        try:
            await runs_index.delete_sweep_epoch(
                namespace,
                sweep_name,
                epoch_dir.name,
            )
        except Exception as exc:  # noqa: BLE001 - index is a rebuildable cache
            logger.warning(
                "runs_index.delete_sweep_epoch failed for %s/%s/%s: %s",
                namespace,
                sweep_name,
                epoch_dir.name,
                exc,
            )
    for namespace, sweep_name in sorted(sweep_roots):
        try:
            await asyncio.to_thread(
                reconcile_sweep_latest,
                base_dir,
                namespace,
                sweep_name,
            )
        except (OSError, shutil.Error) as exc:
            logger.warning(
                "Failed to reconcile sweep latest pointer for %s/%s: %s",
                namespace,
                sweep_name,
                exc,
            )


def _sweep_epoch_dirs(base_dir: Path) -> list[tuple[str, str, Path]]:
    """Discover canonical durable sweep epoch directories deterministically."""
    if not base_dir.is_dir():
        return []
    found: list[tuple[str, str, Path]] = []
    for namespace_dir in sorted(base_dir.iterdir(), key=lambda path: path.name):
        sweeps_dir = namespace_dir / "sweeps"
        if not namespace_dir.is_dir() or not sweeps_dir.is_dir():
            continue
        for sweep_dir in sorted(sweeps_dir.iterdir(), key=lambda path: path.name):
            if not sweep_dir.is_dir():
                continue
            for epoch_dir in sorted(sweep_dir.iterdir(), key=lambda path: path.name):
                if epoch_dir.is_dir() and EPOCH_RE.match(epoch_dir.name):
                    found.append((namespace_dir.name, sweep_dir.name, epoch_dir))
    return found


def _sweep_results_ttl_days(epoch_dir: Path) -> int:
    """Read an epoch's durable TTL, falling back to the operator default."""
    fallback = OperatorEnvironment.RESULTS.TTL_DAYS
    try:
        doc = orjson.loads((epoch_dir / "aggregate.json").read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return fallback
    snapshot = doc.get("specSnapshot") if isinstance(doc, dict) else None
    if not isinstance(snapshot, dict):
        return fallback
    value = snapshot.get("resultsTtlDays", snapshot.get("results_ttl_days"))
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return fallback
    return value


def _sweep_epoch_expired(epoch_dir: Path, ttl_days: int) -> bool:
    """Return whether the archive directory is older than its TTL."""
    try:
        age_seconds = datetime.now(UTC).timestamp() - epoch_dir.stat().st_mtime
    except OSError:
        return False
    return age_seconds > ttl_days * 86400


def _is_epoch_result_path(results_dir: Path, namespace: str, job_id: str) -> bool:
    """Return whether a status path names the canonical epoch layout."""
    expected_job_dir = job_dir(OperatorEnvironment.RESULTS.DIR, namespace, job_id)
    return (
        EPOCH_RE.match(results_dir.name) is not None
        and results_dir.parent.resolve() == expected_job_dir.resolve()
    )


async def _cleanup_legacy_result(
    *,
    body: dict[str, Any],
    results_dir: Path,
    namespace: str | None,
    job_id: str,
    ttl_days: int,
) -> None:
    """Preserve cleanup for pre-epoch-layout result directories."""
    if not results_dir.exists():
        return

    try:
        mtime = results_dir.stat().st_mtime
        age_days = (datetime.now(UTC).timestamp() - mtime) / 86400

        if age_days > ttl_days:
            await asyncio.to_thread(shutil.rmtree, results_dir)
            logger.info(
                f"Cleaned up old results for {job_id} (age: {age_days:.0f} days)"
            )
            events.results_cleaned(body, job_id, int(age_days))
            if namespace:
                try:
                    await runs_index.delete_run(namespace, job_id, results_dir.name)
                except Exception as exc:  # noqa: BLE001 - best-effort index sync
                    logger.warning(
                        "runs_index.delete_run failed for %s/%s/%s: %s",
                        namespace,
                        job_id,
                        results_dir.name,
                        exc,
                    )
    except (OSError, shutil.Error) as e:
        logger.warning(f"Failed to clean up results for {job_id}: {e}")


async def _cleanup_expired_epochs(
    *,
    body: dict[str, Any],
    namespace: str,
    job_id: str,
    ttl_days: int,
) -> None:
    """Reap every expired epoch and reconcile disk/index latest state."""
    base = OperatorEnvironment.RESULTS.DIR
    ages = await asyncio.to_thread(_epoch_ages, base, namespace, job_id)

    try:
        deleted = await asyncio.to_thread(
            enforce_retention,
            base,
            namespace,
            job_id,
            keep=0,
            protect_epoch=None,
            retain_days=ttl_days,
        )
    except (OSError, shutil.Error) as exc:
        logger.warning(f"Failed to clean up results for {job_id}: {exc}")
        return

    latest_resolved, latest = await _reconcile_disk_latest(base, namespace, job_id)
    await _delete_index_rows(namespace, job_id, deleted)
    if latest_resolved:
        await _reconcile_index_latest(namespace, job_id, latest)

    if deleted:
        max_age = max(
            (ages.get(epoch, ttl_days) for epoch in deleted), default=ttl_days
        )
        logger.info(
            "Cleaned up %d old result epoch(s) for %s (oldest age: %d days)",
            len(deleted),
            job_id,
            max_age,
        )
        events.results_cleaned(body, job_id, max_age)


def _epoch_ages(base: Path, namespace: str, job_id: str) -> dict[str, int]:
    """Capture epoch ages before deletion for an accurate cleanup event."""
    now = datetime.now(UTC).timestamp()
    root = job_dir(base, namespace, job_id)
    ages: dict[str, int] = {}
    for epoch in list_run_epochs(base, namespace, job_id):
        try:
            ages[epoch] = int((now - (root / epoch).stat().st_mtime) / 86400)
        except OSError:
            continue
    return ages


async def _reconcile_disk_latest(
    base: Path, namespace: str, job_id: str
) -> tuple[bool, str | None]:
    """Resolve a valid pointer or repair it, returning whether resolution succeeded."""
    try:
        current = await asyncio.to_thread(_valid_latest_epoch, base, namespace, job_id)
    except OSError:
        current = None
    if current is not None:
        return True, current
    try:
        latest = await asyncio.to_thread(reconcile_latest, base, namespace, job_id)
    except (OSError, shutil.Error) as exc:
        logger.warning(f"Failed to reconcile latest results for {job_id}: {exc}")
        return False, None
    return True, latest


def _valid_latest_epoch(base: Path, namespace: str, job_id: str) -> str | None:
    """Return the pointer epoch only when its run directory survives."""
    current = resolve_latest(base, namespace, job_id)
    if current is None or EPOCH_RE.match(current) is None:
        return None
    return current if run_dir(base, namespace, job_id, current).is_dir() else None


async def _delete_index_rows(namespace: str, job_id: str, deleted: list[str]) -> None:
    """Best-effort drop every index row whose result directory was removed."""
    for epoch in deleted:
        try:
            await runs_index.delete_run(namespace, job_id, epoch)
        except Exception as exc:  # noqa: BLE001 - best-effort index sync
            logger.warning(
                "runs_index.delete_run failed for %s/%s/%s: %s",
                namespace,
                job_id,
                epoch,
                exc,
            )


async def _reconcile_index_latest(
    namespace: str, job_id: str, latest: str | None
) -> None:
    """Mirror a successfully reconciled disk pointer into the runs index."""
    try:
        if latest is None:
            await runs_index.clear_latest(namespace, job_id)
        else:
            await runs_index.set_latest(namespace, job_id, latest)
    except Exception as exc:  # noqa: BLE001 - best-effort index sync
        logger.warning(
            "runs_index latest reconciliation failed for %s/%s/%s: %s",
            namespace,
            job_id,
            latest,
            exc,
        )


async def on_aiperfjob_delete_index_cleanup(
    namespace: str, name: str, status: dict[str, Any]
) -> None:
    """Drop every index row for a deleted AIPerfJob.

    Wired from ``main.on_aiperfjob_delete``. The CR delete handler in
    ``lifecycle.on_delete`` does not touch disk (results retention is
    independent of CR lifecycle), but the index entries become orphaned
    when the CR is gone — ``aiperf kube history`` would still surface
    them. Walk every epoch dir on disk plus every index row and drop
    matching index rows; missing-on-both is a no-op.

    Best-effort: any failure logs and swallows so on_delete remains fast.
    """
    job_id = status.get("jobId", name)
    base = OperatorEnvironment.RESULTS.DIR
    epochs: set[str] = set()
    try:
        epochs.update(list_run_epochs(base, namespace, job_id))
    except OSError as exc:
        logger.warning(
            "list_run_epochs failed for %s/%s during on_delete: %s",
            namespace,
            job_id,
            exc,
        )
    try:
        rows = await runs_index.list_runs_for_job(namespace, job_id)
        epochs.update(r.epoch for r in rows)
    except Exception as exc:  # noqa: BLE001 - best-effort index read
        logger.warning(
            "runs_index.list_runs_for_job failed for %s/%s: %s",
            namespace,
            job_id,
            exc,
        )
    for epoch in epochs:
        try:
            await runs_index.delete_run(namespace, job_id, epoch)
        except Exception as exc:  # noqa: BLE001 - best-effort index sync
            logger.warning(
                "runs_index.delete_run failed for %s/%s/%s: %s",
                namespace,
                job_id,
                epoch,
                exc,
            )
