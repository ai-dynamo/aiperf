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

from aiperf.operator import events, runs_index
from aiperf.operator.environment import OperatorEnvironment
from aiperf.operator.results_layout import list_run_epochs
from aiperf.operator.status import Phase

logger = logging.getLogger(__name__)


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
    if status.get("phase") not in (Phase.COMPLETED, Phase.FAILED):
        return

    ttl_days = status.get("resultsTtlDays", OperatorEnvironment.RESULTS.TTL_DAYS)
    if ttl_days <= 0:
        return  # 0 = never clean per RESULTS.TTL_DAYS contract

    job_id = status.get("jobId", name)
    results_path = status.get("resultsPath")

    if not results_path:
        return

    results_dir = Path(results_path)
    if not results_dir.exists():
        return

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

    # Check if results are older than TTL
    try:
        mtime = results_dir.stat().st_mtime
        age_days = (datetime.now(UTC).timestamp() - mtime) / 86400

        if age_days > ttl_days:
            await asyncio.to_thread(shutil.rmtree, results_dir)
            logger.info(
                f"Cleaned up old results for {job_id} (age: {age_days:.0f} days)"
            )
            events.results_cleaned(body, job_id, int(age_days))
            # Best-effort: drop the index row for the deleted epoch so the
            # index never lags disk. The epoch is the last component of
            # ``results_path`` (``<base>/<ns>/<job>/<epoch>``); namespace
            # comes from the CR body since the timer signature does not
            # plumb it through. Failures log and continue — disk truth wins.
            namespace = (body.get("metadata") or {}).get("namespace")
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
