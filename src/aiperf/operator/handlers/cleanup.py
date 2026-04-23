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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aiperf.operator import events
from aiperf.operator.environment import OperatorEnvironment
from aiperf.operator.status import Phase

logger = logging.getLogger(__name__)


async def cleanup_old_results(
    body: dict[str, Any],
    status: dict[str, Any],
    name: str,
    **_: Any,
) -> None:
    """Clean up old results based on TTL."""
    # Only run cleanup for completed jobs
    if status.get("phase") != Phase.COMPLETED:
        return

    ttl_days = status.get("resultsTtlDays", OperatorEnvironment.RESULTS.TTL_DAYS)
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
        age_days = (datetime.now(timezone.utc).timestamp() - mtime) / 86400

        if age_days > ttl_days:
            await asyncio.to_thread(shutil.rmtree, results_dir)
            logger.info(
                f"Cleaned up old results for {job_id} (age: {age_days:.0f} days)"
            )
            events.results_cleaned(body, job_id, int(age_days))
    except (OSError, shutil.Error) as e:
        logger.warning(f"Failed to clean up results for {job_id}: {e}")
