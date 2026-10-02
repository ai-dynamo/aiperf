# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared status-precedence table for rolling up worker statuses into a group status.

Canonical home for ``STATUS_RANK`` / ``worst_status`` so the WGM-published group
status (``aiperf.workers.worker_group_state``) and the synthetic in-process
group status served by the dashboard/API (``aiperf.common.mixins.worker_tracker_mixin``)
never diverge.
"""

from collections.abc import Iterable

from aiperf.common.enums import WorkerStatus

STATUS_RANK: dict[WorkerStatus, int] = {
    WorkerStatus.IDLE: 0,
    WorkerStatus.HEALTHY: 1,
    WorkerStatus.HIGH_LOAD: 2,
    WorkerStatus.STALE: 3,
    WorkerStatus.ERROR: 4,
}
"""Precedence used to roll up child statuses into a group status.

Higher rank wins when aggregating across workers:

- ``IDLE = 0`` -- no work, no concern.
- ``HEALTHY = 1`` -- actively working, no concern.
- ``HIGH_LOAD = 2`` -- actively working but CPU-saturated; results may be inaccurate.
- ``STALE = 3`` -- no recent heartbeat; we don't know what state it's in, so we
  treat that uncertainty as worse than known HIGH_LOAD.
- ``ERROR = 4`` -- terminal failure observed.
"""


def worst_status(statuses: Iterable[WorkerStatus]) -> WorkerStatus:
    """Return the highest-precedence status from ``statuses`` per ``STATUS_RANK``.

    Empty input returns ``WorkerStatus.IDLE`` (no workers, no concern).

    Example:
        >>> worst_status([WorkerStatus.HEALTHY, WorkerStatus.ERROR])
        WorkerStatus.ERROR
    """
    materialized = list(statuses)
    if not materialized:
        return WorkerStatus.IDLE
    return max(materialized, key=lambda s: STATUS_RANK.get(s, 0))


__all__ = [
    "STATUS_RANK",
    "worst_status",
]
