# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WorkerGroupManager service export and shared worker-group state helpers."""

from __future__ import annotations

from aiperf.workers.worker_group_state import (
    WorkerStatusInfo,
    build_worker_status_summary,
    mark_stale_workers,
    update_worker_status,
)
from aiperf.workers.worker_pod_manager import WorkerGroupManagerBase


class WorkerGroupManager(WorkerGroupManagerBase):
    """Kubernetes worker group manager service."""


__all__ = [
    "WorkerGroupManager",
    "WorkerStatusInfo",
    "build_worker_status_summary",
    "mark_stale_workers",
    "update_worker_status",
]
