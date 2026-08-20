# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime adapter contracts for WorkerGroupManager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from aiperf.common.messages.worker_messages import WorkerStatusSummaryMessage


@dataclass(slots=True)
class GroupRuntimeRegistration:
    """Runtime-provided registration details for a worker group."""

    group_id: str
    """Stable identifier for the worker group."""

    declared_workers: int
    """Maximum worker capacity declared by the active runtime adapter."""

    declared_record_processors: int
    """Maximum record-processor capacity declared by the active runtime adapter."""


class GroupRuntimeAdapter(Protocol):
    """Run-mode specific adapter for group registration and summary publishing."""

    def build_registration(self) -> GroupRuntimeRegistration:
        """Return the group registration for the active runtime."""

    async def publish_summary(self, summary: WorkerStatusSummaryMessage) -> None:
        """Publish the current child summary through the active runtime."""
