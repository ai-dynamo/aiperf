# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset readiness contract for WorkerGroupManager."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True, kw_only=True)
class GroupDatasetSnapshot:
    """Local-only dataset state shared with worker groups across run modes."""

    benchmark_generation: str | None = None
    """Benchmark generation currently associated with this dataset snapshot."""

    dataset_generation: str | None = None
    """Dataset generation currently associated with this snapshot."""

    ready: bool = False
    """Whether the dataset is ready for child workers to use."""

    error_message: str | None = None
    """Dataset acquisition error, if the current snapshot is not ready."""


class GroupDatasetAuthority:
    """Tracks the current dataset snapshot used for dispatch gating."""

    def __init__(self) -> None:
        self._snapshot = GroupDatasetSnapshot()

    @property
    def snapshot(self) -> GroupDatasetSnapshot:
        """Return the current dataset snapshot."""
        return self._snapshot

    @property
    def is_ready(self) -> bool:
        """Return whether the group may dispatch children against the dataset."""
        return self._snapshot.ready

    def update_snapshot(self, snapshot: GroupDatasetSnapshot) -> GroupDatasetSnapshot:
        """Store and return the latest dataset snapshot."""
        self._snapshot = snapshot
        return self._snapshot
