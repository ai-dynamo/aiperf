# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The group-startup recovery path must not re-enter the ready lock.

Regression: ``_complete_group_startup_flow`` holds ``_worker_ready_lock`` while
calling ``_open_dataset_client``, whose tail called ``_mark_worker_ready`` --
which re-acquires that same lock. :class:`asyncio.Lock` is not reentrant, so a
worker recovering from a missed dataset broadcast hung forever holding the lock
and never became dispatchable.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from aiperf.common.pod_lifecycle_structs import GroupDatasetStateSnapshot
from aiperf.workers.worker import Worker


class _FakeWorker:
    """Only the state the group-startup flow touches."""

    def __init__(self) -> None:
        self._is_kubernetes = True
        self._worker_ready_event = asyncio.Event()
        self._worker_ready_lock = asyncio.Lock()
        self._dataset_configured_event = asyncio.Event()
        self.session_manager = AsyncMock()
        self.debug = lambda *a, **k: None
        self.opened: list[object] = []
        self.marked_ready_locked = 0

    async def _open_dataset_client(self, client_metadata, mark_ready=True) -> None:
        self.opened.append(client_metadata)
        self._dataset_configured_event.set()
        if mark_ready:
            await Worker._mark_worker_ready(self)

    async def _mark_worker_ready_locked(self) -> None:
        self.marked_ready_locked += 1
        self._worker_ready_event.set()

    def _ensure_group_dataset_state_retry(self) -> None:
        pass

    def _is_group_managed_mode(self) -> bool:
        return self._is_kubernetes


def _snapshot() -> GroupDatasetStateSnapshot:
    return GroupDatasetStateSnapshot(
        rid="r-1",
        service_id="wgm",
        ready=True,
        data_file_path="/pod/dataset.dat",
        index_file_path="/pod/index.dat",
        conversation_count=1,
        total_size_bytes=8,
    )


@pytest.mark.asyncio
async def test_group_startup_recovery_does_not_self_deadlock() -> None:
    worker = _FakeWorker()

    await asyncio.wait_for(
        Worker._complete_group_startup_flow(worker, _snapshot()), timeout=5
    )

    assert [m.data_file_path for m in worker.opened] == [Path("/pod/dataset.dat")]
    assert worker.marked_ready_locked == 1
    assert worker._worker_ready_event.is_set()
    assert not worker._worker_ready_lock.locked()
