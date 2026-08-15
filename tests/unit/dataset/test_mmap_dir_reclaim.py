# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The conversation mmap store must leave no directory behind.

``MemoryMapDatasetBackingStore._cleanup`` unlinked its four files but never
removed ``aiperf_mmap_<benchmark_id>/`` itself, so EVERY run -- clean exits
included -- leaked one empty directory under the mmap base path. 2,197 had
accumulated on one developer machine, and the base path is commonly the system
temp dir, whose dirent table every process pays to scan.
"""

from __future__ import annotations

import pytest

from aiperf.common.enums import MemoryMapFormat
from aiperf.common.environment import Environment
from aiperf.common.models import Conversation, Turn
from aiperf.dataset.memory_map_utils import MemoryMapDatasetBackingStore


def _conversation(session_id: str) -> Conversation:
    return Conversation(
        session_id=session_id,
        turns=[Turn(role="user", raw_payload={"messages": [], "model": "m"})],
    )


@pytest.mark.asyncio
async def test_stop_removes_the_run_directory_not_just_its_files(
    tmp_path, monkeypatch
) -> None:
    """A clean run leaves no ``aiperf_mmap_<id>/`` directory behind."""
    monkeypatch.setattr(Environment.DATASET, "MMAP_BASE_PATH", tmp_path)

    store = MemoryMapDatasetBackingStore(
        benchmark_id="reclaim_me", format=MemoryMapFormat.PAYLOAD_BYTES
    )
    await store.initialize()
    await store.add_conversation("conv-1", _conversation("conv-1"))
    await store.finalize()

    run_dir = tmp_path / "aiperf_mmap_reclaim_me"
    assert run_dir.is_dir(), "the store should have created its run dir"

    await store.stop()

    assert not run_dir.exists(), "stop unlinked the files but left the directory behind"


@pytest.mark.asyncio
async def test_stop_keeps_a_directory_that_still_holds_foreign_files(
    tmp_path, monkeypatch
) -> None:
    """Reclaim is conservative: an unexpected file keeps the directory alive."""
    monkeypatch.setattr(Environment.DATASET, "MMAP_BASE_PATH", tmp_path)

    store = MemoryMapDatasetBackingStore(
        benchmark_id="keep_me", format=MemoryMapFormat.PAYLOAD_BYTES
    )
    await store.initialize()
    await store.add_conversation("conv-1", _conversation("conv-1"))
    await store.finalize()

    run_dir = tmp_path / "aiperf_mmap_keep_me"
    (run_dir / "someone-elses.txt").write_text("do not delete me")

    await store.stop()

    assert run_dir.is_dir(), "a non-empty run dir must not be removed"
    assert (run_dir / "someone-elses.txt").exists()
