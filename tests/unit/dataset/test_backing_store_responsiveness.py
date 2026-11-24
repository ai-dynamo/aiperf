# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset saving must leave the service event loop responsive."""

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from aiperf.common.models import Conversation, Turn
from aiperf.dataset.memory_map_utils import (
    MemoryMapDatasetBackingStore,
    MemoryMapDatasetIndex,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["conversation", "index", "write", "close"])
async def test_saving_does_not_block_event_loop(
    operation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    started = asyncio.Event()
    release = threading.Event()
    store = MemoryMapDatasetBackingStore(
        compress_only=operation in {"write", "close"},
    )
    await store.initialize()
    conv = Conversation(
        session_id="s", turns=[Turn(role="user", raw_payload={"text": "hi"})]
    )
    if operation in {"index", "close"}:
        await store.add_conversation("s", conv)

    if operation == "conversation":
        owner, attribute = Conversation, "model_dump_json"
    elif operation == "index":
        owner, attribute = MemoryMapDatasetIndex, "model_dump_json"
    else:
        stream = store._stream_writer
        store._stream_writer = SimpleNamespace(write=stream.write, close=stream.close)
        owner, attribute = store._stream_writer, operation
    original = getattr(owner, attribute)

    def slow_operation(*args: Any, **kwargs: Any) -> Any:
        assert threading.get_ident() != loop_thread
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, attribute, slow_operation)
    task = asyncio.create_task(
        store.finalize()
        if operation in {"index", "close"}
        else store.add_conversation("s", conv)
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        assert not task.done()
        release.set()
        await task
        if not store._finalized:
            await store.finalize()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await store.stop()


@pytest.mark.asyncio
async def test_cancelled_compressed_write_finishes_before_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
    store = MemoryMapDatasetBackingStore(compress_only=True)
    await store.initialize()
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    order = []
    stream = store._stream_writer

    def slow_write(data: bytes) -> None:
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=5)
        stream.write(data)
        order.append("write")

    def close() -> None:
        order.append("close")
        stream.close()

    store._stream_writer = SimpleNamespace(write=slow_write, close=close)
    task = asyncio.create_task(
        store.add_conversation("s", Conversation(session_id="s", turns=[]))
    )
    cleanup_started = asyncio.Event()
    original_close = store._close_compressed

    def begin_cleanup() -> None:
        loop.call_soon_threadsafe(cleanup_started.set)
        original_close()

    monkeypatch.setattr(store, "_close_compressed", begin_cleanup)
    cleanup = None
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        cleanup = asyncio.create_task(store.stop())
        await asyncio.wait_for(cleanup_started.wait(), timeout=2)
        assert "close" not in order
        release.set()
        await asyncio.wait_for(cleanup, timeout=2)
        assert order == ["write", "close"]
        assert not store._compressed_data_path.exists()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        if cleanup is not None:
            await cleanup
        else:
            await store.stop()


@pytest.mark.asyncio
async def test_compressed_write_error_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
    store = MemoryMapDatasetBackingStore(compress_only=True)
    await store.initialize()
    stream = store._stream_writer
    store._stream_writer = SimpleNamespace(
        write=Mock(side_effect=OSError("disk full")), close=stream.close
    )
    try:
        with pytest.raises(OSError, match="disk full"):
            await store.add_conversation("s", Conversation(session_id="s", turns=[]))
        assert not store._finalized
    finally:
        await store.stop()
