# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concurrency and resource-lifetime tests for MemoryMapDatasetClient.

``get_conversation`` used to read through the mmap's shared file position
(``seek()`` then ``read()``). Any two readers sharing one client could
interleave those calls and serve each other's bytes, surfacing either as a
spurious ``MemoryMapSerializationError`` on a random conversation or, when the
foreign slice happened to parse, as silently wrong data.

The reads are now position-free slices, so no reader depends on the shared
position.
"""

import asyncio
import mmap
import threading
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from aiperf.common.enums import MemoryMapFormat
from aiperf.common.environment import Environment
from aiperf.common.models import Conversation, MemoryMapClientMetadata, Text, Turn
from aiperf.dataset.memory_map_utils import (
    MemoryMapDatasetBackingStore,
    MemoryMapDatasetClient,
    MemoryMapDatasetClientStore,
    MemoryMapDatasetIndex,
)

# Wildly varying sizes so a cross-read lands mid-record and is caught either as
# a decode error or as a mismatched session_id.
_CONVERSATION_COUNT = 24
_THREADS = 8
_LOOKUPS_PER_THREAD = 150


def _make_conversation(index: int) -> Conversation:
    """Conversation whose size grows with ``index`` and whose text encodes it."""
    text = f"conv-{index}:" + ("x" * (64 * (index + 1)))
    return Conversation(
        session_id=f"conv-{index}",
        turns=[Turn(role="user", texts=[Text(contents=[text])])],
    )


@asynccontextmanager
async def _open_client(
    tmp_path: Path, benchmark_id: str, count: int
) -> AsyncIterator[MemoryMapDatasetClient]:
    """Write ``count`` conversations and open a client over them.

    The backing store is stopped on exit, so its ``@on_stop`` cleanup unlinks
    the mmap files instead of leaking them for the run's lifetime. The client
    is closed first: the store's cleanup deletes the files it maps.
    """
    store = MemoryMapDatasetBackingStore(
        benchmark_id=benchmark_id, format=MemoryMapFormat.CONVERSATION
    )
    await store.initialize()
    for i in range(count):
        await store.add_conversation(f"conv-{i}", _make_conversation(i))
    await store.finalize()

    metadata = store.get_client_metadata()
    client: MemoryMapDatasetClient | None = None
    try:
        # Constructed inside the guard: if opening the mmap raises, the store
        # still gets stopped and its files still get unlinked.
        client = MemoryMapDatasetClient(
            metadata.data_file_path,
            metadata.index_file_path,
        )
        yield client
    finally:
        if client is not None:
            client.close()
        await store.stop()


@pytest.mark.asyncio
async def test_get_conversation_concurrent_readers_returns_own_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent get_conversation() calls must not read each other's bytes."""
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
    async with _open_client(
        tmp_path, "test_conv_concurrency", _CONVERSATION_COUNT
    ) as client:

        def _read_many(thread_index: int) -> None:
            for n in range(_LOOKUPS_PER_THREAD):
                # Stagger the starting id per thread so the interleaving covers
                # many (offset, size) pairs rather than repeatedly hitting one
                # record.
                i = (thread_index + n) % _CONVERSATION_COUNT
                conv = client.get_conversation(f"conv-{i}")
                assert conv.session_id == f"conv-{i}"

        with ThreadPoolExecutor(max_workers=_THREADS) as pool:
            # list() resolves every future so exceptions propagate.
            list(pool.map(_read_many, range(_THREADS)))


class _InterleavingMmap:
    """mmap wrapper that moves the shared position between seek() and read().

    This is the race, made deterministic. Merely leaving a stale position
    behind proves nothing: a ``seek()`` + ``read()`` reader overwrites it on
    the way in and still returns the right record. The bug needs a competing
    seek to land *after* this reader seeks and *before* it reads.

    A position-free reader goes through ``__getitem__`` and is unaffected.
    """

    def __init__(self, real: mmap.mmap, foreign_offset: int) -> None:
        self._real = real
        self._foreign_offset = foreign_offset

    def seek(self, pos: int) -> None:
        self._real.seek(pos)

    def read(self, size: int) -> bytes:
        # The competing reader lands here, between our seek and our read.
        self._real.seek(self._foreign_offset)
        return self._real.read(size)

    def __getitem__(self, item: slice | int) -> bytes | int:
        return self._real[item]

    def __len__(self) -> int:
        return len(self._real)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


@pytest.mark.asyncio
async def test_get_conversation_ignores_competing_seek_between_seek_and_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interleaved foreign seek must not change what get_conversation reads.

    Fails against a ``seek()`` + ``read()`` implementation, which serves the
    foreign record's bytes.
    """
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
    async with _open_client(tmp_path, "test_conv_position", 3) as client:
        foreign = client.index.offsets["conv-2"].offset
        client.data_mmap = _InterleavingMmap(client.data_mmap, foreign)

        conversation = client.get_conversation("conv-0")

        assert conversation.session_id == "conv-0"
        assert conversation.turns[0].texts[0].contents[0].startswith("conv-0:")


@pytest.mark.asyncio
@pytest.mark.parametrize("prefault", [True, False])  # fmt: skip
async def test_get_conversation_matches_written_data_under_prefault_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefault: bool
) -> None:
    """Reads are identical whether or not pages were prefaulted at open."""
    monkeypatch.setenv("AIPERF_DATASET_MMAP_BASE_PATH", str(tmp_path))
    monkeypatch.setattr(
        "aiperf.common.environment.Environment.DATASET.MMAP_PREFAULT", prefault
    )
    async with _open_client(tmp_path, f"test_conv_prefault_{prefault}", 5) as client:
        for i in range(5):
            conv = client.get_conversation(f"conv-{i}")
            assert conv.session_id == f"conv-{i}"
            assert conv.turns[0].texts[0].contents[0].startswith(f"conv-{i}:")


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])  # fmt: skip
async def test_client_setup_keeps_event_loop_live_during_prefault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    """A stalled page preload must allow callbacks and cancellation to run."""
    data_path = tmp_path / "data"
    index_path = tmp_path / "index"
    data_path.write_bytes(b"x")
    index_path.write_text(MemoryMapDatasetIndex().model_dump_json())
    store = MemoryMapDatasetClientStore(
        client_metadata=MemoryMapClientMetadata(
            data_file_path=data_path,
            index_file_path=index_path,
            conversation_count=0,
            total_size_bytes=1,
        )
    )
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    entered = asyncio.Event()
    release = threading.Event()
    closed = threading.Event()
    resources: list[Any] = []
    cleanup = MemoryMapDatasetClient._cleanup_finalizer

    def slow_prefault(client: MemoryMapDatasetClient) -> None:
        assert threading.get_ident() != loop_thread
        resources.extend(
            [client.data_mmap, client.index_mmap, client.data_file, client.index_file]
        )
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(timeout=5), "event loop did not release the preload"

    def record_cleanup(*args: Any) -> None:
        cleanup(*args)
        closed.set()

    monkeypatch.setattr(Environment.DATASET, "MMAP_PREFAULT", True)
    monkeypatch.setattr(MemoryMapDatasetClient, "_prefault_data_mmap", slow_prefault)
    monkeypatch.setattr(
        MemoryMapDatasetClient, "_cleanup_finalizer", staticmethod(record_cleanup)
    )
    setup = asyncio.create_task(store._setup())
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert not setup.done()
        assert store._client is None
        if cancel:
            setup.cancel()
            with pytest.raises(asyncio.CancelledError):
                await setup
            await store._cleanup()
            release.set()
            assert await asyncio.to_thread(closed.wait, 5)
            assert store._client is None
        else:
            release.set()
            await asyncio.wait_for(setup, timeout=5)
            assert store._client is not None
            await store._cleanup()
        assert all(resource.closed for resource in resources)
    finally:
        release.set()
        await asyncio.gather(setup, return_exceptions=True)
        await store._cleanup()
