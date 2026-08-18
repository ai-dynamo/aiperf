# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the mmap shared-position race fixed by absolute slicing.

This experiment builds a real MemoryMapDatasetBackingStore, then wraps the
client mmap object with a tiny shim that forces two concurrent readers to
interleave after ``seek()`` and before ``read()``.

Expected output:

    old seek/read: FAIL (expected)
    fixed slicing: PASS

The old path is intentionally implemented in this file to match the previous
MemoryMapDatasetClient.get_conversation() behavior. The fixed path calls the
current production method.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aiperf.common.models import Conversation, Turn
from aiperf.dataset.memory_map_utils import (
    MemoryMapDatasetBackingStore,
    MemoryMapDatasetClient,
)

ConversationReader = Callable[[MemoryMapDatasetClient, str], Conversation]


class InterleavingMmap:
    """Force both readers to share one mutable cursor at the phase boundary."""

    def __init__(self, base_mmap) -> None:
        self._base_mmap = base_mmap
        self._position = 0
        self._barrier = threading.Barrier(2)
        self._lock = threading.Lock()

    def __getitem__(self, key: int | slice) -> int | bytes:
        return self._base_mmap[key]

    def seek(self, offset: int) -> None:
        with self._lock:
            self._position = offset
        self._barrier.wait(timeout=5)

    def read(self, size: int) -> bytes:
        with self._lock:
            start = self._position
            self._position += size
        return self._base_mmap[start : start + size]

    def close(self) -> None:
        self._base_mmap.close()


async def _build_client(base_path: Path, benchmark_id: str) -> MemoryMapDatasetClient:
    os.environ["AIPERF_DATASET_MMAP_BASE_PATH"] = str(base_path)

    store = MemoryMapDatasetBackingStore(benchmark_id=benchmark_id)
    await store.initialize()
    for conversation_id, content in (("conv-a", "a" * 4096), ("conv-b", "b" * 4096)):
        await store.add_conversation(
            conversation_id,
            Conversation(
                session_id=conversation_id,
                turns=[Turn(role="user", content=content)],
            ),
        )
    await store.finalize()

    metadata = store.get_client_metadata()
    client = MemoryMapDatasetClient(metadata.data_file_path, metadata.index_file_path)
    await store.stop()
    return client


def _old_seek_read(client: MemoryMapDatasetClient, conversation_id: str) -> Conversation:
    """Previous implementation: a shared seek cursor followed by read()."""
    offset_info = client.index.offsets[conversation_id]
    client.data_mmap.seek(offset_info.offset)
    conv_bytes = client.data_mmap.read(offset_info.size)
    return client._deserialize_conversation(conv_bytes)


def _fixed_slice_read(
    client: MemoryMapDatasetClient, conversation_id: str
) -> Conversation:
    return client.get_conversation(conversation_id)


def _run_pair(client: MemoryMapDatasetClient, reader: ConversationReader) -> list[str]:
    requested = ("conv-a", "conv-b")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(reader, client, conversation_id)
            for conversation_id in requested
        ]

    results: list[str] = []
    for future in futures:
        try:
            results.append(future.result().session_id)
        except Exception as exc:  # noqa: BLE001 - show the observable failure mode
            results.append(f"{type(exc).__name__}: {exc}")
    return results


async def _main() -> int:
    expected = ["conv-a", "conv-b"]
    with tempfile.TemporaryDirectory(prefix="aiperf-mmap-race-") as tmp:
        base_path = Path(tmp)

        old_client = await _build_client(base_path, "old")
        old_base_mmap = old_client.data_mmap
        old_client.data_mmap = InterleavingMmap(old_base_mmap)
        old_results = _run_pair(old_client, _old_seek_read)
        old_client.close()

        fixed_client = await _build_client(base_path, "fixed")
        fixed_base_mmap = fixed_client.data_mmap
        fixed_client.data_mmap = InterleavingMmap(fixed_base_mmap)
        fixed_results = _run_pair(fixed_client, _fixed_slice_read)
        fixed_client.close()

    old_failed = old_results != expected
    fixed_passed = fixed_results == expected

    print(f"requested:      {expected}")
    print(f"old seek/read:  {old_results} -> {'FAIL (expected)' if old_failed else 'PASS'}")
    print(f"fixed slicing:  {fixed_results} -> {'PASS' if fixed_passed else 'FAIL'}")

    if old_failed and fixed_passed:
        print("\nReproduced: shared mmap cursor reads can corrupt concurrent conversations.")
        return 0

    print("\nUnexpected result: experiment did not distinguish old and fixed behavior.")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
