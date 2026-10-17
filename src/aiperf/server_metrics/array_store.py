# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disk-backed NumPy storage for exact server-metrics histories."""

from __future__ import annotations

import asyncio
import mmap
import os
import tempfile
import weakref
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import BinaryIO, ParamSpec, TypeVar

import numpy as np
from numpy.typing import DTypeLike, NDArray

from aiperf.common.constants import IS_LINUX

_P = ParamSpec("_P")
_R = TypeVar("_R")


async def run_history_io(
    operation: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs
) -> _R:
    """Finish an in-flight history operation before propagating cancellation.

    Cancelling asyncio.to_thread cannot stop its worker. Releasing the history
    lock early would let export or close race a still-running array write.
    """
    task = asyncio.create_task(asyncio.to_thread(operation, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        task.result()
        raise


class ServerMetricsStorageError(RuntimeError):
    """Collection exhausted its configured storage or series budget."""


class ServerMetricsArrayStore:
    """Allocate numeric histories in one anonymous, disk-backed file.

    The file and mappings grow geometrically, avoiding a descriptor per array.
    Old views remain valid when a larger mapping is added. Flushes release
    clean pages so the collection working set does not grow with benchmark
    duration. Existing NumPy statistics retain all samples.

    The file lives on the artifact filesystem, rather than the system temp
    directory, which is often a memory-backed tmpfs inside containers. It has
    no directory entry on Linux and cannot be mistaken for an export artifact.
    """

    def __init__(self, directory: Path, *, max_bytes: int, flush_bytes: int) -> None:
        self._directory = directory
        self.max_bytes = max_bytes
        self.flush_bytes = flush_bytes
        self.allocated_bytes = 0
        self.dirty_bytes = 0
        self._file: BinaryIO | None = None
        self._mapping: mmap.mmap | None = None
        self._mappings: list[mmap.mmap] = []
        self._cleanup: weakref.finalize | None = None

    @property
    def initialized(self) -> bool:
        """Whether the backing mapping is available for allocation."""
        return self._mapping is not None

    def open(self) -> None:
        """Create the sparse backing file; call through asyncio.to_thread."""
        self._directory.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            file = stack.enter_context(tempfile.TemporaryFile(dir=self._directory))
            capacity = min(self.max_bytes, self.flush_bytes)
            file.truncate(capacity)
            mapping = mmap.mmap(file.fileno(), capacity)
            self._file = file
            self._mapping = mapping
            self._mappings.append(mapping)
            self._cleanup = weakref.finalize(self, stack.pop_all().close)

    def allocate(self, shape: int | tuple[int, ...], dtype: DTypeLike) -> NDArray:
        """Allocate an aligned array, failing before the disk budget is exceeded."""
        if self._mapping is None:
            raise ServerMetricsStorageError("Server metrics storage is not open")
        dtype = np.dtype(dtype)
        shape = (shape,) if isinstance(shape, int) else shape
        size = int(np.prod(shape)) * dtype.itemsize
        offset = (self.allocated_bytes + 63) // 64 * 64
        if offset + size > self.max_bytes:
            raise ServerMetricsStorageError(
                f"Server metrics history exceeded its {self.max_bytes}-byte disk "
                "budget; increase AIPERF_SERVER_METRICS_STORAGE_MAX_BYTES. "
                "Server metrics collection is incomplete; inference results are retained."
            )
        if self.dirty_bytes + size >= self.flush_bytes:
            self.flush()
        if offset + size > len(self._mapping):
            capacity = min(self.max_bytes, max(offset + size, 2 * len(self._mapping)))
            self._file.truncate(capacity)
            self._mapping = mmap.mmap(self._file.fileno(), capacity)
            self._mappings.append(self._mapping)
        array = np.ndarray(shape, dtype=dtype, buffer=self._mapping, offset=offset)
        self.allocated_bytes = offset + size
        self.dirty_bytes += size
        return array

    def note_write(self, size: int) -> None:
        """Account for appended samples between page-cache releases."""
        self.dirty_bytes += size

    def flush(self) -> None:
        """Persist dirty pages and release the cache; call through asyncio.to_thread."""
        if self._mapping is None or self._file is None:
            return
        self._mapping.flush()
        if IS_LINUX:
            for mapping in self._mappings:
                mapping.madvise(mmap.MADV_DONTNEED)
            os.posix_fadvise(self._file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        self.dirty_bytes = 0

    def close(self) -> None:
        """Release storage after consumers finish; call through asyncio.to_thread."""
        try:
            self.flush()
        finally:
            if self._cleanup is not None:
                self._cleanup()
            self._file = None
            # Array views keep the mapping valid until their last reference is
            # released; forcibly closing it would invalidate exported views.
            self._mapping = None
            self._mappings = []
