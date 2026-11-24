# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disk-backed NumPy storage for exact server-metrics histories."""

from __future__ import annotations

import asyncio
import errno
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
        self._mapping_addresses: dict[mmap.mmap, int] = {}
        self._dirty_ranges: dict[mmap.mmap, dict[int, tuple[int, int]]] = {}

    @property
    def initialized(self) -> bool:
        """Whether the backing mapping is available for allocation."""
        return self._mapping is not None

    def open(self) -> None:
        """Reserve backing storage before mapping it; call through asyncio.to_thread."""
        self._directory.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            file = stack.enter_context(tempfile.TemporaryFile(dir=self._directory))
            capacity = min(self.max_bytes, self.flush_bytes)
            mapping = self._grow_mapping(file, 0, capacity)
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
        if offset + size > len(self._mapping):
            capacity = min(self.max_bytes, max(offset + size, 2 * len(self._mapping)))
            self._mapping = self._grow_mapping(self._file, len(self._mapping), capacity)
            self._mappings.append(self._mapping)
        array = np.ndarray(shape, dtype=dtype, buffer=self._mapping, offset=offset)
        self.allocated_bytes = offset + size
        self.note_write(array)
        return array

    def _grow_mapping(self, file: BinaryIO, offset: int, capacity: int) -> mmap.mmap:
        """Reserve the new extent before any mapped write can fault on ENOSPC."""
        try:
            self._reserve_extent(file, offset, capacity - offset)
            mapping = mmap.mmap(file.fileno(), capacity)
        except OSError as error:
            raise ServerMetricsStorageError(
                f"Cannot reserve server metrics history in {self._directory} "
                f"up to {capacity} bytes: {error}. Free artifact disk space. "
                "Server metrics collection is incomplete; inference results are retained."
            ) from error
        self._mapping_addresses[mapping] = np.frombuffer(
            mapping, dtype=np.uint8, count=1
        ).ctypes.data
        return mapping

    @staticmethod
    def _reserve_extent(file: BinaryIO, offset: int, size: int) -> None:
        """Use physical allocation, with bounded writes when fallocate is unavailable."""
        if IS_LINUX:
            try:
                os.posix_fallocate(file.fileno(), offset, size)
                return
            except OSError as error:
                if error.errno not in (errno.EINVAL, errno.EOPNOTSUPP, errno.ENOSYS):
                    raise
        file.seek(offset)
        block = bytes(min(size, 1024 * 1024))
        remaining = size
        while remaining:
            written = file.write(block[:remaining])
            if not written:
                raise OSError(errno.EIO, "No progress reserving server metrics storage")
            remaining -= written
        file.flush()
        os.fsync(file.fileno())

    def note_write(self, array: NDArray) -> None:
        """Track contiguous modified views and copy sources for bounded cache release.

        Copy sources are included so growing a buffer also releases the pages
        read through its old mapping. No flush occurs until the write finishes.
        """
        if not array.size:
            return
        root = array
        while isinstance(root.base, np.ndarray):
            root = root.base
        mapping = root.base
        if not isinstance(mapping, mmap.mmap):
            return
        start = array.ctypes.data - self._mapping_addresses[mapping]
        end = start + array.nbytes
        ranges = self._dirty_ranges.setdefault(mapping, {})
        key = root.ctypes.data
        previous = ranges.get(key)
        ranges[key] = (
            (min(start, previous[0]), max(end, previous[1]))
            if previous is not None
            else (start, end)
        )
        self.dirty_bytes += array.nbytes

    def flush(self) -> None:
        """Flush and evict touched page ranges; call through asyncio.to_thread."""
        if self._file is None:
            return
        alignment = mmap.ALLOCATIONGRANULARITY
        for mapping, ranges in self._dirty_ranges.items():
            start, end = 0, 0
            for low, high in sorted(ranges.values()):
                low = low // alignment * alignment
                high = min(
                    len(mapping), (high + alignment - 1) // alignment * alignment
                )
                if low > end and end > start:
                    self._flush_range(mapping, start, end)
                    start = low
                elif end == start:
                    start = low
                end = max(end, high)
            if end > start:
                self._flush_range(mapping, start, end)
        self._dirty_ranges.clear()
        self.dirty_bytes = 0

    def _flush_range(self, mapping: mmap.mmap, start: int, end: int) -> None:
        """Release one merged range without revisiting clean historical mappings."""
        mapping.flush(start, end - start)
        if IS_LINUX:
            mapping.madvise(mmap.MADV_DONTNEED, start, end - start)
            os.posix_fadvise(
                self._file.fileno(), start, end - start, os.POSIX_FADV_DONTNEED
            )

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
            self._mapping_addresses.clear()
            self._dirty_ranges.clear()
