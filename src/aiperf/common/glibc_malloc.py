# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""glibc malloc tuning for long-lived services. Every call is a no-op off Linux."""

import ctypes

from aiperf.common.constants import IS_LINUX

M_TRIM_THRESHOLD = -1
M_MMAP_THRESHOLD = -3
PINNED_THRESHOLD_BYTES = 128 * 1024


def _libc() -> ctypes.CDLL | None:
    if not IS_LINUX:
        return None
    try:
        return ctypes.CDLL("libc.so.6")
    except OSError:
        return None


def pin_malloc_thresholds(threshold: int = PINNED_THRESHOLD_BYTES) -> bool:
    """Serve allocations of ``threshold`` bytes and up with mmap, trim the heap at the same size.

    glibc's mmap threshold is dynamic: it grows to the largest chunk ever freed,
    after which multi-hundred-KB request payloads and conversation strings are
    carved from the main arena and their freed chunks stay resident. Pinning it
    returns each such block to the kernel on free. Must run in the process that
    allocates: a forked child inherits the parent's already initialised malloc
    state, so ``MALLOC_MMAP_THRESHOLD_`` in the environment does not reach it.
    """
    libc = _libc()
    if libc is None:
        return False
    return bool(libc.mallopt(M_MMAP_THRESHOLD, threshold)) and bool(
        libc.mallopt(M_TRIM_THRESHOLD, threshold)
    )


def trim_heap() -> bool:
    """Return freed arena memory to the kernel (``malloc_trim(0)``)."""
    libc = _libc()
    if libc is None:
        return False
    return bool(libc.malloc_trim(0))
