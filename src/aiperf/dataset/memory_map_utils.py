# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory-mapped dataset for zero-copy conversation access.

Eliminates the DatasetManager network bottleneck at high QPS by letting workers
read conversations directly from shared files in O(1) time.

Flow (local):
    1. DatasetManager writes conversations to disk via MemoryMapDatasetBackingStore
    2. Workers read via mmap (zero-copy) through MemoryMapDatasetClientStore

Flow (Kubernetes):
    1. DatasetManager streams conversations to zstd-compressed files (compress_only mode)
    2. WorkerGroupManager downloads compressed files once per pod from control-plane via HTTP API
    3. WorkerGroupManager decompresses files locally
    4. Workers read via mmap through MemoryMapDatasetClientStore

Implementation is split across sibling modules; this module re-exports the
public surface for backward compatibility.
"""

from aiperf.dataset.memory_map_client import (
    MemoryMapDatasetClient,
    MemoryMapDatasetClientStore,
)
from aiperf.dataset.memory_map_models import (
    ConversationOffset,
    MemoryMapDatasetIndex,
)
from aiperf.dataset.memory_map_store import MemoryMapDatasetBackingStore

__all__ = [
    "ConversationOffset",
    "MemoryMapDatasetBackingStore",
    "MemoryMapDatasetClient",
    "MemoryMapDatasetClientStore",
    "MemoryMapDatasetIndex",
]
