# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamo trace-format loader (``dynamo.request.trace.v1``).

The public entry points (and the chain-grouping / session-tree build) live in
:mod:`.trace`; the remaining modules are the internal reader
(:mod:`.trace_reader`), trie-lowering (:mod:`.trie_lowering`), fused-parallel
build (:mod:`.trace_parallel`), and segment-pool shim (:mod:`.store_backed_pool`)
stages.
"""

from aiperf.dataset.graph.adapters.dynamo.trace import (
    DynamoTraceAdapter,
    from_dynamo_trace,
)

__all__ = ["DynamoTraceAdapter", "from_dynamo_trace"]
