# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapters that convert third-party benchmark formats into graph ParsedGraph.

Each format adapter exposes a `from_X(path)` callable for direct use (the
native facade has none; it delegates to ``parser.parse_native``), and every
adapter class is registered under the `graph_adapter` plugin category.
"""

from aiperf.dataset.graph.adapters.dynamo.trace import DynamoTraceAdapter
from aiperf.dataset.graph.adapters.native import NativeGraphAdapter
from aiperf.dataset.graph.adapters.protocols import GraphAdapterProtocol
from aiperf.dataset.graph.adapters.weka.trace import WekaTraceAdapter

__all__ = [
    "DynamoTraceAdapter",
    "GraphAdapterProtocol",
    "NativeGraphAdapter",
    "WekaTraceAdapter",
]
