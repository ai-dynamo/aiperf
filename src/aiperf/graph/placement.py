# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dispatch payload/context carriers for the graph runtime.

`DispatchRequest` / `PlacementContext` carry the per-call payload and runtime
context that the live dispatch path (`graph.dispatch.llm`) threads into the
credit issuer.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DispatchRequest",
    "PlacementContext",
]


@dataclass(slots=True)
class DispatchRequest:
    """One dispatch: the fired node id.

    The credit path resolves a node's actual request worker-side from the
    recorded envelope keyed by ``node_id``.
    """

    node_id: str


@dataclass(slots=True)
class PlacementContext:
    """Per-dispatch routing context the executor stamps for the credit path.

    Populated by the executor; not authorable. The credit adapter reads only
    ``parent_trace_id`` (the per-trace instance id); ``parent_node_id`` rides
    along as the fired node's identity.
    """

    parent_trace_id: str | None = None
    parent_node_id: str | None = None
