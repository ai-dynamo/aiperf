# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tool-calling seam, shaped like the executor's LLM dispatch seam.

`TraceExecutor` injects its LLM caller (`credit_issuer`) rather than importing
one, so the dataflow plane never depends on how a request reaches a server. Tool
calling follows the same rule: the executor holds a `ToolDispatcher` and calls
`dispatch` on it, so what "running a tool step" means is a swap, not an edit.

Implementations this shape anticipates:
  * sandbox-backed (shipped) -- run the recorded commands locally or in a container;
  * live-agentic -- read tool calls from the producing node's captured reply;
  * recorded-delay -- sleep the recorded duration and execute nothing;
  * remote -- ship the step to an agent running elsewhere.

None of those require touching `_execute_tool`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from aiperf.dataset.graph.models import ToolNode
from aiperf.graph.placement import PlacementContext

__all__ = ["ToolDispatchRequest", "ToolDispatchResult", "ToolDispatcher"]


@dataclass(slots=True, frozen=True)
class ToolDispatchRequest:
    """What the executor hands the dispatcher for one tool step."""

    node_id: str
    """The firing node's id, mirroring `DispatchRequest` on the LLM path."""


@dataclass(slots=True, frozen=True)
class ToolDispatchResult:
    """What a dispatcher returns for one tool step."""

    observation: str
    """Text written to the node's output channel."""
    durations_s: list[float] = field(default_factory=list)
    """One entry per executed command, in execution order. The tool-time series
    is built from these; they never enter any request-latency metric."""
    timed_out: bool = False
    """Whether any command in the step hit its ceiling."""


class ToolDispatcher(Protocol):
    """Executes one graph tool step, however it chooses to."""

    async def open_trace(self, trace_id: str) -> None:
        """Prepare per-trace resources. Called outside the measured window."""
        ...

    async def dispatch(
        self,
        node: ToolNode,
        request: ToolDispatchRequest,
        ctx: PlacementContext,
    ) -> ToolDispatchResult:
        """Run one tool step.

        Must not raise on a command's nonzero exit or timeout -- those are
        recorded outcomes carried in the result. Raise only when the step could
        not be attempted at all.
        """
        ...

    async def close_trace(self, trace_id: str) -> None:
        """Release per-trace resources. Idempotent; outside the measured window."""
        ...
