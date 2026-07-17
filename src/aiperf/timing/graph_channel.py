# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""First-class carrier for graph-phase state shared across phase runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiperf.dataset.graph.models import ParsedGraph
    from aiperf.timing.graph_warmup_handoff import GraphWarmupHandoff


@dataclass(slots=True)
class GraphPhaseChannel:
    """Graph-run state threaded orchestrator -> runner -> strategy.

    Replaces the former attribute smuggling on the generic
    ``ConversationSource`` (``parsed_graph`` / ``graph_warmup_handoff``):
    graph phases receive this typed channel instead of a conversation-shaped
    object they never sample from.
    """

    parsed_graph: ParsedGraph
    """The structural conversation DAG the graph strategies plan from."""

    warmup_handoff: GraphWarmupHandoff | None = None
    """Consume-once WARMUP -> PROFILING handoff; the WARMUP strategy stashes
    it at teardown and the first PROFILING graph phase pops it."""
