# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warmup graph rewriting and first-token anchoring for agent-graph replay.

Pure graph->graph transforms shared by ``AgentGraphReplayStrategy`` and
``GraphTracePlanner``. Kept in their own module so the planner can rewrite a
warmup graph without importing the strategy that owns it.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import msgspec

from aiperf.dataset.graph.models import START_NODE_ID, LlmNode, StaticEdge
from aiperf.graph.ids import chain_key

if TYPE_CHECKING:
    from aiperf.dataset.graph.models import (
        GraphRecord,
        ParsedGraph,
    )

__all__ = ["GraphWarmupKind", "first_token_sources", "rewrite_for_warmup"]


class GraphWarmupKind(str, Enum):
    """Which warmup transform a graph replay instance applies before dispatch.

    ``None`` (not this enum) means no warmup transform -- normal profiling
    dispatch, t*-chopped when a snapshot window is active.

    ``BOUNDARY_SNAPSHOT``: synthesized priming for a t*>0 snapshot run.
    The graph is DERIVED from a profiled trace by ``rewrite_for_warmup``,
    which keeps only each live chain's boundary turn. Empty at t*<=0 by
    design: with full recorded replay there is no pre-t* prefix to prime.

    ``RECORDED``: the corpus authored this graph as warmup
    (``ParsedGraph.warmup_traces``). It is already exactly what must go on the
    wire; running it through ``rewrite_for_warmup`` (the snapshot-derivation
    transform) would erase it, so it dispatches verbatim.
    """

    BOUNDARY_SNAPSHOT = "boundary_snapshot"
    RECORDED = "recorded"


def first_token_sources(graph: GraphRecord) -> frozenset[str]:
    """Source node ids of every first-token-anchored ``StaticEdge`` in ``graph``.

    A ``StaticEdge`` carrying ``delay_after_predecessor_first_token_us`` anchors
    its successor to the SOURCE node's observed first token (post-TTFT
    anchoring). That source must therefore emit a ``FirstToken`` --
    i.e. its issued turn carries ``first_token_event=True`` -- so the runtime can
    release the successor when the token arrives. Returns the (possibly empty)
    set of such source node ids; a gap-free / pre-anchoring graph yields the
    empty set (no node emits a first-token event).
    """
    return frozenset(
        edge.source
        for edge in graph.edges
        if edge.delay_after_predecessor_first_token_us is not None
    )


def _warmup_boundary_nodes(graph: GraphRecord, t_star_us: float) -> dict[str, LlmNode]:
    """Return ``{node_id: node}`` of each chain-live-at-``t*`` boundary turn.

    Chains are the per-session linear paths the trie node ids encode
    (:func:`~aiperf.graph.ids.chain_key`), ordered by recorded arrival. A chain is LIVE when it
    has BOTH a node arriving before ``t*`` and a node arriving at/after ``t*``;
    its boundary is the LAST pre-``t*`` node. Chains with no pre-``t*`` node
    need no priming (profiling replays them from their own start); chains
    entirely pre-``t*`` are not live (nothing of them is profiled).

    Only ``LlmNode`` steps are candidates. Warmup exists to prime the SERVER's
    KV cache, and a ``ToolNode`` issues no request -- priming one would run the
    recorded shell command a second time for no cache benefit, and its duration
    would land in a phase whose records are discarded.
    """
    chains: dict[str, list[tuple[int, str]]] = {}
    for nid, node in graph.nodes.items():
        if not isinstance(node, LlmNode):
            continue
        arrival = node.arrival_offset_us or 0
        chains.setdefault(chain_key(nid), []).append((arrival, nid))
    boundary: dict[str, LlmNode] = {}
    for members in chains.values():
        members.sort()
        pre = [nid for arrival, nid in members if arrival < t_star_us]
        if pre and any(arrival >= t_star_us for arrival, _ in members):
            boundary[pre[-1]] = graph.nodes[pre[-1]]
    return boundary


def rewrite_for_warmup(parsed: ParsedGraph, t_star_us: float) -> ParsedGraph:
    """Rewrite ``parsed`` into the WARMUP boundary-priming graph at ``t*``.

    AgentX-parity contract (``timing.config._build_graph_auto_warmup_config``):
    warmup dispatches exactly ONE priming credit per chain LIVE at ``t*`` --
    the chain's boundary turn, the last node of that per-session chain whose
    recorded arrival precedes ``t*`` (:func:`_warmup_boundary_nodes`). Because
    trie prompts are cumulative along a chain, priming the boundary turn's
    prompt (at the worker-side warmup ``max_tokens`` cap, keyed off the
    ``"warmup"`` phase variant) warms the chain's whole prefix.

    The produced graph is FLAT: only the boundary nodes survive, each re-rooted
    from ``START`` with NO leading offset (warmup bursts every priming credit
    at phase start rather than replaying recorded gaps) and with fan-in
    ``inputs`` cleared (their predecessors are gone). Node identity, the trie
    envelope, and ``dispatch_overrides`` are preserved so the worker resolves
    the unmodified catalog ordinal and materializes the exact recorded prompt.
    ``t_star_us <= 0`` (full recorded replay, or a zero-duration trace) yields an
    EMPTY graph so the warmup phase finalizes immediately.
    """
    graph = parsed.graph
    boundary = _warmup_boundary_nodes(graph, t_star_us) if t_star_us > 0 else {}
    new_nodes = {
        nid: msgspec.structs.replace(node, inputs=[], min_start_delay_us=None)
        for nid, node in boundary.items()
    }
    new_edges = [StaticEdge(source=START_NODE_ID, target=nid) for nid in new_nodes]
    new_graph = msgspec.structs.replace(graph, nodes=new_nodes, edges=new_edges)
    return msgspec.structs.replace(parsed, graph=new_graph)
