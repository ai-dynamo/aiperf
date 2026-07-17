# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TC1 — boundary-turn warmup (``rewrite_for_warmup``) + warmup phase routing.

The AgentX-parity auto-warmup contract (``timing.config``): the WARMUP phase
dispatches exactly ONE priming credit per chain live at t* -- the chain's
boundary turn (the last node recorded before t*) -- instead of replaying the
ENTIRE post-t* workload at ``max_tokens=1``. Chains are the per-session linear
paths the trie node ids encode (root chain + each subagent chain). These tests
pin:

* the pure rewrite on a two-chain (root + subagent) trace: exactly the two
  boundary nodes, START-rooted, zero leading offsets, fan-in inputs cleared;
* liveness: a chain entirely pre-t* is skipped; t*<=0 yields an EMPTY graph;
* the strategy's WARMUP phase variant dispatches ONLY the boundary nodes while
  the PROFILING variant at the same t* dispatches the post-t* chop set;
* a zero t* window warmup issues no credits and finalizes immediately.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
from aiperf.graph.analysis import trace_duration_us
from aiperf.timing.strategies.graph_ir_replay import (
    GraphIRReplayStrategy,
    rewrite_for_warmup,
)

_SUBAGENT_FIX = Path(__file__).parent / "fixtures" / "weka_subagent.json"


def _arrivals(parsed: Any) -> dict[str, int]:
    return {nid: node.arrival_offset_us for nid, node in parsed.graph.nodes.items()}


def _midpoint(parsed: Any, a: str, b: str) -> float:
    arr = _arrivals(parsed)
    return (arr[a] + arr[b]) / 2.0


# ---------------------------------------------------------------------------
# rewrite_for_warmup — pure rewrite
# ---------------------------------------------------------------------------


def test_two_live_chains_yield_exactly_the_two_boundary_nodes():
    """t* between the subagent's turns: root boundary trace_sub_n2s1:0 + subagent boundary agent_001:0.

    Chains on the subagent fixture: root = [trace_sub_n2s1:0, trace_sub_n2s1:1],
    subagent = [agent_001:0, agent_001:1]. With t* between agent_001:0 and
    agent_001:1 both chains are live (each has a pre-t* and a post-t* node), so
    warmup primes each chain's LAST pre-t* turn -- and nothing else.
    """
    parsed = from_weka_trace(str(_SUBAGENT_FIX))
    t_star_us = _midpoint(parsed, "agent_001:0", "agent_001:1")

    warmup = rewrite_for_warmup(parsed, t_star_us)

    assert set(warmup.graph.nodes) == {"trace_sub_n2s1:0", "agent_001:0"}
    # START-rooted with ZERO leading offsets (warmup bursts, never replays
    # recorded gaps): one plain START edge per boundary node, no delay fields.
    assert {(e.source, e.target) for e in warmup.graph.edges} == {
        ("START", "trace_sub_n2s1:0"),
        ("START", "agent_001:0"),
    }
    for edge in warmup.graph.edges:
        assert not edge.min_start_delay_us
        assert not edge.delay_after_predecessor_us
        assert not edge.delay_after_predecessor_start_us
        assert not edge.delay_after_predecessor_first_token_us
    for node in warmup.graph.nodes.values():
        assert node.inputs == []
        assert not node.min_start_delay_us


def test_chain_entirely_pre_tstar_is_not_live():
    """t* past the whole subagent chain: only the root chain is primed."""
    parsed = from_weka_trace(str(_SUBAGENT_FIX))
    t_star_us = _midpoint(parsed, "agent_001:1", "trace_sub_n2s1:1")

    warmup = rewrite_for_warmup(parsed, t_star_us)

    # Subagent chain [agent_001:0, agent_001:1] is entirely pre-t* (nothing of it
    # is profiled) -> no priming; root chain boundary is trace_sub_n2s1:0
    # (trace_sub_n2s1:1 is post-t*).
    assert set(warmup.graph.nodes) == {"trace_sub_n2s1:0"}


def test_boundary_nodes_keep_identity_and_dispatch_payload():
    """Boundary nodes keep ids, trie envelope, and dispatch body fields verbatim.

    The worker resolves the unmodified catalog ordinal from the node id and
    materializes the recorded prompt from the store; only ``inputs`` and the
    leading offset are cleared by the rewrite.
    """
    parsed = from_weka_trace(str(_SUBAGENT_FIX))
    t_star_us = _midpoint(parsed, "agent_001:0", "agent_001:1")

    warmup = rewrite_for_warmup(parsed, t_star_us)

    for nid, node in warmup.graph.nodes.items():
        original = parsed.graph.nodes[nid]
        assert node.metadata == original.metadata
        assert node.extra_body == original.extra_body
        assert node.model == original.model
        assert node.max_tokens == original.max_tokens
        assert node.arrival_offset_us == original.arrival_offset_us


@pytest.mark.parametrize("t_star_us", [0, -1.0])  # fmt: skip
def test_tstar_zero_yields_empty_warmup_graph(t_star_us):
    """t*<=0 (full native replay / zero-duration trace) => EMPTY warmup graph."""
    parsed = from_weka_trace(str(_SUBAGENT_FIX))
    warmup = rewrite_for_warmup(parsed, t_star_us)
    assert warmup.graph.nodes == {}
    assert warmup.graph.edges == []


# ---------------------------------------------------------------------------
# strategy warmup phase routing — dispatch sets
# ---------------------------------------------------------------------------


class _Config:
    """Minimal per-phase config stub (mirrors test_lane_fanout_recycle)."""

    timing_mode = None

    def __init__(self, *, phase: CreditPhase | None = None) -> None:
        self.phase = phase
        self.concurrency = None
        self.expected_num_sessions = None
        self.total_expected_requests = None
        self.expected_duration_sec = None


class _StubIssuer:
    """Issuer whose graph credits resolve immediately via the return observer."""

    def __init__(self) -> None:
        self.observer = None
        self.issued: list[Any] = []

    def bind(self, strategy: GraphIRReplayStrategy) -> None:
        self.observer = strategy._on_graph_return

    async def issue_graph_credit(self, turn: Any) -> bool:
        self.issued.append(turn)
        observer = self.observer
        loop = asyncio.get_running_loop()
        loop.call_soon(lambda: observer(turn, None, False))
        return True

    def mark_graph_sending_complete(self) -> None: ...

    def graph_all_returned(self) -> bool:
        return True

    def set_graph_all_returned_event(self) -> None: ...


def _strategy(
    parsed: Any, phase: CreditPhase | None, ratio: float
) -> tuple[GraphIRReplayStrategy, _StubIssuer]:
    issuer = _StubIssuer()
    strategy = GraphIRReplayStrategy(
        config=_Config(phase=phase),
        credit_issuer=issuer,
        parsed_graph=parsed,
        register_observer=lambda _obs: None,
        start_min_ratio=ratio,
        start_max_ratio=ratio,
    )
    issuer.bind(strategy)
    return strategy, issuer


def _issued_node_ids(issuer: _StubIssuer) -> set[str]:
    # Node id ``{scope}:{turn}`` is recovered from the credit's own legacy-shaped
    # identity: conversation_id (``{trace}`` root / ``{trace}::{scope}`` child)
    # + turn_index (the node's 0-based turn).
    node_ids = set()
    for turn in issuer.issued:
        trace, sep, scope = turn.conversation_id.partition("::")
        node_ids.add(f"{scope if sep else trace}:{turn.turn_index}")
    return node_ids


@pytest.mark.asyncio
async def test_warmup_phase_dispatches_only_boundary_turns():
    """WARMUP at a two-live-chain t* dispatches exactly {trace_sub_n2s1:0, agent_001:0} at 'warmup'."""
    parsed = from_weka_trace(str(_SUBAGENT_FIX))
    duration = trace_duration_us(parsed, parsed.traces[0])
    ratio = _midpoint(parsed, "agent_001:0", "agent_001:1") / duration

    strategy, issuer = _strategy(parsed, CreditPhase.WARMUP, ratio)
    await strategy.execute_phase()

    assert _issued_node_ids(issuer) == {"trace_sub_n2s1:0", "agent_001:0"}
    assert all(turn.phase_variant == "warmup" for turn in issuer.issued)
    assert strategy.completed_traces == 1
    assert strategy.errored_traces == 0


@pytest.mark.asyncio
async def test_profiling_phase_at_same_tstar_dispatches_post_tstar_set():
    """Control: PROFILING at the same t* runs the chop survivors, not boundaries."""
    parsed = from_weka_trace(str(_SUBAGENT_FIX))
    duration = trace_duration_us(parsed, parsed.traces[0])
    ratio = _midpoint(parsed, "agent_001:0", "agent_001:1") / duration

    strategy, issuer = _strategy(parsed, CreditPhase.PROFILING, ratio)
    await strategy.execute_phase()

    assert _issued_node_ids(issuer) == {"agent_001:1", "trace_sub_n2s1:1"}
    assert all(turn.phase_variant == "profiling" for turn in issuer.issued)


@pytest.mark.asyncio
async def test_warmup_with_zero_window_issues_nothing_and_finalizes():
    """t*=0: empty warmup graph -- no credit issued, phase completes cleanly."""
    parsed = from_weka_trace(str(_SUBAGENT_FIX))
    strategy, issuer = _strategy(parsed, CreditPhase.WARMUP, 0.0)

    await asyncio.wait_for(strategy.execute_phase(), timeout=5.0)

    assert issuer.issued == []
    assert strategy.completed_traces == strategy.admitted_traces
    assert strategy.errored_traces == 0
