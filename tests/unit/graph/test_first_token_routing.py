# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task 4 — post-TTFT first-token routing: strategy -> adapter -> dispatch cb.

Locks the delivery chain wired in Task 4:

* ``first_token_sources(graph)`` derives the set of node ids that SOURCE a
  first-token-anchored ``StaticEdge`` (``delay_after_predecessor_first_token_us``
  set), from the same per-trace graph the adapter dispatches.
* ``CreditDispatchAdapter`` stamps ``TurnToSend.first_token_event`` for those
  source nodes, parks an optional per-dispatch ``first_token_cb`` under the
  waiter key, and fires it AT MOST ONCE from ``on_first_token`` (unknown / None
  keys are graceful no-ops).
* ``GraphIRReplayStrategy._on_graph_first_token`` de-muxes each ``FirstToken`` to
  the owning adapter by ``trace_id`` (mirroring ``_on_graph_return``), tolerating
  an unknown / None trace id.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from aiperf.credit.messages import FirstToken
from aiperf.dataset.graph.adapters.weka.trace_models import WekaTrace
from aiperf.dataset.graph.adapters.weka.trie_build import (
    ReconCallbacks,
    build_trie_graph,
)
from aiperf.dataset.graph.graph_path_catalog import CatalogContext
from aiperf.dataset.graph.models import GraphRecord, LlmNode, StaticEdge
from aiperf.graph.credit_dispatch_adapter import CreditDispatchAdapter
from aiperf.graph.placement import DispatchRequest, PlacementContext
from aiperf.timing.strategies.graph_ir_replay import (
    GraphIRReplayStrategy,
    first_token_sources,
)

_BLOCK_SIZE = 64
_WEKA_MIN = Path(__file__).parent / "fixtures" / "weka_min.json"


# ---------------------------------------------------------------------------
# first_token_sources(graph) — the pure derivation
# ---------------------------------------------------------------------------


def _stub_decode_block_tokens(hash_ids: list[int]) -> list[int]:
    out: list[int] = []
    for h in hash_ids:
        out.extend(range(h * 100, h * 100 + _BLOCK_SIZE))
    return out


def _stub_partial_tail_tokens(n_tokens: int, seed: str) -> list[int]:
    base = sum(ord(c) for c in seed) * 1000
    return list(range(base, base + n_tokens))


def _stub_decode_tokens_to_text(tokens: list[int]) -> str:
    return "|".join(str(t) for t in tokens)


_STUB_CALLBACKS = ReconCallbacks(
    decode_block_tokens=_stub_decode_block_tokens,
    sample_partial_tail_tokens=_stub_partial_tail_tokens,
    decode_tokens_to_text=_stub_decode_tokens_to_text,
)

# ttft_anchor:0: streaming parent (ttft 2.0) whose completed subagent's first inner
# request starts at t=4.0 -- inside the parent's [0, 8) interval and at/after its
# first token, so the lowered edge carries delay_after_predecessor_first_token_us.
_TTFT_TRACE = {
    "id": "ttft_anchor", "models": ["M"], "block_size": 64,
    "hash_id_scope": "local",
    "requests": [
        {"t": 0.0, "type": "s", "model": "M", "in": 128, "out": 64,
         "hash_ids": [1, 2], "api_time": 8.0, "ttft": 2.0, "stop": "tool_use"},
        {"t": 2.0, "type": "subagent", "agent_id": "a1",
         "subagent_type": "Explore", "status": "completed", "models": ["M"],
         "requests": [
             {"t": 4.0, "type": "n", "model": "M", "in": 128, "out": 16,
              "hash_ids": [50, 51], "api_time": 1.0},
         ]},
    ],
}  # fmt: skip


def test_first_token_sources_computed_from_parsed_graph():
    """The post-TTFT weka trace lowers exactly one first-token-anchored edge,
    sourced at ``ttft_anchor:0`` -- so the derivation returns ``{"ttft_anchor:0"}``."""
    parsed, _pool = build_trie_graph(
        WekaTrace.model_validate(_TTFT_TRACE), callbacks=_STUB_CALLBACKS
    )
    assert first_token_sources(parsed.graph) == frozenset({"ttft_anchor:0"})


def test_first_token_sources_excludes_plain_and_start_only_edges():
    """Only edges carrying ``delay_after_predecessor_first_token_us`` count; a
    plain delay edge and a start-anchored-only edge are excluded."""
    nodes = {
        "A": LlmNode(prompt=["@a"], output="a"),
        "B": LlmNode(prompt=["@b"], output="b"),
        "C": LlmNode(prompt=["@c"], output="c"),
        "D": LlmNode(prompt=["@d"], output="d"),
    }
    edges = [
        StaticEdge(source="START", target="A"),
        StaticEdge(source="A", target="B", delay_after_predecessor_us=3.0e6),
        StaticEdge(source="B", target="C", delay_after_predecessor_start_us=4.0e6),
        StaticEdge(
            source="C",
            target="D",
            delay_after_predecessor_start_us=4.0e6,
            delay_after_predecessor_first_token_us=2.0e6,
        ),
    ]
    graph = GraphRecord(nodes=nodes, edges=edges, state={})
    assert first_token_sources(graph) == frozenset({"C"})


def test_first_token_sources_empty_for_gap_free_graph():
    graph = GraphRecord(
        nodes={"A": LlmNode(prompt=["@a"], output="a")},
        edges=[StaticEdge(source="START", target="A")],
        state={},
    )
    assert first_token_sources(graph) == frozenset()


# ---------------------------------------------------------------------------
# Adapter — first_token_event stamping + per-dispatch cb routing
# ---------------------------------------------------------------------------


class FakeIssuer:
    def __init__(self) -> None:
        self.sent: list[object] = []

    async def issue_graph_credit(self, turn: object) -> bool:
        self.sent.append(turn)
        return True


@dataclass
class FakeCredit:
    x_correlation_id: str
    turn_index: int
    trace_id: str
    node_ordinal: int
    phase_variant: str = "profiling"


@dataclass
class FakeLlmNode:
    output: str = "out"


def _ctx(parent_trace_id: str, node_id: str) -> PlacementContext:
    return PlacementContext(parent_trace_id=parent_trace_id, parent_node_id=node_id)


def _request(node_id: str) -> DispatchRequest:
    return DispatchRequest(node_id=node_id)


def _catalog(trace_id: str, node_key_to_ordinal: dict[str, int]) -> CatalogContext:
    return CatalogContext(
        catalog={trace_id: dict(node_key_to_ordinal)},
    )


def _make_adapter(issuer: FakeIssuer, trace_id: str, ordinals: dict[str, int], **kw):
    return CreditDispatchAdapter(
        credit_issuer=issuer,
        catalog_context=_catalog(trace_id, ordinals),
        trace_id=trace_id,
        **kw,
    )


def _credit_for(turn: object) -> FakeCredit:
    return FakeCredit(
        x_correlation_id=turn.x_correlation_id,
        turn_index=turn.turn_index,
        trace_id=turn.trace_id,
        node_ordinal=turn.node_ordinal,
        phase_variant=turn.phase_variant,
    )


@pytest.mark.asyncio
async def test_adapter_registers_and_routes_first_token_cb():
    """A dispatch on a first-token source stamps ``first_token_event`` and parks
    the ``first_token_cb``; ``on_first_token`` fires it exactly once, and a repeat
    call for the same key after resolve is a graceful no-op."""
    issuer = FakeIssuer()
    adapter = _make_adapter(
        issuer, "t0", {"t0:0": 0}, first_token_sources=frozenset({"t0:0"})
    )
    fired: list[int] = []
    task = asyncio.create_task(
        adapter.dispatch(
            FakeLlmNode(),
            _request("t0:0"),
            _ctx("t0", "t0:0"),
            first_token_cb=lambda: fired.append(1),
        )
    )
    await asyncio.sleep(0)

    turn = issuer.sent[0]
    assert turn.first_token_event is True

    adapter.on_first_token(turn.x_correlation_id, turn.turn_index)
    assert fired == [1]

    adapter.resolve(_credit_for(turn), error=None, cancelled=False)
    assert isinstance(await task, str)

    # A second first-token for the same key (or after resolve) must not re-fire.
    adapter.on_first_token(turn.x_correlation_id, turn.turn_index)
    assert fired == [1]


@pytest.mark.asyncio
async def test_first_token_event_false_for_non_source_node():
    """A node that sources no first-token-anchored edge issues its turn with
    ``first_token_event=False`` (no wasted TTFT emission)."""
    issuer = FakeIssuer()
    adapter = _make_adapter(
        issuer, "t0", {"t0:0": 0}, first_token_sources=frozenset({"other:0"})
    )
    task = asyncio.create_task(
        adapter.dispatch(FakeLlmNode(), _request("t0:0"), _ctx("t0", "t0:0"))
    )
    await asyncio.sleep(0)
    assert issuer.sent[0].first_token_event is False
    adapter.resolve(_credit_for(issuer.sent[0]), error=None, cancelled=False)
    assert isinstance(await task, str)


@pytest.mark.asyncio
async def test_on_first_token_unknown_key_is_noop():
    """An ``on_first_token`` for a key that parked no callback (or None fields)
    is a graceful no-op."""
    issuer = FakeIssuer()
    adapter = _make_adapter(issuer, "t0", {"t0:0": 0})
    adapter.on_first_token("nope", 99)  # never registered
    adapter.on_first_token(None, None)  # None fast-path fields
    # Nothing raised, no cb to fire.


@pytest.mark.asyncio
async def test_first_token_cb_dropped_when_dispatch_resolves_without_ttft():
    """A resolve that arrives with no preceding TTFT drops the parked cb; a late
    ``on_first_token`` afterwards must not fire it."""
    issuer = FakeIssuer()
    adapter = _make_adapter(
        issuer, "t0", {"t0:0": 0}, first_token_sources=frozenset({"t0:0"})
    )
    fired: list[int] = []
    task = asyncio.create_task(
        adapter.dispatch(
            FakeLlmNode(),
            _request("t0:0"),
            _ctx("t0", "t0:0"),
            first_token_cb=lambda: fired.append(1),
        )
    )
    await asyncio.sleep(0)
    turn = issuer.sent[0]
    adapter.resolve(_credit_for(turn), error=None, cancelled=False)
    assert isinstance(await task, str)
    adapter.on_first_token(turn.x_correlation_id, turn.turn_index)
    assert fired == []


# ---------------------------------------------------------------------------
# Strategy — _on_graph_first_token de-mux by trace_id
# ---------------------------------------------------------------------------


class _FakeAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[str | None, int | None]] = []

    def on_first_token(
        self, x_correlation_id: str | None, turn_index: int | None
    ) -> None:
        self.calls.append((x_correlation_id, turn_index))


def _min_strategy() -> GraphIRReplayStrategy:
    from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace

    parsed = from_weka_trace(str(_WEKA_MIN))
    return GraphIRReplayStrategy(
        credit_issuer=object(),
        parsed_graph=parsed,
        register_observer=lambda obs: None,
    )


def _ft(trace_id, x_correlation_id="x", turn_index=0) -> FirstToken:
    from aiperf.common.enums import CreditPhase

    return FirstToken(
        credit_id=1,
        phase=CreditPhase.PROFILING,
        ttft_ns=5,
        trace_id=trace_id,
        x_correlation_id=x_correlation_id,
        turn_index=turn_index,
    )


def test_strategy_demux_routes_by_trace_id():
    """A FirstToken reaches ONLY the adapter registered under its trace_id."""
    strategy = _min_strategy()
    a, b = _FakeAdapter(), _FakeAdapter()
    strategy._adapters = {"A": a, "B": b}

    strategy._on_graph_first_token(_ft("A", x_correlation_id="xa", turn_index=3))
    assert a.calls == [("xa", 3)]
    assert b.calls == []


def test_strategy_demux_unknown_trace_id_is_noop():
    strategy = _min_strategy()
    a = _FakeAdapter()
    strategy._adapters = {"A": a}
    strategy._on_graph_first_token(_ft("ZZZ", x_correlation_id="xz", turn_index=1))
    assert a.calls == []


def test_strategy_demux_none_trace_id_is_noop():
    strategy = _min_strategy()
    a = _FakeAdapter()
    strategy._adapters = {"A": a}
    strategy._on_graph_first_token(_ft(None))
    assert a.calls == []


@pytest.mark.asyncio
async def test_setup_phase_installs_first_token_observer_when_wired():
    """When a first-token registrar is supplied, setup_phase installs the de-mux
    observer; teardown detaches it with None."""
    from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace

    parsed = from_weka_trace(str(_WEKA_MIN))
    installed_return: list[object] = []
    installed_ft: list[object] = []
    strategy = GraphIRReplayStrategy(
        credit_issuer=object(),
        parsed_graph=parsed,
        register_observer=installed_return.append,
        register_first_token_observer=installed_ft.append,
    )
    await strategy.setup_phase()
    assert len(installed_ft) == 1
    assert callable(installed_ft[0])

    await strategy.teardown_phase()
    assert installed_ft[-1] is None
