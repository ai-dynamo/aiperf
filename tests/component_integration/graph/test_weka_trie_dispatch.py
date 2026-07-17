# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""T6.6 e2e: the schedule plane DISPATCHES every weka segment-trie ``LlmNode``.

Drives a real ``GraphIRReplayStrategy.execute_phase`` over a trie ``ParsedGraph``
built from ``weka_subagent.json`` with
a fake/echo credit issuer (no worker, no ZMQ). Proves the catalog/ordinal
resolution the build plane wrote its unified segment store at is the SAME ordinal
the dispatch adapter resolves a fired node to -- so credits actually flow and the
worker would read the right envelope:

* (a) EVERY flat trie ``LlmNode`` is dispatched exactly once;
* (b) each dispatched credit's ``node_ordinal`` equals the ordinal
  ``trie_node_ordinals`` assigns to that node id (the build<->schedule contract);
* (c) the phase completes with no error and no ``GraphEnvelopeMissing`` / unresolved
  (``node_ordinal is None``) dispatch.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.component_integration, pytest.mark.asyncio]

_FIX_DIR = Path(__file__).parents[2] / "unit" / "graph" / "fixtures"
_SUBAGENT = _FIX_DIR / "weka_subagent.json"


@pytest.fixture(autouse=True)
def _offline_hf(monkeypatch):
    """Pin the tokenizer load to the local HuggingFace cache (offline HF)."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


@dataclass
class _EchoIssuer:
    """Fake issuer recording each issued turn; echoes its return next tick.

    Mirrors ``test_graph_ir_overlap_barrier._OrderedEchoIssuer`` (the established
    no-worker credit-loop pattern): every ``issue_graph_credit`` records the turn
    and schedules the strategy's return observer to fire on the next loop tick,
    so the parked dispatch Future resolves exactly as a real worker round-trip
    would. The echoed object IS the ``TurnToSend`` -- it carries
    ``x_correlation_id`` / ``turn_index`` / ``trace_id`` / ``node_ordinal``, the
    only fields the observer + adapter read.
    """

    observer: Any = None
    issued_turns: list[Any] = field(default_factory=list)
    issued: int = 0
    returned: int = 0

    async def issue_graph_credit(self, turn: Any) -> bool:
        self.issued += 1
        self.issued_turns.append(turn)
        asyncio.get_running_loop().call_soon(self._echo, turn)
        return True

    def _echo(self, turn: Any) -> None:
        self.returned += 1
        if self.observer is not None:
            self.observer(turn, None, False)

    def mark_graph_sending_complete(self) -> None:
        pass

    def graph_all_returned(self) -> bool:
        return self.returned >= self.issued

    def set_graph_all_returned_event(self) -> None:
        pass


def _parsed_trie():
    from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace

    parsed = from_weka_trace(str(_SUBAGENT))
    assert parsed.segment_pool is not None, "fixture did not build a trie graph"
    return parsed


def _make_strategy(parsed, issuer):
    from aiperf.timing.strategies.graph_ir_replay import GraphIRReplayStrategy

    # Full-replay window (t*=0): assert the WHOLE trie graph dispatches; a
    # positive t* would chop pre-t* nodes and the every-node count would not hold.
    return GraphIRReplayStrategy(
        credit_issuer=issuer,
        parsed_graph=parsed,
        register_observer=lambda obs: setattr(issuer, "observer", obs),
        max_concurrent_traces=8,
        start_min_ratio=0.0,
        start_max_ratio=0.0,
    )


async def test_trie_dispatch_every_node_resolves_build_plane_ordinal():
    """Every trie node dispatches once, each at its ``trie_node_ordinals`` ordinal."""
    from aiperf.dataset.graph.models import LlmNode
    from aiperf.dataset.graph.segment_ir.store_builder import trie_node_ordinals

    parsed = _parsed_trie()
    llm_nodes = {
        nid: n for nid, n in parsed.graph.nodes.items() if isinstance(n, LlmNode)
    }
    expected_ordinals = trie_node_ordinals(llm_nodes)
    assert expected_ordinals, "trie fixture produced no LlmNodes"

    issuer = _EchoIssuer()
    strategy = _make_strategy(parsed, issuer)
    await strategy.setup_phase()
    await asyncio.wait_for(strategy.execute_phase(), timeout=10.0)

    # (c) phase completed cleanly, no errored trace.
    assert strategy.completed_traces == 1
    assert strategy.errored_traces == 0

    # (a) every trie LlmNode dispatched exactly once.
    assert issuer.issued == len(llm_nodes)
    ordinals_issued = sorted(t.node_ordinal for t in issuer.issued_turns)
    assert ordinals_issued == sorted(expected_ordinals.values())

    # (b) each dispatched credit's ordinal is a real build-plane ordinal (never
    # the unresolved None that produces GraphEnvelopeMissing at the worker).
    assert all(t.node_ordinal is not None for t in issuer.issued_turns)
    assert set(ordinals_issued) == set(expected_ordinals.values())


async def test_trie_dispatch_catalog_matches_build_plane_store():
    """The strategy's catalog is byte-identical to the build plane's ordinal map.

    The build plane's unified trie builders key each node's manifest by
    ``trie_node_ordinals``; the schedule plane's ``build_catalog_context`` must
    produce the SAME ``{node_id: ordinal}`` so the dispatched credit reads the
    right manifest. A drift here is exactly the GraphEnvelopeMissing failure T6.6
    closes.
    """
    from aiperf.dataset.graph.graph_path_catalog import build_catalog_context
    from aiperf.dataset.graph.models import LlmNode
    from aiperf.dataset.graph.segment_ir.store_builder import trie_node_ordinals

    parsed = _parsed_trie()
    trace = parsed.traces[0]
    llm_nodes = {
        nid: n for nid, n in parsed.graph.nodes.items() if isinstance(n, LlmNode)
    }

    catalog = build_catalog_context(parsed).catalog[trace.id]
    assert catalog == trie_node_ordinals(llm_nodes)
