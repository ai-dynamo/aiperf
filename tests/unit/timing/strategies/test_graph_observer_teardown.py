# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deferred graph-phase teardown must not clear the NEXT phase's observers.

One ``CreditCallbackHandler`` is shared by every ``PhaseRunner``. A seamless
non-final graph phase defers ``teardown_phase`` to its background return-wait
completion, which can fire AFTER the next phase's ``setup_phase`` installed ITS
observers on the same shared slots. Teardown must therefore compare-and-clear:
null a slot only when the observer installed there is the tearing-down
strategy's own. These tests drive the REAL ``CreditCallbackHandler`` (a
MagicMock handler hides exactly this bug) through the production
``PhaseRunner._build_graph_ir_strategy`` wiring.
"""

from __future__ import annotations

from pathlib import Path

from aiperf.common.enums import CreditPhase
from aiperf.credit.callback_handler import CreditCallbackHandler
from aiperf.credit.messages import CreditReturn, FirstToken
from aiperf.credit.structs import Credit
from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
from aiperf.plugin.enums import TimingMode
from aiperf.timing.graph_channel import GraphPhaseChannel
from aiperf.timing.phase.runner import PhaseRunner
from aiperf.timing.strategies.graph_ir_replay import GraphIRReplayStrategy

_WEKA_MIN = Path(__file__).parents[2] / "graph" / "fixtures" / "weka_min.json"


class _FakeConcurrency:
    def release_session_slot(self, phase: CreditPhase) -> None: ...

    def release_prefill_slot(self, phase: CreditPhase) -> None: ...


class _FakeAdapter:
    """Records the de-mux calls the strategy routes to the owning adapter."""

    def __init__(self) -> None:
        self.resolved: list[tuple[str | None, bool]] = []
        self.first_tokens: list[tuple[str | None, int | None]] = []

    def resolve(self, credit: Credit, error: str | None, cancelled: bool) -> None:
        self.resolved.append((error, cancelled))

    def on_first_token(
        self, x_correlation_id: str | None, turn_index: int | None
    ) -> None:
        self.first_tokens.append((x_correlation_id, turn_index))


def _build_strategy(handler: CreditCallbackHandler) -> GraphIRReplayStrategy:
    """Build through the production runner seam so the register/unregister
    wiring under test is the real one (not a hand-rolled approximation)."""

    class _Config:
        timing_mode = TimingMode.GRAPH_IR
        phase = None
        concurrency = None
        expected_num_sessions = None

    channel = GraphPhaseChannel(parsed_graph=from_weka_trace(str(_WEKA_MIN)))
    runner = PhaseRunner.__new__(PhaseRunner)
    runner._config = _Config()
    runner._conversation_source = None
    runner._graph_channel = channel
    runner._scheduler = None
    runner._stop_checker = None
    runner._credit_issuer = object()
    runner._lifecycle = None
    runner._callback_handler = handler
    return runner._build_graph_ir_strategy(GraphIRReplayStrategy)


def _graph_credit(trace_id: str) -> Credit:
    return Credit(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id="c",
        x_correlation_id="x0",
        turn_index=0,
        num_turns=1,
        issued_at_ns=0,
        trace_id=trace_id,
        node_ordinal=0,
        phase_variant="profiling",
    )


def _first_token(trace_id: str) -> FirstToken:
    return FirstToken(
        credit_id=1,
        phase=CreditPhase.PROFILING,
        ttft_ns=5,
        trace_id=trace_id,
        x_correlation_id="x0",
        turn_index=0,
    )


async def test_stale_deferred_teardown_preserves_next_phase_observers() -> None:
    """Phase A's late teardown must not null phase B's live observers."""
    handler = CreditCallbackHandler(_FakeConcurrency())
    strategy_a = _build_strategy(handler)
    strategy_b = _build_strategy(handler)

    await strategy_a.setup_phase()
    await strategy_b.setup_phase()  # the next phase takes over the shared slots

    # Phase A's deferred (return-wait done-callback) teardown fires late.
    await strategy_a.teardown_phase()

    # B's return observer must still dispatch into B's adapter registry.
    adapter = _FakeAdapter()
    strategy_b._adapters["t-live"] = adapter
    await handler.on_credit_return(
        "w0", CreditReturn(credit=_graph_credit("t-live"), cancelled=False, error=None)
    )
    assert adapter.resolved == [(None, False)], (
        "phase B's graph return was dropped: the stale phase A teardown "
        "cleared the shared graph-return observer slot"
    )

    # B's first-token observer must survive too (post-TTFT anchoring).
    await handler.on_first_token(_first_token("t-live"))
    assert adapter.first_tokens == [("x0", 0)], (
        "phase B's first-token event was dropped: the stale phase A teardown "
        "cleared the shared first-token observer slot"
    )


async def test_teardown_clears_own_still_installed_observers() -> None:
    """A strategy whose observers ARE still installed must clear both slots."""
    handler = CreditCallbackHandler(_FakeConcurrency())
    strategy = _build_strategy(handler)

    await strategy.setup_phase()
    assert handler._graph_return_observer is not None
    assert handler._graph_first_token_observer is not None

    await strategy.teardown_phase()
    assert handler._graph_return_observer is None
    assert handler._graph_first_token_observer is None
