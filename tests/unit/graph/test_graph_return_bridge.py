# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R2 — unconditional graph-return hook on CreditCallbackHandler.

Graph credit returns must reach the adapter via a DEDICATED, UNCONDITIONAL
observer (not the gated ``strategy.handle_credit_return``, which is skipped
when ``can_send_any_turn()`` is False). On a graph credit the observer fires
regardless of phase-send gating, carrying ``(credit, error, cancelled)`` so the
adapter can resolve OR reject the parked Future.
"""

from __future__ import annotations

import pytest

from aiperf.credit.messages import CreditReturn
from aiperf.credit.structs import Credit

pytestmark = pytest.mark.asyncio


def _graph_credit(
    *, x_corr: str = "x0", trace_id: str = "t0", final: bool = True
) -> Credit:
    return Credit(
        id=1,
        phase="profiling",
        conversation_id="c",
        x_correlation_id=x_corr,
        turn_index=0,
        num_turns=1,
        issued_at_ns=0,
        trace_id=trace_id,
        node_ordinal=0,
        phase_variant="profiling",
    )


def _non_graph_credit() -> Credit:
    return Credit(
        id=2,
        phase="profiling",
        conversation_id="c",
        x_correlation_id="x1",
        turn_index=0,
        num_turns=1,
        issued_at_ns=0,
    )


def _make_handler():
    from aiperf.credit.callback_handler import CreditCallbackHandler

    class _FakeConcurrency:
        def release_session_slot(self, phase) -> None: ...
        def release_prefill_slot(self, phase) -> None: ...

    return CreditCallbackHandler(_FakeConcurrency())


async def test_graph_return_observer_fires_unconditionally(monkeypatch):
    """Even when no phase handler is registered (can_send_any_turn would be
    False / the gated path is unreachable), the graph observer still fires."""
    handler = _make_handler()
    seen: list[tuple[Credit, str | None, bool]] = []

    def _observer(credit: Credit, error: str | None, cancelled: bool) -> None:
        seen.append((credit, error, cancelled))

    handler.set_graph_return_observer(_observer)

    credit = _graph_credit()
    ret = CreditReturn(credit=credit, cancelled=False, error=None)
    # No phase registered -> the gated strategy path no-ops, but the graph
    # observer must still receive the return.
    await handler.on_credit_return("w0", ret)

    assert len(seen) == 1
    got_credit, got_error, got_cancelled = seen[0]
    assert got_credit.trace_id == "t0"
    assert got_error is None
    assert got_cancelled is False


async def test_graph_return_observer_forwards_error_and_cancelled():
    handler = _make_handler()
    seen: list[tuple[str | None, bool]] = []
    handler.set_graph_return_observer(lambda c, e, x: seen.append((e, x)))

    await handler.on_credit_return(
        "w0",
        CreditReturn(credit=_graph_credit(x_corr="xe"), cancelled=False, error="boom"),
    )
    await handler.on_credit_return(
        "w0",
        CreditReturn(credit=_graph_credit(x_corr="xc"), cancelled=True, error=None),
    )

    assert ("boom", False) in seen
    assert (None, True) in seen


async def test_non_graph_credit_does_not_invoke_graph_observer():
    handler = _make_handler()
    seen: list[Credit] = []
    handler.set_graph_return_observer(lambda c, e, x: seen.append(c))

    await handler.on_credit_return(
        "w0", CreditReturn(credit=_non_graph_credit(), cancelled=False, error=None)
    )
    assert seen == []
