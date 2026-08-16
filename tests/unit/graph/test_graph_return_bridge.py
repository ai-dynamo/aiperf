# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R2 -- the graph-return hook on CreditCallbackHandler fires outside the gated strategy path."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.credit.callback_handler import CreditCallbackHandler
from aiperf.credit.messages import CreditReturn
from aiperf.credit.structs import Credit

pytestmark = pytest.mark.asyncio


def _graph_credit(*, x_corr: str = "x0", trace_id: str = "t0") -> Credit:
    """A graph-addressed credit (trace_id set) as the worker returns it."""
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
    )


def _non_graph_credit() -> Credit:
    """A classic credit with no graph addressing."""
    return Credit(
        id=2,
        phase="profiling",
        conversation_id="c",
        x_correlation_id="x1",
        turn_index=0,
        num_turns=1,
        issued_at_ns=0,
    )


def _make_handler() -> CreditCallbackHandler:
    """A handler with no phase registered, so only the ungated paths can run."""

    class _FakeConcurrency:
        def release_session_slot(self, phase: object) -> None: ...
        def release_prefill_slot(self, phase: object) -> None: ...

    return CreditCallbackHandler(_FakeConcurrency())


@pytest.mark.parametrize(
    ("cancelled", "error"),
    [
        param(False, None, id="clean-return"),
        param(False, "boom", id="errored-return"),
        param(True, None, id="cancelled-return"),
    ],
)  # fmt: skip
async def test_graph_return_observer_fires_unconditionally(
    cancelled: bool, error: str | None
) -> None:
    """The graph observer receives every graph return verbatim, error/cancel flags included."""
    # No phase is registered, so can_send_any_turn is False and the gated strategy
    # path no-ops -- the observer must fire anyway.
    handler = _make_handler()
    seen: list[tuple[Credit, str | None, bool, int | None, int | None, int | None]] = []

    def _observer(
        credit: Credit,
        err: str | None,
        was_cancelled: bool,
        *,
        osl: int | None,
        request_latency_ns: int | None,
        ttft_ns: int | None,
    ) -> None:
        seen.append(
            (
                credit,
                err,
                was_cancelled,
                osl,
                request_latency_ns,
                ttft_ns,
            )
        )

    handler.set_graph_return_observer(_observer)

    ret = CreditReturn(
        credit=_graph_credit(),
        cancelled=cancelled,
        error=error,
        request_latency_ns=900_000_000,
        ttft_ns=300_000_000,
    )
    await handler.on_credit_return("w0", ret)

    assert len(seen) == 1
    got_credit, got_error, got_cancelled, got_osl, got_latency, got_ttft = seen[0]
    assert got_credit.trace_id == "t0"
    assert got_error == error
    assert got_cancelled is cancelled
    assert got_osl is None
    assert got_latency == 900_000_000
    assert got_ttft == 300_000_000


async def test_non_graph_credit_does_not_invoke_graph_observer() -> None:
    """A credit without a trace_id is not routed to the graph observer."""
    handler = _make_handler()
    seen: list[Credit] = []
    handler.set_graph_return_observer(
        lambda c, e, x, osl, latency, ttft: seen.append(c)
    )

    await handler.on_credit_return(
        "w0", CreditReturn(credit=_non_graph_credit(), cancelled=False, error=None)
    )
    assert seen == []


async def test_raising_observer_is_contained() -> None:
    """A throwing observer must not abort the rest of ``on_credit_return``.

    The observer fires ahead of counting, slot release, and the drain event; an
    escaping exception would skip all three and hang the phase on its drain
    wait. Both observers in the tests above are non-raising, so nothing else
    exercises the containment.
    """
    handler = _make_handler()

    def _boom(
        credit: Credit,
        err: str | None,
        was_cancelled: bool,
        output_sequence_length: int | None,
        request_latency_ns: int | None,
        ttft_ns: int | None,
    ) -> None:
        raise RuntimeError("observer exploded")

    handler.set_graph_return_observer(_boom)

    await handler.on_credit_return(
        "w0", CreditReturn(credit=_graph_credit(), cancelled=False, error=None)
    )
