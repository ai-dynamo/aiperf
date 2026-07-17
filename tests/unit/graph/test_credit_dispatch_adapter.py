# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R4/R2 — CreditDispatchAdapter (Future bridge) + unconditional return hook.

Component-level tests with a FAKE issuer + return loop. No real worker, no ZMQ.

The adapter bridges the dataflow ``TraceExecutor.dispatch(node, request, ctx)``
contract onto the v1 credit pipeline: it issues a real ``TurnToSend`` carrying
``trace_id``/``node_ordinal``/``phase_variant``, parks an ``asyncio.Future``
keyed by a per-trajectory ``(x_correlation_id, turn_index)`` coordinate, and
awaits it. The strategy resolves that Future on the correlated ``CreditReturn``
(success/cancel/error). A timeout guard rejects rather than hangs when a return
is dropped.

Identity contract under test:
* node id is ``{scope}:{turn}`` (0-based turn within the trajectory scope);
* ``x_correlation_id`` is ``{conversation_id}::{uuid4().hex}`` -- ONE per scope
  per adapter instance (all turns of a trajectory share it);
* ``turn_index`` is the node's own 0-based turn parsed from the node id.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from aiperf.dataset.graph.graph_path_catalog import CatalogContext
from aiperf.graph.placement import DispatchRequest, PlacementContext

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeIssuer:
    """Records issued graph TurnToSends; lets the test echo a CreditReturn.

    Mirrors the surface the adapter calls: a single ``issue_graph_credit``
    entrypoint that records the turn and reports "can send more" (True).
    """

    def __init__(self) -> None:
        self.sent: list[object] = []

    async def issue_graph_credit(self, turn: object) -> bool:
        self.sent.append(turn)
        return True


@dataclass
class FakeCredit:
    """Minimal Credit-like carrying the correlation identity the bridge keys on."""

    x_correlation_id: str
    turn_index: int
    trace_id: str
    node_ordinal: int
    phase_variant: str = "profiling"


class FalseIssuer:
    """Issuer whose ``issue_graph_credit`` always refuses (returns False).

    Mirrors the real ``CreditIssuer.issue_graph_credit`` stop-gate / prefill-gate
    refusal: no credit reaches the wire, so no ``CreditReturn`` can ever arrive to
    resolve the parked Future. Records the turn it was asked to issue so the test
    can confirm the adapter still minted addressing before being refused.
    """

    def __init__(self) -> None:
        self.sent: list[object] = []

    async def issue_graph_credit(self, turn: object) -> bool:
        self.sent.append(turn)
        return False


@dataclass
class FakeLlmNode:
    """Stand-in IR node; the adapter does not read node fields for weka."""

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
    from aiperf.graph.credit_dispatch_adapter import CreditDispatchAdapter

    return CreditDispatchAdapter(
        credit_issuer=issuer,
        catalog_context=_catalog(trace_id, ordinals),
        trace_id=trace_id,
        phase_variant=kw.pop("phase_variant", "profiling"),
        parent_correlation_id=kw.pop("parent_correlation_id", None),
        **kw,
    )


# ---------------------------------------------------------------------------
# Happy path: issue carries addressing; echoing the return resolves the await
# ---------------------------------------------------------------------------


async def test_dispatch_issues_credit_with_graph_addressing():
    issuer = FakeIssuer()
    adapter = _make_adapter(issuer, "t0", {"t0:0": 7}, phase_variant="profiling")

    task = asyncio.create_task(
        adapter.dispatch(FakeLlmNode(), _request("t0:0"), _ctx("t0", "t0:0"))
    )
    await asyncio.sleep(0)  # let it issue + park the future

    assert len(issuer.sent) == 1
    turn = issuer.sent[0]
    assert turn.trace_id == "t0"
    assert turn.node_ordinal == 7
    assert turn.phase_variant == "profiling"
    # conversation_id is the root-scope TEMPLATE id; x_correlation_id is
    # {conversation_id}::{nonce}; turn_index is the node's own 0-based turn.
    assert turn.conversation_id == "t0"
    assert turn.x_correlation_id.startswith("t0::")
    assert turn.turn_index == 0

    adapter.resolve(_credit_for(turn), error=None, cancelled=False)
    result = await task
    assert isinstance(result, str)


async def test_dispatch_returns_placeholder_str_on_success():
    issuer = FakeIssuer()
    adapter = _make_adapter(issuer, "t0", {"t0:0": 0})
    task = asyncio.create_task(
        adapter.dispatch(FakeLlmNode(), _request("t0:0"), _ctx("t0", "t0:0"))
    )
    await asyncio.sleep(0)
    adapter.resolve(_credit_for(issuer.sent[0]), error=None, cancelled=False)
    assert isinstance(await task, str)


def _credit_for(turn: object) -> FakeCredit:
    return FakeCredit(
        x_correlation_id=turn.x_correlation_id,
        turn_index=turn.turn_index,
        trace_id=turn.trace_id,
        node_ordinal=turn.node_ordinal,
        phase_variant=turn.phase_variant,
    )


# ---------------------------------------------------------------------------
# Session-routing identity: real per-trajectory num_turns + root corr
# ---------------------------------------------------------------------------


async def test_turns_carry_recorded_num_turns_and_root_corr():
    """num_turns is the trajectory's RECORDED turn count from the catalog
    (is_final_turn = the recorded session-final fact for session-routing
    bind/close semantics), and every turn carries the instance's ROOT
    trajectory corr as root_correlation_id."""
    issuer = FakeIssuer()
    adapter = _make_adapter(issuer, "t0", {"t0:0": 0, "t0:1": 1, "child:0": 2})

    tasks = []
    for nid in ("t0:0", "t0:1", "child:0"):
        tasks.append(
            asyncio.create_task(
                adapter.dispatch(FakeLlmNode(), _request(nid), _ctx("t0", nid))
            )
        )
    await asyncio.sleep(0)

    root0, root1, child0 = issuer.sent
    assert (root0.num_turns, root1.num_turns) == (2, 2)
    assert root0.is_final_turn is False
    assert root1.is_final_turn is True  # recorded session-final turn
    assert child0.num_turns == 1
    assert child0.is_final_turn is True
    # Root corr: one stable id for the whole instance, equal to the root
    # trajectory's own corr.
    assert root0.root_correlation_id == root0.x_correlation_id
    assert child0.root_correlation_id == root0.x_correlation_id
    assert child0.x_correlation_id != child0.root_correlation_id

    for turn, task in zip(issuer.sent, tasks, strict=True):
        adapter.resolve(_credit_for(turn), error=None, cancelled=False)
        assert isinstance(await task, str)


async def test_unknown_runtime_scope_reads_non_final():
    """A runtime scope the catalog does not know must never read as final:
    a wrong session-close on the wire is worse than a missing one."""
    issuer = FakeIssuer()
    adapter = _make_adapter(issuer, "t0", {"t0:0": 0})

    task = asyncio.create_task(
        adapter.dispatch(FakeLlmNode(), _request("mystery:0"), _ctx("t0", "mystery:0"))
    )
    await asyncio.sleep(0)

    (turn,) = issuer.sent
    assert turn.is_final_turn is False

    adapter.resolve(_credit_for(turn), error=None, cancelled=False)
    assert isinstance(await task, str)


# ---------------------------------------------------------------------------
# Native-authored node ids (no {scope}:{turn} shape)
# ---------------------------------------------------------------------------


async def test_native_bare_node_ids_share_one_root_trajectory():
    """Author-chosen ids like ``"plan"`` map to the ROOT trajectory with the
    catalog ordinal as the turn: one corr for the whole graph (dynamic-slot
    content stays on one sticky session), unique waiter keys per node.
    Regression: this used to crash with ``ValueError: invalid literal for
    int()`` at the first dispatch of any native graph."""
    issuer = FakeIssuer()
    adapter = _make_adapter(issuer, "t0", {"plan": 0, "review": 1})

    tasks = [
        asyncio.create_task(
            adapter.dispatch(FakeLlmNode(), _request(nid), _ctx("t0", nid))
        )
        for nid in ("plan", "review")
    ]
    await asyncio.sleep(0)

    assert len(issuer.sent) == 2
    plan, review = issuer.sent
    assert plan.x_correlation_id == review.x_correlation_id
    assert plan.x_correlation_id.startswith("t0::")
    assert plan.conversation_id == review.conversation_id == "t0"
    assert (plan.turn_index, review.turn_index) == (0, 1)
    assert (plan.node_ordinal, review.node_ordinal) == (0, 1)

    for turn, task in zip(issuer.sent, tasks, strict=True):
        adapter.resolve(_credit_for(turn), error=None, cancelled=False)
        assert isinstance(await task, str)


async def test_native_non_int_tail_id_uses_root_fallback():
    """A colon-bearing id whose tail is not an int (``phase:review``) is a
    native-authored id, not a {scope}:{turn} coordinate."""
    issuer = FakeIssuer()
    adapter = _make_adapter(issuer, "t0", {"phase:review": 3})

    task = asyncio.create_task(
        adapter.dispatch(
            FakeLlmNode(), _request("phase:review"), _ctx("t0", "phase:review")
        )
    )
    await asyncio.sleep(0)

    (turn,) = issuer.sent
    assert turn.conversation_id == "t0"
    assert turn.x_correlation_id.startswith("t0::")
    assert turn.turn_index == 3  # catalog ordinal, NOT parsed from the id

    adapter.resolve(_credit_for(turn), error=None, cancelled=False)
    assert isinstance(await task, str)


# ---------------------------------------------------------------------------
# Cancel / error reject the Future (no hang)
# ---------------------------------------------------------------------------


async def test_error_return_rejects_future_no_hang():
    issuer = FakeIssuer()
    adapter = _make_adapter(issuer, "t0", {"t0:0": 0})
    task = asyncio.create_task(
        adapter.dispatch(FakeLlmNode(), _request("t0:0"), _ctx("t0", "t0:0"))
    )
    await asyncio.sleep(0)
    adapter.resolve(_credit_for(issuer.sent[0]), error="boom", cancelled=False)
    from aiperf.graph.credit_dispatch_adapter import GraphDispatchError

    with pytest.raises(GraphDispatchError):
        await asyncio.wait_for(task, timeout=1.0)


async def test_cancelled_return_rejects_future_no_hang():
    issuer = FakeIssuer()
    adapter = _make_adapter(issuer, "t0", {"t0:0": 0})
    task = asyncio.create_task(
        adapter.dispatch(FakeLlmNode(), _request("t0:0"), _ctx("t0", "t0:0"))
    )
    await asyncio.sleep(0)
    adapter.resolve(_credit_for(issuer.sent[0]), error=None, cancelled=True)
    from aiperf.graph.credit_dispatch_adapter import GraphDispatchError

    with pytest.raises(GraphDispatchError):
        await asyncio.wait_for(task, timeout=1.0)


# ---------------------------------------------------------------------------
# Timeout guard rejects rather than hangs
# ---------------------------------------------------------------------------


async def test_timeout_rejects_future_when_no_return_arrives():
    issuer = FakeIssuer()
    adapter = _make_adapter(issuer, "t0", {"t0:0": 0}, dispatch_timeout_s=0.01)
    with pytest.raises(asyncio.TimeoutError):
        await adapter.dispatch(FakeLlmNode(), _request("t0:0"), _ctx("t0", "t0:0"))


# ---------------------------------------------------------------------------
# Correlation-key uniqueness: distinct scopes (loop iterations, fork branches)
# park distinct Futures. The differentiator now rides the NODE ID's scope
# (``{scope}:{turn}``), not ``ctx.parent_trace_id`` (which ``_mint`` ignores).
# ---------------------------------------------------------------------------


async def test_loop_refire_distinct_scopes_do_not_collide():
    """Two in-flight dispatches in distinct loop-iteration scopes must park
    distinct Futures; resolving the first must not orphan the second.

    Loop iterations scope their nodes as ``{trace}::loop#N:{turn}`` (the
    executor's loop scoping now lives in the node id, so distinct scopes mint
    distinct trajectory correlation ids)."""
    issuer = FakeIssuer()
    adapter = _make_adapter(issuer, "t0", {"t0::loop#1:0": 0, "t0::loop#2:0": 0})

    t1 = asyncio.create_task(
        adapter.dispatch(
            FakeLlmNode(), _request("t0::loop#1:0"), _ctx("t0", "t0::loop#1:0")
        )
    )
    t2 = asyncio.create_task(
        adapter.dispatch(
            FakeLlmNode(), _request("t0::loop#2:0"), _ctx("t0", "t0::loop#2:0")
        )
    )
    await asyncio.sleep(0)
    assert len(issuer.sent) == 2
    # Both in-flight at once -> two distinct waiters.
    assert adapter.inflight_count == 2
    # Distinct scopes -> distinct trajectory correlation ids.
    assert issuer.sent[0].x_correlation_id != issuer.sent[1].x_correlation_id

    adapter.resolve(_credit_for(issuer.sent[0]), error=None, cancelled=False)
    # First-resolved ordering: iteration 1 completes; iteration 2 stays parked
    # (its Future is unresolved, so it CANNOT have completed), NOT orphaned.
    assert isinstance(await t1, str)
    assert not t2.done()
    assert adapter.inflight_count == 1
    adapter.resolve(_credit_for(issuer.sent[1]), error=None, cancelled=False)
    assert isinstance(await t2, str)


async def test_fork_branch_same_ordinal_does_not_collide():
    """Two fork branches reaching nodes that share an ordinal across distinct
    branch scopes must park distinct Futures."""
    issuer = FakeIssuer()
    # Both branches share ordinal 0, but distinct branch scopes (``::brA`` /
    # ``::brB``) so they mint distinct trajectory correlation ids.
    adapter = _make_adapter(issuer, "t0", {"t0::brA:0": 0, "t0::brB:0": 0})
    a = asyncio.create_task(
        adapter.dispatch(FakeLlmNode(), _request("t0::brA:0"), _ctx("t0", "t0::brA:0"))
    )
    b = asyncio.create_task(
        adapter.dispatch(FakeLlmNode(), _request("t0::brB:0"), _ctx("t0", "t0::brB:0"))
    )
    await asyncio.sleep(0)
    assert adapter.inflight_count == 2
    adapter.resolve(_credit_for(issuer.sent[0]), error=None, cancelled=False)
    adapter.resolve(_credit_for(issuer.sent[1]), error=None, cancelled=False)
    assert isinstance(await a, str)
    assert isinstance(await b, str)


async def test_duplicate_inflight_same_coordinate_raises():
    """Re-firing the SAME ``(scope, turn)`` node while its first dispatch is still
    in flight raises ``RuntimeError`` -- the executor fires each node at most once
    per instance run, so a duplicate waiter key is a programming error, never a
    silently-shared Future."""
    issuer = FakeIssuer()
    adapter = _make_adapter(issuer, "t0", {"t0:0": 0})
    first = asyncio.create_task(
        adapter.dispatch(FakeLlmNode(), _request("t0:0"), _ctx("t0", "t0:0"))
    )
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError):
        await adapter.dispatch(FakeLlmNode(), _request("t0:0"), _ctx("t0", "t0:0"))
    adapter.resolve(_credit_for(issuer.sent[0]), error=None, cancelled=False)
    assert isinstance(await first, str)


# ---------------------------------------------------------------------------
# Stall guard (adv2 A1): a refused issue must reject PROMPTLY, not time out
# ---------------------------------------------------------------------------


async def test_refused_issue_rejects_promptly_not_after_timeout():
    """When ``issue_graph_credit`` returns False, no credit reaches the wire so
    no ``CreditReturn`` can ever resolve the parked Future. The adapter MUST pop
    the waiter and reject immediately rather than await out ``dispatch_timeout_s``.

    Guard: a generous timeout (the would-be hang) with a tight ``wait_for`` bound
    proves promptness -- the dispatch resolves in well under the timeout.
    """
    issuer = FalseIssuer()
    # dispatch_timeout_s large enough that timing-out would dwarf the wait_for
    # bound; if the fix is missing, dispatch hangs until this 30s elapses.
    adapter = _make_adapter(issuer, "t0", {"t0:0": 0}, dispatch_timeout_s=30.0)

    from aiperf.graph.credit_dispatch_adapter import GraphDispatchError

    with pytest.raises(GraphDispatchError):
        await asyncio.wait_for(
            adapter.dispatch(FakeLlmNode(), _request("t0:0"), _ctx("t0", "t0:0")),
            timeout=1.0,
        )
    # The adapter minted addressing before being refused, then unwound cleanly.
    assert len(issuer.sent) == 1
    assert adapter.inflight_count == 0


async def test_unknown_return_is_dropped_not_raised():
    """A return whose key matches no parked waiter is a graceful no-op."""
    issuer = FakeIssuer()
    adapter = _make_adapter(issuer, "t0", {"t0:0": 0})
    stray = FakeCredit(
        x_correlation_id="x-nope", turn_index=99, trace_id="t0", node_ordinal=0
    )
    adapter.resolve(stray, error=None, cancelled=False)  # must not raise
    assert adapter.inflight_count == 0


# ---------------------------------------------------------------------------
# ``on_drained`` fires when the in-flight set empties.
#
# The strategy registers an ``on_drained`` callback so it can defer popping a
# detached-spawn instance's adapter from its de-mux registry until the adapter is
# idle. The adapter must invoke that callback EXACTLY when a return drains the
# last in-flight waiter, and never while another dispatch is still parked.
# ---------------------------------------------------------------------------


async def test_on_drained_fires_when_last_waiter_resolves():
    """A single dispatch's return drains the waiter set -> ``on_drained`` fires."""
    issuer = FakeIssuer()
    drained: list[object] = []
    adapter = _make_adapter(
        issuer, "t0", {"t0:0": 0}, on_drained=lambda a: drained.append(a)
    )
    task = asyncio.create_task(
        adapter.dispatch(FakeLlmNode(), _request("t0:0"), _ctx("t0", "t0:0"))
    )
    await asyncio.sleep(0)
    assert adapter.inflight_count == 1
    assert drained == []  # not yet drained

    adapter.resolve(_credit_for(issuer.sent[0]), error=None, cancelled=False)
    assert isinstance(await task, str)
    assert drained == [adapter]  # fired exactly once, with self
    assert adapter.inflight_count == 0


async def test_on_drained_not_fired_while_another_dispatch_parked():
    """With two in-flight dispatches, resolving the first must NOT fire
    ``on_drained`` (the set is not empty); resolving the second then fires it."""
    issuer = FakeIssuer()
    drained: list[object] = []
    adapter = _make_adapter(
        issuer,
        "t0",
        {"t0::brA:0": 0, "t0::brB:0": 0},
        on_drained=lambda a: drained.append(a),
    )
    t1 = asyncio.create_task(
        adapter.dispatch(FakeLlmNode(), _request("t0::brA:0"), _ctx("t0", "t0::brA:0"))
    )
    t2 = asyncio.create_task(
        adapter.dispatch(FakeLlmNode(), _request("t0::brB:0"), _ctx("t0", "t0::brB:0"))
    )
    await asyncio.sleep(0)
    assert adapter.inflight_count == 2

    adapter.resolve(_credit_for(issuer.sent[0]), error=None, cancelled=False)
    await asyncio.sleep(0)
    assert drained == []  # second still parked -> NOT drained
    assert adapter.inflight_count == 1

    adapter.resolve(_credit_for(issuer.sent[1]), error=None, cancelled=False)
    assert isinstance(await t1, str)
    assert isinstance(await t2, str)
    assert drained == [adapter]  # drained exactly once, when the set emptied


async def test_on_drained_fires_on_rejected_return_too():
    """A cancel/error return is a terminal drain too -> ``on_drained`` fires."""
    issuer = FakeIssuer()
    drained: list[object] = []
    adapter = _make_adapter(
        issuer, "t0", {"t0:0": 0}, on_drained=lambda a: drained.append(a)
    )
    task = asyncio.create_task(
        adapter.dispatch(FakeLlmNode(), _request("t0:0"), _ctx("t0", "t0:0"))
    )
    await asyncio.sleep(0)
    adapter.resolve(_credit_for(issuer.sent[0]), error="boom", cancelled=False)
    from aiperf.graph.credit_dispatch_adapter import GraphDispatchError

    with pytest.raises(GraphDispatchError):
        await asyncio.wait_for(task, timeout=1.0)
    assert drained == [adapter]


# ---------------------------------------------------------------------------
# context-overflow early-termination
# ---------------------------------------------------------------------------


async def test_overflow_error_raises_node_overflow_terminate():
    """A context-overflow error body resolves the parked Future with
    ``_NodeOverflowTerminate`` (NOT ``GraphDispatchError``), so the executor
    terminates the trajectory cleanly instead of unwinding it as a trace error.

    A non-final turn whose response is a context-overflow stops dispatching the
    rest of that trace's turns -- later turns carry more context and would only
    overflow too.
    """
    issuer = FakeIssuer()
    adapter = _make_adapter(issuer, "t0", {"t0:0": 0})
    task = asyncio.create_task(
        adapter.dispatch(FakeLlmNode(), _request("t0:0"), _ctx("t0", "t0:0"))
    )
    await asyncio.sleep(0)

    overflow_body = (
        '{"error": {"message": "This model\'s maximum context length is 8192 '
        'tokens (context_length_exceeded)", "code": "context_length_exceeded"}}'
    )
    adapter.resolve(_credit_for(issuer.sent[0]), error=overflow_body, cancelled=False)

    from aiperf.graph.context import _NodeExpectedExit
    from aiperf.graph.credit_dispatch_adapter import (
        GraphDispatchError,
        _NodeOverflowTerminate,
    )

    with pytest.raises(_NodeOverflowTerminate) as exc_info:
        await asyncio.wait_for(task, timeout=1.0)
    # Subclass of the executor's clean-exit sentinel so ``_run_node`` catches it
    # on the expected-exit branch (no TaskGroup error cascade).
    assert isinstance(exc_info.value, _NodeExpectedExit)
    # And NOT the generic error type, so it is not mistaken for a real failure.
    assert not isinstance(exc_info.value, GraphDispatchError)


async def test_non_overflow_error_still_raises_graph_dispatch_error():
    """A non-overflow error body keeps the existing ``GraphDispatchError`` path
    (unwinds the trace as a real error) -- overflow handling does not regress it."""
    issuer = FakeIssuer()
    adapter = _make_adapter(issuer, "t0", {"t0:0": 0})
    task = asyncio.create_task(
        adapter.dispatch(FakeLlmNode(), _request("t0:0"), _ctx("t0", "t0:0"))
    )
    await asyncio.sleep(0)
    adapter.resolve(
        _credit_for(issuer.sent[0]), error="HTTP 500 internal", cancelled=False
    )
    from aiperf.graph.credit_dispatch_adapter import (
        GraphDispatchError,
        _NodeOverflowTerminate,
    )

    with pytest.raises(GraphDispatchError) as exc_info:
        await asyncio.wait_for(task, timeout=1.0)
    assert not isinstance(exc_info.value, _NodeOverflowTerminate)


# ---------------------------------------------------------------------------
# dag identity map: agent_depth + derived parent correlation id
# ---------------------------------------------------------------------------


async def test_dispatch_node_identity_map_stamps_depth_and_parent_corr():
    """A ``node_identity`` hit stamps the mapped depth and derives the parent
    NODE's correlation id from the same trajectory-scope corr ``_mint`` folds.

    The child node lives in scope ``child`` (turn 0); its triggering parent node
    lives in the ROOT scope ``t0`` (turn 0). The derived parent correlation id is
    therefore the root scope's ``{conversation_id}::{nonce}`` corr -- distinct
    from the child's own corr."""
    issuer = FakeIssuer()
    adapter = _make_adapter(
        issuer,
        "t0",
        {"child:0": 0, "t0:0": 1},
        node_identity={"child:0": (1, "t0:0"), "t0:0": (0, None)},
    )
    task = asyncio.create_task(
        adapter.dispatch(FakeLlmNode(), _request("child:0"), _ctx("t0", "child:0"))
    )
    await asyncio.sleep(0)

    turn = issuer.sent[0]
    assert turn.agent_depth == 1
    # The derived parent corr IS the trajectory-scope corr _mint folds for the
    # parent node's scope (root scope "t0"), a {conversation_id}::{nonce} shape.
    parent_x_corr, _, _ = adapter._mint("t0:0", 1)
    assert turn.parent_correlation_id == parent_x_corr
    assert parent_x_corr.startswith("t0::")
    # ... and distinct from the child's own trajectory corr.
    assert turn.parent_correlation_id != turn.x_correlation_id

    adapter.resolve(_credit_for(turn), error=None, cancelled=False)
    await task


async def test_dispatch_node_identity_parent_none_falls_back_to_ctor_parent():
    """A map hit with no parent node (root / pre-session child) keeps the
    constructor's ``parent_correlation_id`` fallback."""
    issuer = FakeIssuer()
    adapter = _make_adapter(
        issuer,
        "t0",
        {"t0:0": 0},
        node_identity={"t0:0": (1, None)},
        parent_correlation_id="legacy-parent",
    )
    task = asyncio.create_task(
        adapter.dispatch(FakeLlmNode(), _request("t0:0"), _ctx("t0", "t0:0"))
    )
    await asyncio.sleep(0)

    turn = issuer.sent[0]
    assert turn.agent_depth == 1
    assert turn.parent_correlation_id == "legacy-parent"

    adapter.resolve(_credit_for(turn), error=None, cancelled=False)
    await task


async def test_dispatch_node_identity_miss_uses_root_identity():
    """A node absent from the map dispatches with root-chain identity."""
    issuer = FakeIssuer()
    adapter = _make_adapter(
        issuer,
        "t0",
        {"t0:0": 0},
        node_identity={"other:0": (3, "p:0")},
    )
    task = asyncio.create_task(
        adapter.dispatch(FakeLlmNode(), _request("t0:0"), _ctx("t0", "t0:0"))
    )
    await asyncio.sleep(0)

    turn = issuer.sent[0]
    assert turn.agent_depth == 0
    assert turn.parent_correlation_id is None

    adapter.resolve(_credit_for(turn), error=None, cancelled=False)
    await task


async def test_dispatch_without_node_identity_map_byte_identical_to_today():
    """No map (weka/dynamo default) => depth 0 and the constructor's
    ``parent_correlation_id`` pass-through, exactly as before."""
    issuer = FakeIssuer()
    adapter = _make_adapter(
        issuer, "t0", {"t0:0": 0}, parent_correlation_id="pc-passthrough"
    )
    task = asyncio.create_task(
        adapter.dispatch(FakeLlmNode(), _request("t0:0"), _ctx("t0", "t0:0"))
    )
    await asyncio.sleep(0)

    turn = issuer.sent[0]
    assert turn.agent_depth == 0
    assert turn.parent_correlation_id == "pc-passthrough"

    adapter.resolve(_credit_for(turn), error=None, cancelled=False)
    await task
