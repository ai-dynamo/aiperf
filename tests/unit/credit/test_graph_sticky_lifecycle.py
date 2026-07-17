# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Instance-keyed graph sticky routing + GraphTraceEnd lifecycle.

Linear sessions key on ``x_correlation_id`` (legacy contract); graph credits
(``trace_id`` set) key their session on the trace INSTANCE id, so EVERY
trajectory of one instance shares one session -- and therefore one worker --
because dynamic-slot capture/splice pools and spawn state are worker-local
and keyed by the instance. Graph credits mint ``turn_index`` per node, so a
trajectory's recorded final turn can complete while spawned work is still in
flight -- turn counting cannot express trace lifecycle. These tests pin the
router rules that make graph stickiness work:

1. graph credits create their instance session on first sight and are NOT
   torn down by the final-turn cleanup (without the gate, the session is
   destroyed on the same credit that creates it -- invisible to
   accounting-only assertions, hence the multi-credit PLACEMENT assertions);
2. every trajectory (fresh corr, same instance) joins the instance session;
3. ``end_graph_trace`` owns the close: ONE call per instance, synchronous
   state mutation, idempotent, dead-worker safe, worker-forward only when a
   session existed;
4. linear (no ``trace_id``) credits keep byte-identical behavior.

Plus the issuer side: the ``end_graph_trace`` passthrough to the router.
(Template-hash URL affinity is pinned in ``test_issuer_graph_path.py``.)
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.issuer import CreditIssuer
from aiperf.credit.messages import GraphTraceEnd
from aiperf.credit.sticky_router import StickyCreditRouter
from aiperf.credit.structs import Credit


def _graph_credit(
    credit_id: int,
    *,
    corr: str,
    node_turn: int = 0,
    trace_id: str = "t-1::inst0",
) -> Credit:
    """A graph credit as the adapter mints it: per-trajectory corr, per-node
    turn_index, num_turns=1 (every fire looks final to turn counting)."""
    return Credit(
        id=credit_id,
        phase=CreditPhase.PROFILING,
        conversation_id=trace_id.split("::", 1)[0],
        x_correlation_id=corr,
        turn_index=node_turn,
        num_turns=1,
        issued_at_ns=0,
        trace_id=trace_id,
        node_ordinal=node_turn,
    )


def _linear_credit(credit_id: int, *, corr: str, turn: int, num_turns: int) -> Credit:
    return Credit(
        id=credit_id,
        phase=CreditPhase.PROFILING,
        conversation_id="conv",
        x_correlation_id=corr,
        turn_index=turn,
        num_turns=num_turns,
        issued_at_ns=0,
    )


def _router(benchmark_run, workers: list[str]) -> StickyCreditRouter:
    router = StickyCreditRouter(run=benchmark_run, service_id="test-router")
    router._router_client.send_to = AsyncMock()
    for w in workers:
        router._register_worker(w)
    return router


class TestGraphStickyPlacement:
    async def test_instance_pins_one_worker_despite_final_turns(
        self, benchmark_run
    ) -> None:
        """The placement assertion: N final-looking credits, ONE worker,
        ONE instance-keyed session."""
        router = _router(benchmark_run, ["worker-A", "worker-B"])

        for turn in range(3):
            await router.send_credit(
                _graph_credit(turn, corr="t-1::c1", node_turn=turn)
            )

        placements = {
            call[0][0] for call in router._router_client.send_to.call_args_list
        }
        assert len(placements) == 1
        assert router._sticky_sessions.get("t-1::inst0") in placements
        worker_id = router._sticky_sessions["t-1::inst0"]
        assert router._workers[worker_id].active_sessions == 1
        assert "t-1::inst0" in router._workers[worker_id].active_session_ids

    async def test_distinct_instances_balance_across_workers(
        self, benchmark_run
    ) -> None:
        router = _router(benchmark_run, ["worker-A", "worker-B"])

        await router.send_credit(_graph_credit(1, corr="t-1::c1"))
        await router.send_credit(
            _graph_credit(2, corr="t-2::c2", trace_id="t-2::inst0")
        )

        assert (
            router._sticky_sessions["t-1::inst0"]
            != router._sticky_sessions["t-2::inst0"]
        )

    async def test_trajectories_of_one_instance_co_place(self, benchmark_run) -> None:
        """INSTANCE co-placement: a child trajectory (own corr, same
        credit.trace_id) shares the instance session -- worker-local
        dynamic-slot/spawn state must be reachable from every trajectory."""
        router = _router(benchmark_run, ["worker-A", "worker-B"])

        await router.send_credit(_graph_credit(1, corr="t-1::root-c"))
        await router.send_credit(_graph_credit(2, corr="t-1::child-c"))

        placements = {
            call[0][0] for call in router._router_client.send_to.call_args_list
        }
        assert len(placements) == 1
        worker_id = router._sticky_sessions["t-1::inst0"]
        assert placements == {worker_id}
        # ONE session for the whole instance, not one per trajectory.
        assert router._workers[worker_id].active_sessions == 1

    async def test_instance_reroutes_after_worker_death(self, benchmark_run) -> None:
        """A dead sticky worker releases the instance session: the next
        credit re-places on a survivor instead of raising or routing dead."""
        router = _router(benchmark_run, ["worker-A", "worker-B"])
        await router.send_credit(_graph_credit(1, corr="t-1::root-c"))
        dead_worker = router._sticky_sessions["t-1::inst0"]

        router._unregister_worker(dead_worker)
        assert "t-1::inst0" not in router._sticky_sessions

        await router.send_credit(_graph_credit(2, corr="t-1::child-c"))
        survivor = ({"worker-A", "worker-B"} - {dead_worker}).pop()
        assert router._sticky_sessions["t-1::inst0"] == survivor

    async def test_linear_final_turn_creates_no_session(self, benchmark_run) -> None:
        """Existing linear behavior is byte-identical (no trace_id)."""
        router = _router(benchmark_run, ["worker-A"])

        await router.send_credit(_linear_credit(1, corr="c1", turn=0, num_turns=1))

        assert router._sticky_sessions == {}
        assert router._workers["worker-A"].active_sessions == 0

    async def test_linear_multi_turn_lifecycle_unchanged(self, benchmark_run) -> None:
        router = _router(benchmark_run, ["worker-A"])

        await router.send_credit(_linear_credit(1, corr="c1", turn=0, num_turns=2))
        assert "c1" in router._sticky_sessions
        await router.send_credit(_linear_credit(2, corr="c1", turn=1, num_turns=2))
        assert "c1" not in router._sticky_sessions
        assert router._workers["worker-A"].active_sessions == 0


class TestGraphTraceEndLifecycle:
    async def test_end_closes_instance_session_and_forwards_to_worker(
        self, benchmark_run
    ) -> None:
        """ONE end call closes the whole instance (all trajectories) and the
        worker receives exactly ONE GraphTraceEnd."""
        router = _router(benchmark_run, ["worker-A"])
        await router.send_credit(_graph_credit(1, corr="t-1::root-c"))
        await router.send_credit(_graph_credit(2, corr="t-1::child-c"))
        router._router_client.send_to.reset_mock()

        await router.end_graph_trace("t-1::inst0", "profiling")

        assert "t-1::inst0" not in router._sticky_sessions
        assert router._workers["worker-A"].active_sessions == 0
        assert "t-1::inst0" not in router._workers["worker-A"].active_session_ids
        router._router_client.send_to.assert_called_once()
        worker_id, message = router._router_client.send_to.call_args[0]
        assert worker_id == "worker-A"
        assert message == GraphTraceEnd(
            trace_id="t-1::inst0", phase_variant="profiling"
        )

    async def test_end_is_idempotent_and_no_session_is_noop(
        self, benchmark_run
    ) -> None:
        router = _router(benchmark_run, ["worker-A"])
        await router.send_credit(_graph_credit(1, corr="t-1::c1"))
        await router.end_graph_trace("t-1::inst0", "profiling")
        router._router_client.send_to.reset_mock()

        await router.end_graph_trace("t-1::inst0", "profiling")
        await router.end_graph_trace("t-9::never", "profiling")

        router._router_client.send_to.assert_not_called()
        assert router._workers["worker-A"].active_sessions == 0

    async def test_end_after_worker_unregistered_is_safe(self, benchmark_run) -> None:
        router = _router(benchmark_run, ["worker-A", "worker-B"])
        await router.send_credit(_graph_credit(1, corr="t-1::c1"))
        worker_id = router._sticky_sessions["t-1::inst0"]
        router._unregister_worker(worker_id)
        router._router_client.send_to.reset_mock()

        await router.end_graph_trace("t-1::inst0", "profiling")

        assert "t-1::inst0" not in router._sticky_sessions
        router._router_client.send_to.assert_not_called()

    async def test_unregister_cleans_graph_sessions(self, benchmark_run) -> None:
        router = _router(benchmark_run, ["worker-A"])
        await router.send_credit(_graph_credit(1, corr="t-1::c1"))

        router._unregister_worker("worker-A")

        assert "t-1::inst0" not in router._sticky_sessions

    async def test_same_instance_after_end_creates_fresh_session(
        self, benchmark_run
    ) -> None:
        router = _router(benchmark_run, ["worker-A"])
        await router.send_credit(_graph_credit(1, corr="t-1::c1"))
        await router.end_graph_trace("t-1::inst0", "profiling")

        await router.send_credit(_graph_credit(2, corr="t-1::c2", node_turn=1))

        assert router._sticky_sessions["t-1::inst0"] == "worker-A"
        assert router._workers["worker-A"].active_sessions == 1


def _issuer(router: MagicMock) -> CreditIssuer:
    progress = MagicMock()
    progress.increment_sent = MagicMock(return_value=(1, False))
    progress.all_credits_sent_event = asyncio.Event()
    lifecycle = MagicMock()
    lifecycle.started_at_ns = time.time_ns()
    lifecycle.started_at_perf_ns = time.perf_counter_ns()
    cancellation = MagicMock()
    cancellation.next_cancellation_delay_ns = MagicMock(return_value=None)
    return CreditIssuer(
        phase=CreditPhase.PROFILING,
        stop_checker=MagicMock(),
        progress=progress,
        concurrency_manager=MagicMock(),
        credit_router=router,
        cancellation_policy=cancellation,
        lifecycle=lifecycle,
        url_selection_strategy=None,
    )


class TestIssuerGraphTraceEndPassthrough:
    async def test_end_graph_trace_forwards_to_router(self) -> None:
        router = MagicMock()
        router.send_credit = AsyncMock()
        router.end_graph_trace = AsyncMock()
        issuer = _issuer(router)

        await issuer.end_graph_trace("t-1::inst0", "profiling")

        router.end_graph_trace.assert_awaited_once_with("t-1::inst0", "profiling")


@pytest.mark.asyncio
async def test_graph_trace_end_survives_msgpack_roundtrip() -> None:
    import msgspec

    from aiperf.credit.messages import RouterToWorkerMessage

    msg = GraphTraceEnd(trace_id="t-1::inst0", phase_variant="profiling")
    decoded = msgspec.msgpack.Decoder(RouterToWorkerMessage).decode(
        msgspec.msgpack.Encoder().encode(msg)
    )
    assert decoded == msg
