# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Instance-keyed graph sticky routing + GraphTraceEnd lifecycle."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import msgspec
import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.issuer import CreditIssuer
from aiperf.credit.messages import GraphTraceEnd, RouterToWorkerMessage
from aiperf.credit.sticky_router import StickyCreditRouter
from tests.unit.credit.conftest import (
    make_graph_credit,
    make_linear_credit,
    make_sticky_router,
    sticky_owner,
)


def _placements(router: StickyCreditRouter) -> set[str]:
    """Distinct worker ids the router has sent anything to."""
    return {call[0][0] for call in router._router_client.send_to.call_args_list}


class TestGraphStickyPlacement:
    @pytest.mark.asyncio
    async def test_instance_pins_one_worker_despite_final_turns(
        self, benchmark_run
    ) -> None:
        """Every credit of an instance lands on one worker under one instance-keyed session."""
        # Each graph credit looks like a final turn (num_turns=1), which in the
        # linear path would close the session after the first send.
        router = make_sticky_router(benchmark_run, ["worker-A", "worker-B"])

        for turn in range(3):
            await router.send_credit(
                make_graph_credit(turn, corr="t-1::c1", node_turn=turn)
            )

        worker_id = sticky_owner(router, "t-1::inst0")
        assert _placements(router) == {worker_id}
        assert router._workers[worker_id].active_sessions == 1
        assert "t-1::inst0" in router._workers[worker_id].active_session_ids

    @pytest.mark.asyncio
    async def test_distinct_instances_balance_across_workers(
        self, benchmark_run
    ) -> None:
        """Separate trace instances are placed independently rather than pinned together."""
        router = make_sticky_router(benchmark_run, ["worker-A", "worker-B"])

        await router.send_credit(make_graph_credit(1, corr="t-1::c1"))
        await router.send_credit(
            make_graph_credit(2, corr="t-2::c2", trace_id="t-2::inst0")
        )

        assert sticky_owner(router, "t-1::inst0") != sticky_owner(router, "t-2::inst0")

    @pytest.mark.asyncio
    async def test_trajectories_of_one_instance_co_place(self, benchmark_run) -> None:
        """Distinct trajectories of one instance share a worker AND a single session."""
        router = make_sticky_router(benchmark_run, ["worker-A", "worker-B"])

        await router.send_credit(make_graph_credit(1, corr="t-1::root-c"))
        await router.send_credit(make_graph_credit(2, corr="t-1::child-c"))

        worker_id = sticky_owner(router, "t-1::inst0")
        assert _placements(router) == {worker_id}
        # ONE session for the whole instance, not one per trajectory.
        assert router._workers[worker_id].active_sessions == 1

    @pytest.mark.asyncio
    async def test_instance_reroutes_after_worker_death(self, benchmark_run) -> None:
        """Unregistering the sticky worker frees the instance to re-place on a survivor."""
        router = make_sticky_router(benchmark_run, ["worker-A", "worker-B"])
        await router.send_credit(make_graph_credit(1, corr="t-1::root-c"))
        dead_worker = sticky_owner(router, "t-1::inst0")

        router._unregister_worker(dead_worker)
        assert "t-1::inst0" not in router._sticky_sessions

        await router.send_credit(make_graph_credit(2, corr="t-1::child-c"))
        survivor = ({"worker-A", "worker-B"} - {dead_worker}).pop()
        assert sticky_owner(router, "t-1::inst0") == survivor

    @pytest.mark.asyncio
    async def test_linear_final_turn_creates_no_session(self, benchmark_run) -> None:
        """A single-turn non-graph credit opens no sticky session at all."""
        router = make_sticky_router(benchmark_run, ["worker-A"])

        await router.send_credit(make_linear_credit(1, corr="c1", turn=0, num_turns=1))

        assert router._sticky_sessions == {}
        assert router._workers["worker-A"].active_sessions == 0

    @pytest.mark.asyncio
    async def test_linear_multi_turn_lifecycle_unchanged(self, benchmark_run) -> None:
        """A multi-turn non-graph session opens on turn 0 and closes on its final turn."""
        router = make_sticky_router(benchmark_run, ["worker-A"])

        await router.send_credit(make_linear_credit(1, corr="c1", turn=0, num_turns=2))
        assert "c1" in router._sticky_sessions
        await router.send_credit(make_linear_credit(2, corr="c1", turn=1, num_turns=2))
        assert "c1" not in router._sticky_sessions
        assert router._workers["worker-A"].active_sessions == 0


class TestGraphTraceEndLifecycle:
    @pytest.mark.asyncio
    async def test_end_closes_instance_session_and_forwards_to_worker(
        self, benchmark_run
    ) -> None:
        """One end call closes every trajectory and sends the worker exactly one GraphTraceEnd."""
        router = make_sticky_router(benchmark_run, ["worker-A"])
        await router.send_credit(make_graph_credit(1, corr="t-1::root-c"))
        await router.send_credit(make_graph_credit(2, corr="t-1::child-c"))
        router._router_client.send_to.reset_mock()

        await router.end_graph_trace("t-1::inst0")

        assert "t-1::inst0" not in router._sticky_sessions
        assert router._workers["worker-A"].active_sessions == 0
        assert "t-1::inst0" not in router._workers["worker-A"].active_session_ids
        router._router_client.send_to.assert_called_once()
        worker_id, message = router._router_client.send_to.call_args[0]
        assert worker_id == "worker-A"
        assert message == GraphTraceEnd(trace_id="t-1::inst0")

    @pytest.mark.asyncio
    async def test_end_is_idempotent_and_no_session_is_noop(
        self, benchmark_run
    ) -> None:
        """Re-ending a closed instance or ending an unknown one sends nothing."""
        router = make_sticky_router(benchmark_run, ["worker-A"])
        await router.send_credit(make_graph_credit(1, corr="t-1::c1"))
        await router.end_graph_trace("t-1::inst0")
        router._router_client.send_to.reset_mock()

        await router.end_graph_trace("t-1::inst0")
        await router.end_graph_trace("t-9::never")

        router._router_client.send_to.assert_not_called()
        assert router._workers["worker-A"].active_sessions == 0

    @pytest.mark.asyncio
    async def test_end_after_worker_unregistered_is_safe(self, benchmark_run) -> None:
        """Ending an instance whose worker already died drops the session without a send."""
        router = make_sticky_router(benchmark_run, ["worker-A", "worker-B"])
        await router.send_credit(make_graph_credit(1, corr="t-1::c1"))
        router._unregister_worker(sticky_owner(router, "t-1::inst0"))
        router._router_client.send_to.reset_mock()

        await router.end_graph_trace("t-1::inst0")

        assert "t-1::inst0" not in router._sticky_sessions
        router._router_client.send_to.assert_not_called()

    @pytest.mark.asyncio
    async def test_unregister_cleans_graph_sessions(self, benchmark_run) -> None:
        """Worker unregistration evicts the graph sticky sessions it owned."""
        router = make_sticky_router(benchmark_run, ["worker-A"])
        await router.send_credit(make_graph_credit(1, corr="t-1::c1"))

        router._unregister_worker("worker-A")

        assert "t-1::inst0" not in router._sticky_sessions

    @pytest.mark.asyncio
    async def test_same_instance_after_end_creates_fresh_session(
        self, benchmark_run
    ) -> None:
        """A credit arriving after an end re-opens the instance session from scratch."""
        router = make_sticky_router(benchmark_run, ["worker-A"])
        await router.send_credit(make_graph_credit(1, corr="t-1::c1"))
        await router.end_graph_trace("t-1::inst0")

        await router.send_credit(make_graph_credit(2, corr="t-1::c2", node_turn=1))

        assert sticky_owner(router, "t-1::inst0") == "worker-A"
        assert router._workers["worker-A"].active_sessions == 1


def _issuer(router: MagicMock) -> CreditIssuer:
    """Issuer wired to ``router`` with everything else mocked out."""
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
    @pytest.mark.asyncio
    async def test_end_graph_trace_forwards_to_router(self) -> None:
        """The issuer delegates trace ends straight to the router, unmodified."""
        router = MagicMock()
        router.send_credit = AsyncMock()
        router.end_graph_trace = AsyncMock()
        issuer = _issuer(router)

        await issuer.end_graph_trace("t-1::inst0")

        router.end_graph_trace.assert_awaited_once_with("t-1::inst0")


@pytest.mark.asyncio
async def test_graph_trace_end_survives_msgpack_roundtrip() -> None:
    """GraphTraceEnd decodes back to itself through the RouterToWorkerMessage union."""
    msg = GraphTraceEnd(trace_id="t-1::inst0")
    decoded = msgspec.msgpack.Decoder(RouterToWorkerMessage).decode(
        msgspec.msgpack.Encoder().encode(msg)
    )
    assert decoded == msg
