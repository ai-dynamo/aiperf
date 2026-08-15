# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for credit tests."""

from unittest.mock import AsyncMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.sticky_router import StickyCreditRouter, WorkerLoad
from aiperf.credit.structs import Credit


@pytest.fixture
def router_with_worker(benchmark_run) -> StickyCreditRouter:
    """Router with one registered worker."""
    router = StickyCreditRouter(run=benchmark_run, service_id="test-router")
    router._workers = {
        "worker-1": WorkerLoad(worker_id="worker-1", in_flight_credits=0)
    }
    return router


# Single definition of the tree-finality scaffolding lives in the timing
# conftest; credit tests exercise the same issuer seam and reuse it rather
# than keeping a second copy in sync.
from tests.unit.timing.conftest import (  # noqa: E402
    CapturingRouter,
    FakeConcurrency,
    make_registry,
    make_tree_issuer,
)

__all__ = [
    "CapturingRouter",
    "FakeConcurrency",
    "make_registry",
    "make_tree_issuer",
    "router_with_worker",
]


def make_graph_credit(
    credit_id: int,
    *,
    corr: str,
    node_turn: int = 0,
    trace_id: str = "t-1::inst0",
) -> Credit:
    """A graph credit as the adapter mints it (instance-keyed via ``trace_id``)."""
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


def make_linear_credit(
    credit_id: int, *, corr: str, turn: int, num_turns: int
) -> Credit:
    """A non-graph credit (no ``trace_id``), keyed on its correlation id."""
    return Credit(
        id=credit_id,
        phase=CreditPhase.PROFILING,
        conversation_id="conv",
        x_correlation_id=corr,
        turn_index=turn,
        num_turns=num_turns,
        issued_at_ns=0,
    )


def sticky_owner(router: StickyCreditRouter, key: str) -> str | None:
    """Worker id owning the sticky session for ``key`` (entries hold a refcount)."""
    entry = router._sticky_sessions.get(key)
    return entry.worker_id if entry is not None else None


def make_sticky_router(benchmark_run, workers: list[str]) -> StickyCreditRouter:
    """Sticky router with a mocked send path and the given workers registered."""
    router = StickyCreditRouter(run=benchmark_run, service_id="test-router")
    router._router_client.send_to = AsyncMock()
    for worker in workers:
        router._register_worker(worker)
    return router
