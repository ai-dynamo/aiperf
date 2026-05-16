# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for credit tests."""

import pytest

from aiperf.credit.sticky_router import StickyCreditRouter, WorkerLoad
from aiperf.credit.structs import Credit


def stub_credit(credit_id: int, phase: str = "profiling") -> Credit:
    """Create a minimal Credit for unit tests that only need a credit_id."""
    return Credit(
        id=credit_id,
        phase=phase,
        conversation_id="test-conv",
        x_correlation_id=f"test-xcorr-{credit_id}",
        turn_index=0,
        num_turns=1,
        issued_at_ns=0,
    )


@pytest.fixture
def router_with_worker(run) -> StickyCreditRouter:
    """Router with one registered worker."""
    router = StickyCreditRouter(run=run, service_id="test-router")
    router._workers = {
        "worker-1": WorkerLoad(worker_id="worker-1", in_flight_credits=0)
    }
    return router
