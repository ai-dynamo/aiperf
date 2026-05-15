# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for credit tests."""

import pytest

from aiperf.credit.sticky_router import StickyCreditRouter, WorkerLoad


@pytest.fixture
def router_with_worker(benchmark_run) -> StickyCreditRouter:
    """Router with one registered worker."""
    router = StickyCreditRouter(run=benchmark_run, service_id="test-router")
    router._workers = {
        "worker-1": WorkerLoad(worker_id="worker-1", in_flight_credits=0)
    }
    return router
