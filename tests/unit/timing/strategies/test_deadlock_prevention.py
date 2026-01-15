# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for RequestRateStrategy deadlock prevention.

Uses real PhaseOrchestrator with mock credit router - only the transport
layer is mocked to capture credits and inject returns.
"""

from collections.abc import Callable

import pytest

from tests.unit.timing.conftest import OrchestratorHarness


@pytest.mark.asyncio
class TestSingleTurnExitsCleanly:
    """Single-turn conversations complete without deadlock."""

    @pytest.mark.parametrize(
        "concurrency,num_sessions,request_count",
        [
            (None, 3, None),  # Rate-centric with session limit
            (None, None, 3),  # Rate-centric with request limit
            (10, 3, None),  # Session-centric mode
        ],
    )  # fmt: skip
    async def test_single_turn_completes(
        self,
        mock_orchestrator: Callable[..., OrchestratorHarness],
        concurrency: int | None,
        num_sessions: int | None,
        request_count: int | None,
    ) -> None:
        orchestrator = mock_orchestrator(
            [("c1", 1), ("c2", 1), ("c3", 1)],
            num_sessions=num_sessions,
            request_count=request_count,
            concurrency=concurrency,
            request_rate=1000.0,
        )

        await orchestrator.run_with_auto_return()

        assert len(orchestrator.sent_credits) == 3


@pytest.mark.asyncio
class TestMultiTurnHandling:
    """Multi-turn conversations process all turns correctly."""

    # fmt: skip
    @pytest.mark.parametrize(
        "concurrency",
        [
            None,  # Rate-centric mode
            10,  # Session-centric mode
        ],
    )
    async def test_processes_all_turns(
        self,
        mock_orchestrator: Callable[..., OrchestratorHarness],
        concurrency: int | None,
    ) -> None:
        orchestrator = mock_orchestrator(
            [("c1", 3), ("c2", 2)],
            num_sessions=2,
            concurrency=concurrency,
            request_rate=1000.0,
        )

        await orchestrator.run_with_auto_return()

        assert len(orchestrator.sent_credits) == 5


@pytest.mark.asyncio
class TestLimitSemantics:
    """Verify request_count vs num_sessions semantics."""

    async def test_request_count_limits_total_requests(
        self, mock_orchestrator: Callable[..., OrchestratorHarness]
    ) -> None:
        orchestrator = mock_orchestrator(
            [("c1", 5), ("c2", 5)],
            request_count=3,
            concurrency=1,
            request_rate=1000.0,
        )

        await orchestrator.run_with_auto_return()

        assert len(orchestrator.sent_credits) == 3

    async def test_num_sessions_allows_all_turns_within(
        self, mock_orchestrator: Callable[..., OrchestratorHarness]
    ) -> None:
        orchestrator = mock_orchestrator(
            [("c1", 3), ("c2", 3)],
            num_sessions=2,
            concurrency=10,
            request_rate=1000.0,
        )

        await orchestrator.run_with_auto_return()

        assert len(orchestrator.sent_credits) == 6

        sessions = {c.x_correlation_id for c in orchestrator.sent_credits}
        assert len(sessions) == 2
