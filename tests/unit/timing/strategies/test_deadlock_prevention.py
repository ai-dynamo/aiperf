# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable

import pytest

from tests.unit.timing.conftest import OrchestratorHarness


@pytest.mark.asyncio
class TestStopConditions:
    """Tests that orchestrator stops correctly based on various limits."""

    @pytest.mark.parametrize(
        "num_sessions,request_count,convs,expected_credits",
        [
            (3, None, [("c1", 1), ("c2", 1), ("c3", 1)], 3),
            (None, 3, [("c1", 5), ("c2", 5)], 3),
            (2, None, [("c1", 3), ("c2", 2)], 5),
        ],
    )  # fmt: skip
    async def test_respects_stop_limits(
        self,
        create_orchestrator_harness: Callable[..., OrchestratorHarness],
        num_sessions: int | None,
        request_count: int | None,
        convs: list[tuple[str, int]],
        expected_credits: int,
    ) -> None:
        """Orchestrator sends correct number of credits based on session or request limits."""
        orch = create_orchestrator_harness(
            conversations=convs,
            num_sessions=num_sessions,
            request_count=request_count,
            concurrency=10,
            request_rate=1000.0,
        )
        await orch.run_with_auto_return()
        assert len(orch.sent_credits) == expected_credits

    async def test_num_sessions_limits_unique_sessions(
        self, create_orchestrator_harness: Callable[..., OrchestratorHarness]
    ) -> None:
        """With num_sessions=2, only 2 unique session IDs should be used across all credits."""
        orch = create_orchestrator_harness(
            conversations=[("c1", 3), ("c2", 3)],
            num_sessions=2,
            concurrency=10,
            request_rate=1000.0,
        )
        await orch.run_with_auto_return()
        assert len(orch.sent_credits) == 6
        assert len({c.x_correlation_id for c in orch.sent_credits}) == 2
