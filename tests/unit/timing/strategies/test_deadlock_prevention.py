# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable

import pytest

from tests.unit.timing.conftest import OrchestratorHarness


@pytest.mark.asyncio
class TestDeadlockPrevention:
    @pytest.mark.parametrize(
        "concurrency,num_sessions,request_count,convs,expected",
        [
            (None, 3, None, [("c1", 1), ("c2", 1), ("c3", 1)], 3),
            (None, None, 3, [("c1", 1), ("c2", 1), ("c3", 1)], 3),
            (10, 3, None, [("c1", 1), ("c2", 1), ("c3", 1)], 3),
            (None, 2, None, [("c1", 3), ("c2", 2)], 5),
            (10, 2, None, [("c1", 3), ("c2", 2)], 5),
            (1, None, 3, [("c1", 5), ("c2", 5)], 3),
            (10, 2, None, [("c1", 3), ("c2", 3)], 6),
        ],
    )  # fmt: skip
    async def test_completes_correctly(
        self,
        mock_orchestrator: Callable[..., OrchestratorHarness],
        concurrency: int | None,
        num_sessions: int | None,
        request_count: int | None,
        convs: list[tuple[str, int]],
        expected: int,
    ) -> None:
        orch = mock_orchestrator(
            convs,
            num_sessions=num_sessions,
            request_count=request_count,
            concurrency=concurrency,
            request_rate=1000.0,
        )
        await orch.run_with_auto_return()
        assert len(orch.sent_credits) == expected

    async def test_num_sessions_limits_unique_sessions(
        self, mock_orchestrator: Callable[..., OrchestratorHarness]
    ) -> None:
        orch = mock_orchestrator(
            [("c1", 3), ("c2", 3)],
            num_sessions=2,
            concurrency=10,
            request_rate=1000.0,
        )
        await orch.run_with_auto_return()
        assert len(orch.sent_credits) == 6
        assert len({c.x_correlation_id for c in orch.sent_credits}) == 2
