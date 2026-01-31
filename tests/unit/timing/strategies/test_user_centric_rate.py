# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.strategies.user_centric_rate import User, UserCentricStrategy
from tests.unit.timing.conftest import OrchestratorHarness

TWO_TURN = [("c1", 2), ("c2", 2), ("c3", 2), ("c4", 2), ("c5", 2)]
MULTI_TURN = [("c1", 3), ("c2", 3), ("c3", 3), ("c4", 3)]


class TestUserCentricInit:
    @pytest.mark.parametrize(
        "num_users,rate,match",
        [(None, 10.0, "num_users must be set"), (5, None, "request_rate must be set")],
    )  # fmt: skip
    def test_missing_params_raises(self, num_users, rate, match) -> None:
        cfg = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.USER_CENTRIC_RATE,
            request_rate=rate,
            num_users=num_users,
            total_expected_requests=10,
        )
        with pytest.raises(ValueError, match=match):
            UserCentricStrategy(
                config=cfg,
                conversation_source=MagicMock(),
                scheduler=MagicMock(),
                stop_checker=MagicMock(),
                credit_issuer=MagicMock(),
                lifecycle=MagicMock(),
            )

    @pytest.mark.asyncio
    async def test_valid_config(self, create_orchestrator_harness) -> None:
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=TWO_TURN,
            user_centric_rate=10.0,
            num_users=5,
            request_count=10,
        )
        await h.orchestrator.initialize()
        assert len(h.orchestrator._ordered_phase_configs) == 1
        assert h.orchestrator._ordered_phase_configs[0].phase == CreditPhase.PROFILING


@pytest.mark.asyncio
class TestUserCentricExecution:
    @pytest.mark.parametrize(
        "convs,rate,users,count,expected",
        [
            (TWO_TURN * 10, 20.0, 5, 10, 10),
            (TWO_TURN * 4, 10.0, 10, 10, 10),
            (MULTI_TURN * 2, 20.0, 4, 4, 4),
            (TWO_TURN * 4, 50.0, 10, 25, 25),
            (MULTI_TURN * 3, 40.0, 8, 30, 30),
            (MULTI_TURN, 10.0, 1, 10, 10),
            (TWO_TURN * 20, 100.0, 50, 100, 100),
            (TWO_TURN * 2, 1.0, 2, 2, 2),
            (TWO_TURN * 100, 500.0, 100, 500, 500),
            (TWO_TURN, 10.0, 5, 5, 5),
            (TWO_TURN * 4, 20.0, 20, 10, 10),
            (TWO_TURN * 4, 40.0, 10, 10, 10),
        ],
    )  # fmt: skip
    async def test_issues_expected_credits(
        self, create_orchestrator_harness, convs, rate, users, count, expected
    ) -> None:
        """Verify strategy issues correct number of credits across various configurations."""
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=convs,
            user_centric_rate=rate,
            num_users=users,
            request_count=count,
        )
        await h.run_with_auto_return()
        assert len(h.sent_credits) >= expected


@pytest.mark.asyncio
class TestSessionTracking:
    async def test_unique_correlation_ids(self, create_orchestrator_harness) -> None:
        """Each user session should have a unique correlation ID."""
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=TWO_TURN * 4,
            user_centric_rate=25.0,
            num_users=5,
            request_count=15,
        )
        await h.run_with_auto_return()
        assert len({c.x_correlation_id for c in h.sent_credits}) >= 5

    async def test_multi_turn_shares_correlation(
        self, create_orchestrator_harness
    ) -> None:
        """Multiple turns in a session should share the same correlation ID with sequential turn indices."""
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=MULTI_TURN,
            user_centric_rate=20.0,
            num_users=4,
            request_count=20,
        )
        await h.run_with_auto_return()
        sessions: dict[str, list] = {}
        for c in h.sent_credits:
            sessions.setdefault(c.x_correlation_id, []).append(c)
        multi = [s for s in sessions.values() if len(s) > 1]
        assert len(multi) > 0
        for credits in multi:
            indices = [c.turn_index for c in credits]
            assert indices == sorted(indices)
            assert indices[0] == 0


@pytest.mark.asyncio
class TestStopConditions:
    async def test_stops_at_request_count(self, create_orchestrator_harness) -> None:
        """Strategy should stop issuing credits when request count is reached."""
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=TWO_TURN * 10,
            user_centric_rate=50.0,
            num_users=10,
            request_count=25,
        )
        await h.run_with_auto_return()
        assert len(h.sent_credits) == 25

    async def test_stops_at_session_count(self, create_orchestrator_harness) -> None:
        """Strategy should stop starting new sessions when session count is reached."""
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=MULTI_TURN * 5,
            user_centric_rate=40.0,
            num_users=8,
            num_sessions=10,
        )
        await h.run_with_auto_return()
        assert len([c for c in h.sent_credits if c.turn_index == 0]) == 10


@pytest.mark.asyncio
class TestRealisticScenarios:
    async def test_chat_benchmark(self, create_orchestrator_harness) -> None:
        """Simulate high-concurrency chat benchmark with many users and sessions."""
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=[(f"c{i}", 3) for i in range(100)],
            user_centric_rate=100.0,
            num_users=50,
            request_count=200,
        )
        await h.run_with_auto_return()
        assert len(h.sent_credits) >= 200
        assert 0 in {c.turn_index for c in h.sent_credits}

    async def test_kv_cache_benchmark(self, create_orchestrator_harness) -> None:
        """Simulate KV cache benchmark with longer multi-turn conversations."""
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=[(f"c{i}", 5) for i in range(30)],
            user_centric_rate=40.0,
            num_users=20,
            request_count=100,
        )
        await h.run_with_auto_return()
        assert len(h.sent_credits) >= 100


class TestUserClass:
    def test_x_correlation_id_delegates_to_sampled(self) -> None:
        """User.x_correlation_id should delegate to sampled session."""
        m = MagicMock()
        m.x_correlation_id = "test-id"
        assert User(user_id=1, sampled=m).x_correlation_id == "test-id"

    def test_build_first_turn_passes_max_turns(self) -> None:
        """User.build_first_turn should pass max_turns to sampled session."""
        m = MagicMock()
        m.build_first_turn.return_value = "turn"
        u = User(user_id=1, sampled=m, max_turns=5)
        assert u.build_first_turn() == "turn"
        m.build_first_turn.assert_called_once_with(max_turns=5)

    def test_dataclass_fields(self) -> None:
        """Verify User dataclass stores fields correctly."""
        m = MagicMock()
        m.x_correlation_id = "c-42"
        u = User(user_id=42, sampled=m, next_send_time=1000, max_turns=3, order=5)
        assert u.user_id == 42
        assert u.sampled == m
        assert u.next_send_time == 1000
        assert u.max_turns == 3
        assert u.order == 5
