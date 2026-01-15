# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import pytest

from aiperf.common.enums import CreditPhase, TimingMode
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
    async def test_basic_execution(
        self, create_orchestrator_harness, convs, rate, users, count, expected
    ) -> None:
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=convs,
            user_centric_rate=rate,
            num_users=users,
            request_count=count,
        )
        await h.run_with_auto_return()
        assert len(h.sent_credits) >= expected


@pytest.mark.asyncio
class TestMassiveGauntlet:
    @pytest.mark.parametrize("qps", [float(i) for i in range(5, 101)])
    async def test_qps_5_to_100(self, create_orchestrator_harness, qps) -> None:
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=TWO_TURN * 4,
            user_centric_rate=qps,
            num_users=5,
            request_count=5,
        )
        await h.run_with_auto_return()
        assert len(h.sent_credits) == 5

    @pytest.mark.parametrize(
        "users,qps",
        [(u, q) for u in range(2, 21) for q in [10.0, 20.0, 50.0, 100.0]],
    )  # fmt: skip
    async def test_users_2_to_20(self, create_orchestrator_harness, users, qps) -> None:
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=TWO_TURN * 10,
            user_centric_rate=qps,
            num_users=users,
            request_count=users,
        )
        await h.run_with_auto_return()
        assert len(h.sent_credits) == users

    @pytest.mark.parametrize(
        "turns,users,qps",
        [(t, u, q) for t in [2, 3, 4, 5, 6, 8, 10] for u in [3, 5, 10] for q in [20.0, 50.0, 100.0]],
    )  # fmt: skip
    async def test_varying_turns(
        self, create_orchestrator_harness, turns, users, qps
    ) -> None:
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=[(f"c{i}", turns) for i in range(users * 3)],
            user_centric_rate=qps,
            num_users=users,
            request_count=users,
        )
        await h.run_with_auto_return()
        assert len(h.sent_credits) == users

    @pytest.mark.parametrize(
        "qps",
        [5.5, 7.5, 12.5, 15.5, 22.5, 27.5, 33.5, 42.5, 55.5, 67.5, 88.5, 99.5],
    )  # fmt: skip
    async def test_fractional_qps(self, create_orchestrator_harness, qps) -> None:
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=TWO_TURN * 4,
            user_centric_rate=qps,
            num_users=5,
            request_count=5,
        )
        await h.run_with_auto_return()
        assert len(h.sent_credits) == 5

    @pytest.mark.parametrize(
        "users,qps,turns",
        [
            (1, 10.0, 2), (1, 100.0, 5), (50, 200.0, 2), (20, 150.0, 3),
            (15, 75.0, 4), (8, 40.0, 6), (12, 60.0, 3), (25, 125.0, 2),
            (2, 5.0, 2), (3, 7.5, 3), (4, 12.0, 4), (7, 21.0, 5),
        ],
    )  # fmt: skip
    async def test_edge_cases(
        self, create_orchestrator_harness, users, qps, turns
    ) -> None:
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=[(f"c{i}", turns) for i in range(users * 3)],
            user_centric_rate=qps,
            num_users=users,
            request_count=users,
        )
        await h.run_with_auto_return()
        assert len(h.sent_credits) == users

    @pytest.mark.parametrize(
        "turns,qps",
        [(t, q) for t in range(2, 12) for q in range(10, 101, 5)],
    )  # fmt: skip
    async def test_turns_2_to_11(self, create_orchestrator_harness, turns, qps) -> None:
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=[(f"c{i}", turns) for i in range(15)],
            user_centric_rate=float(qps),
            num_users=5,
            request_count=5,
        )
        await h.run_with_auto_return()
        assert len(h.sent_credits) == 5

    @pytest.mark.parametrize(
        "users,qps",
        [(u, q) for u in range(1, 31) for q in range(20, 101, 20)],
    )  # fmt: skip
    async def test_users_1_to_30(self, create_orchestrator_harness, users, qps) -> None:
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=TWO_TURN * 15,
            user_centric_rate=float(qps),
            num_users=users,
            request_count=users,
        )
        await h.run_with_auto_return()
        assert len(h.sent_credits) == users

    @pytest.mark.parametrize(
        "users,qps,turns",
        [(u, q, t) for u in [2, 5, 10, 15, 20] for q in [15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0] for t in [2, 3, 5]],
    )  # fmt: skip
    async def test_comprehensive(
        self, create_orchestrator_harness, users, qps, turns
    ) -> None:
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=[(f"c{i}", turns) for i in range(users * 3)],
            user_centric_rate=qps,
            num_users=users,
            request_count=users,
        )
        await h.run_with_auto_return()
        assert len(h.sent_credits) == users

    @pytest.mark.parametrize(
        "users,qps,turns",
        [(u, q, t) for u in range(2, 12) for q in range(10, 81, 10) for t in [2, 4, 6]],
    )  # fmt: skip
    async def test_ultra_comprehensive(
        self, create_orchestrator_harness, users, qps, turns
    ) -> None:
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=[(f"c{i}", turns) for i in range(users * 3)],
            user_centric_rate=float(qps),
            num_users=users,
            request_count=users,
        )
        await h.run_with_auto_return()
        assert len(h.sent_credits) == users


@pytest.mark.asyncio
class TestSessionTracking:
    async def test_unique_correlation_ids(self, create_orchestrator_harness) -> None:
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
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=TWO_TURN * 10,
            user_centric_rate=50.0,
            num_users=10,
            request_count=25,
        )
        await h.run_with_auto_return()
        assert len(h.sent_credits) == 25

    async def test_stops_at_session_count(self, create_orchestrator_harness) -> None:
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
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=[(f"c{i}", 5) for i in range(30)],
            user_centric_rate=40.0,
            num_users=20,
            request_count=100,
        )
        await h.run_with_auto_return()
        assert len(h.sent_credits) >= 100


class TestUserClass:
    def test_x_correlation_id(self) -> None:
        m = MagicMock()
        m.x_correlation_id = "test-id"
        assert User(user_id=1, sampled=m).x_correlation_id == "test-id"

    def test_build_first_turn(self) -> None:
        m = MagicMock()
        m.build_first_turn.return_value = "turn"
        u = User(user_id=1, sampled=m, max_turns=5)
        assert u.build_first_turn() == "turn"
        m.build_first_turn.assert_called_once_with(max_turns=5)

    @pytest.mark.parametrize("uid", range(1, 22))
    def test_dataclass_creation(self, uid) -> None:
        m = MagicMock()
        m.x_correlation_id = f"c-{uid}"
        u = User(user_id=uid, sampled=m, next_send_time=1000, max_turns=3)
        assert u.user_id == uid
        assert u.sampled == m
        assert u.next_send_time == 1000
        assert u.max_turns == 3


@pytest.mark.asyncio
class TestSessionCountStopping:
    @pytest.mark.parametrize(
        "users,sessions",
        [(2, 4), (2, 6), (3, 6), (3, 9), (4, 8), (5, 10), (5, 15)],
    )  # fmt: skip
    async def test_various_ratios(
        self, create_orchestrator_harness, users, sessions
    ) -> None:
        h: OrchestratorHarness = create_orchestrator_harness(
            conversations=TWO_TURN * (sessions + 5),
            user_centric_rate=100.0,
            num_users=users,
            num_sessions=sessions,
        )
        await h.run_with_auto_return()
        assert len(h.sent_credits) >= sessions
