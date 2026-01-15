# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for user-centric rate timing strategy."""

from unittest.mock import MagicMock

import pytest

from aiperf.common.enums import (
    CreditPhase,
    TimingMode,
)
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.strategies.user_centric_rate import User, UserCentricStrategy
from tests.unit.timing.conftest import OrchestratorHarness


@pytest.fixture
def two_turn_conversations():
    return [("conv1", 2), ("conv2", 2), ("conv3", 2), ("conv4", 2), ("conv5", 2)]


@pytest.fixture
def multi_turn_conversations():
    return [("conv1", 3), ("conv2", 3), ("conv3", 3), ("conv4", 3)]


class TestUserCentricStrategyInitialization:
    @pytest.mark.asyncio
    async def test_initialization_with_valid_config(
        self, create_orchestrator_harness, two_turn_conversations
    ) -> None:
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations,
            user_centric_rate=10.0,
            num_users=5,
            request_count=10,
        )

        await harness.orchestrator.initialize()

        assert len(harness.orchestrator._ordered_phase_configs) == 1
        assert (
            harness.orchestrator._ordered_phase_configs[0].phase
            == CreditPhase.PROFILING
        )

    def test_direct_init_requires_num_users(self) -> None:
        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.USER_CENTRIC_RATE,
            request_rate=10.0,
            num_users=None,
            total_expected_requests=10,
        )

        with pytest.raises(ValueError, match="num_users must be set"):
            UserCentricStrategy(
                config=config,
                conversation_source=MagicMock(),
                scheduler=MagicMock(),
                stop_checker=MagicMock(),
                credit_issuer=MagicMock(),
                lifecycle=MagicMock(),
            )

    def test_direct_init_requires_positive_request_rate(self) -> None:
        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.USER_CENTRIC_RATE,
            request_rate=None,
            num_users=5,
            total_expected_requests=10,
        )

        with pytest.raises(ValueError, match="request_rate must be set"):
            UserCentricStrategy(
                config=config,
                conversation_source=MagicMock(),
                scheduler=MagicMock(),
                stop_checker=MagicMock(),
                credit_issuer=MagicMock(),
                lifecycle=MagicMock(),
            )


class TestUserCentricSetupPhase:
    @pytest.mark.asyncio
    async def test_pre_generates_users_in_precise_mode(
        self, create_orchestrator_harness, two_turn_conversations
    ) -> None:
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * 10,
            user_centric_rate=20.0,
            num_users=5,
            request_count=10,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) == 10

    @pytest.mark.asyncio
    async def test_turn_gap_calculation(
        self, create_orchestrator_harness, two_turn_conversations
    ) -> None:
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * 4,
            user_centric_rate=10.0,
            num_users=10,
            request_count=10,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) == 10

    @pytest.mark.asyncio
    async def test_virtual_history_works_with_multi_turn(
        self, create_orchestrator_harness, multi_turn_conversations
    ) -> None:
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=multi_turn_conversations * 2,
            user_centric_rate=20.0,
            num_users=4,
            request_count=4,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) == 4


class TestPreciseModeExecution:
    @pytest.mark.asyncio
    async def test_precise_mode_basic_execution(
        self, create_orchestrator_harness, two_turn_conversations
    ) -> None:
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * 4,
            user_centric_rate=50.0,
            num_users=10,
            request_count=25,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) >= 25

    @pytest.mark.asyncio
    async def test_precise_mode_with_multi_turn(
        self, create_orchestrator_harness, multi_turn_conversations
    ) -> None:
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=multi_turn_conversations * 3,
            user_centric_rate=40.0,
            num_users=8,
            request_count=30,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) >= 30

        multi_turn_sessions = [c for c in harness.sent_credits if c.num_turns > 1]
        assert len(multi_turn_sessions) > 0

    @pytest.mark.asyncio
    async def test_precise_mode_single_user(
        self, create_orchestrator_harness, multi_turn_conversations
    ) -> None:
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=multi_turn_conversations,
            user_centric_rate=10.0,
            num_users=1,
            request_count=10,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) >= 10

    @pytest.mark.asyncio
    async def test_precise_mode_high_user_count(
        self, create_orchestrator_harness, two_turn_conversations
    ) -> None:
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * 20,
            user_centric_rate=100.0,
            num_users=50,
            request_count=100,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) >= 100


class TestUserCentricEdgeCases:
    @pytest.mark.asyncio
    async def test_very_low_qps(
        self, create_orchestrator_harness, two_turn_conversations
    ) -> None:
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * 2,
            user_centric_rate=1.0,
            num_users=2,
            request_count=2,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) == 2

    @pytest.mark.asyncio
    async def test_very_high_qps(
        self, create_orchestrator_harness, two_turn_conversations
    ) -> None:
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * 100,
            user_centric_rate=500.0,
            num_users=100,
            request_count=500,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) >= 500

    @pytest.mark.asyncio
    async def test_num_users_equals_request_count(
        self, create_orchestrator_harness, two_turn_conversations
    ) -> None:
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations,
            user_centric_rate=10.0,
            num_users=5,
            request_count=5,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) == 5

    @pytest.mark.asyncio
    async def test_num_users_greater_than_request_count(
        self, create_orchestrator_harness, two_turn_conversations
    ) -> None:
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * 4,
            user_centric_rate=20.0,
            num_users=20,
            request_count=10,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) == 10

    @pytest.mark.asyncio
    async def test_works_with_multi_turn_dataset(
        self, create_orchestrator_harness, two_turn_conversations
    ) -> None:
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * 4,
            user_centric_rate=40.0,
            num_users=10,
            request_count=10,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) == 10

        unique_users = {c.x_correlation_id for c in harness.sent_credits}
        assert len(unique_users) >= 1


class TestMassiveGauntlet:
    @pytest.mark.parametrize("qps", [float(i) for i in range(5, 101)])
    @pytest.mark.asyncio
    async def test_every_qps_from_5_to_100(
        self, create_orchestrator_harness, two_turn_conversations, qps
    ) -> None:
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * 4,
            user_centric_rate=qps,
            num_users=5,
            request_count=5,
        )

        await harness.run_with_auto_return()
        assert len(harness.sent_credits) == 5

    @pytest.mark.parametrize(
        "num_users,qps",  # fmt: skip
        [(u, q) for u in range(2, 21) for q in [10.0, 20.0, 50.0, 100.0]],
    )
    @pytest.mark.asyncio
    async def test_users_2_to_20_various_qps(
        self, create_orchestrator_harness, two_turn_conversations, num_users, qps
    ) -> None:
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * 10,
            user_centric_rate=qps,
            num_users=num_users,
            request_count=num_users,
        )

        await harness.run_with_auto_return()
        assert len(harness.sent_credits) == num_users

    @pytest.mark.parametrize(
        "turns,num_users,qps",  # fmt: skip
        [
            (t, u, q)
            for t in [2, 3, 4, 5, 6, 8, 10]
            for u in [3, 5, 10]
            for q in [20.0, 50.0, 100.0]
        ],
    )
    @pytest.mark.asyncio
    async def test_varying_turn_counts_gauntlet(
        self, create_orchestrator_harness, turns, num_users, qps
    ) -> None:
        conversations = [(f"conv{i}", turns) for i in range(num_users * 3)]

        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=conversations,
            user_centric_rate=qps,
            num_users=num_users,
            request_count=num_users,
        )

        await harness.run_with_auto_return()
        assert len(harness.sent_credits) == num_users

    @pytest.mark.parametrize(
        "qps",  # fmt: skip
        [
            5.5,
            7.5,
            12.5,
            15.5,
            22.5,
            27.5,
            33.5,
            42.5,
            55.5,
            67.5,
            88.5,
            99.5,
        ],
    )
    @pytest.mark.asyncio
    async def test_fractional_qps_values(
        self, create_orchestrator_harness, two_turn_conversations, qps
    ) -> None:
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * 4,
            user_centric_rate=qps,
            num_users=5,
            request_count=5,
        )

        await harness.run_with_auto_return()
        assert len(harness.sent_credits) == 5

    @pytest.mark.parametrize(
        "num_users,qps,turns",  # fmt: skip
        [
            (1, 10.0, 2),
            (1, 100.0, 5),
            (50, 200.0, 2),
            (20, 150.0, 3),
            (15, 75.0, 4),
            (8, 40.0, 6),
            (12, 60.0, 3),
            (25, 125.0, 2),
            (2, 5.0, 2),
            (3, 7.5, 3),
            (4, 12.0, 4),
            (7, 21.0, 5),
        ],
    )
    @pytest.mark.asyncio
    async def test_extreme_edge_case_combinations(
        self, create_orchestrator_harness, num_users, qps, turns
    ) -> None:
        conversations = [(f"conv{i}", turns) for i in range(num_users * 3)]

        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=conversations,
            user_centric_rate=qps,
            num_users=num_users,
            request_count=num_users,
        )

        await harness.run_with_auto_return()
        assert len(harness.sent_credits) == num_users

    @pytest.mark.parametrize(
        "turns,qps",  # fmt: skip
        [(t, q) for t in range(2, 12) for q in range(10, 101, 5)],
    )
    @pytest.mark.asyncio
    async def test_all_turn_counts_2_to_11_with_qps(
        self, create_orchestrator_harness, turns, qps
    ) -> None:
        conversations = [(f"conv{i}", turns) for i in range(15)]

        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=conversations,
            user_centric_rate=float(qps),
            num_users=5,
            request_count=5,
        )

        await harness.run_with_auto_return()
        assert len(harness.sent_credits) == 5

    @pytest.mark.parametrize(
        "num_users,qps",  # fmt: skip
        [(u, q) for u in range(1, 31) for q in range(20, 101, 20)],
    )
    @pytest.mark.asyncio
    async def test_users_1_to_30_qps_multiples_of_20(
        self, create_orchestrator_harness, two_turn_conversations, num_users, qps
    ) -> None:
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * 15,
            user_centric_rate=float(qps),
            num_users=num_users,
            request_count=num_users,
        )

        await harness.run_with_auto_return()
        assert len(harness.sent_credits) == num_users

    @pytest.mark.parametrize(
        "num_users,qps,turns",  # fmt: skip
        [
            (u, q, t)
            for u in [2, 5, 10, 15, 20]
            for q in [15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0]
            for t in [2, 3, 5]
        ],
    )
    @pytest.mark.asyncio
    async def test_comprehensive_combinations(
        self, create_orchestrator_harness, num_users, qps, turns
    ) -> None:
        conversations = [(f"conv{i}", turns) for i in range(num_users * 3)]

        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=conversations,
            user_centric_rate=qps,
            num_users=num_users,
            request_count=num_users,
        )

        await harness.run_with_auto_return()
        assert len(harness.sent_credits) == num_users

    @pytest.mark.parametrize(
        "num_users,qps,turns",  # fmt: skip
        [(u, q, t) for u in range(2, 12) for q in range(10, 81, 10) for t in [2, 4, 6]],
    )
    @pytest.mark.asyncio
    async def test_ultra_comprehensive_matrix(
        self, create_orchestrator_harness, num_users, qps, turns
    ) -> None:
        conversations = [(f"conv{i}", turns) for i in range(num_users * 3)]

        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=conversations,
            user_centric_rate=float(qps),
            num_users=num_users,
            request_count=num_users,
        )

        await harness.run_with_auto_return()
        assert len(harness.sent_credits) == num_users


class TestUserGenerationAndVirtualHistory:
    @pytest.mark.asyncio
    async def test_generates_initial_users(
        self, create_orchestrator_harness, multi_turn_conversations
    ) -> None:
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=multi_turn_conversations,
            user_centric_rate=20.0,
            num_users=4,
            request_count=15,
        )

        await harness.run_with_auto_return()

        unique_correlations = {c.x_correlation_id for c in harness.sent_credits}
        assert len(unique_correlations) >= 4

    @pytest.mark.asyncio
    async def test_virtual_history_works_correctly(
        self, create_orchestrator_harness, two_turn_conversations
    ) -> None:
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * 4,
            user_centric_rate=20.0,
            num_users=5,
            request_count=5,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) == 5


class TestUserCentricParameterVariations:
    @pytest.mark.parametrize(
        "num_users,qps,request_count",  # fmt: skip
        [
            (5, 10.0, 10),
            (10, 50.0, 15),
            (1, 5.0, 5),
        ],
    )
    @pytest.mark.asyncio
    async def test_various_configurations(
        self,
        create_orchestrator_harness,
        two_turn_conversations,
        num_users,
        qps,
        request_count,
    ) -> None:
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * 20,
            user_centric_rate=qps,
            num_users=num_users,
            request_count=request_count,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) == request_count

    @pytest.mark.parametrize(
        "num_users,turns_per_session",  # fmt: skip
        [
            (5, 1),
            (5, 3),
            (2, 5),
        ],
    )
    @pytest.mark.asyncio
    async def test_user_count_vs_turn_count_combinations(
        self,
        create_orchestrator_harness,
        num_users,
        turns_per_session,
    ) -> None:
        conversations = [(f"conv{i}", turns_per_session) for i in range(num_users * 2)]

        request_count = num_users

        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=conversations,
            user_centric_rate=float(num_users * 5),
            num_users=num_users,
            request_count=request_count,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) == request_count


class TestSessionTracking:
    @pytest.mark.asyncio
    async def test_unique_correlation_ids_per_user(
        self, create_orchestrator_harness, two_turn_conversations
    ) -> None:
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * 4,
            user_centric_rate=25.0,
            num_users=5,
            request_count=15,
        )

        await harness.run_with_auto_return()

        correlation_ids = [c.x_correlation_id for c in harness.sent_credits]

        unique_ids = set(correlation_ids)
        assert len(unique_ids) >= 5

    @pytest.mark.asyncio
    async def test_multi_turn_sessions_share_correlation_id(
        self, create_orchestrator_harness, multi_turn_conversations
    ) -> None:
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=multi_turn_conversations,
            user_centric_rate=20.0,
            num_users=4,
            request_count=20,
        )

        await harness.run_with_auto_return()

        sessions = {}
        for credit in harness.sent_credits:
            corr_id = credit.x_correlation_id
            if corr_id not in sessions:
                sessions[corr_id] = []
            sessions[corr_id].append(credit)

        multi_turn_sessions = [
            credits for credits in sessions.values() if len(credits) > 1
        ]

        assert len(multi_turn_sessions) > 0

        for session_credits in multi_turn_sessions:
            turn_indices = [c.turn_index for c in session_credits]
            assert turn_indices == sorted(turn_indices)
            assert turn_indices[0] == 0


class TestStopConditions:
    @pytest.mark.asyncio
    async def test_stops_at_request_count(
        self, create_orchestrator_harness, two_turn_conversations
    ) -> None:
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * 10,
            user_centric_rate=50.0,
            num_users=10,
            request_count=25,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) == 25

    @pytest.mark.asyncio
    async def test_stops_at_session_count(
        self, create_orchestrator_harness, multi_turn_conversations
    ) -> None:
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=multi_turn_conversations * 5,
            user_centric_rate=40.0,
            num_users=8,
            num_sessions=10,
        )

        await harness.run_with_auto_return()

        first_turns = [c for c in harness.sent_credits if c.turn_index == 0]
        assert len(first_turns) == 10


class TestRealisticScenarios:
    @pytest.mark.asyncio
    async def test_realistic_chat_benchmark(self, create_orchestrator_harness) -> None:
        conversations = [("conv" + str(i), 3) for i in range(100)]

        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=conversations,
            user_centric_rate=100.0,
            num_users=50,
            request_count=200,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) >= 200

        turn_indices = {c.turn_index for c in harness.sent_credits}
        assert 0 in turn_indices
        assert len(turn_indices) > 1

    @pytest.mark.asyncio
    async def test_kv_cache_benchmark_scenario(
        self, create_orchestrator_harness
    ) -> None:
        conversations = [("conv" + str(i), 5) for i in range(30)]

        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=conversations,
            user_centric_rate=40.0,
            num_users=20,
            request_count=100,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) >= 100


class TestUserClass:
    def test_user_x_correlation_id_property(self) -> None:
        mock_sampled = MagicMock()
        mock_sampled.x_correlation_id = "test-corr-id"

        user = User(user_id=1, sampled=mock_sampled)

        assert user.x_correlation_id == "test-corr-id"

    def test_user_build_first_turn(self) -> None:
        mock_sampled = MagicMock()
        mock_turn = MagicMock()
        mock_sampled.build_first_turn.return_value = mock_turn

        user = User(user_id=1, sampled=mock_sampled, max_turns=5)

        result = user.build_first_turn()

        assert result == mock_turn
        mock_sampled.build_first_turn.assert_called_once_with(max_turns=5)

    @pytest.mark.parametrize("user_id", range(1, 22))
    def test_user_dataclass_creation(self, user_id: int) -> None:
        mock_sampled = MagicMock()
        mock_sampled.x_correlation_id = f"corr-{user_id}"

        user = User(
            user_id=user_id, sampled=mock_sampled, next_send_time=1000, max_turns=3
        )

        assert user.user_id == user_id
        assert user.sampled == mock_sampled
        assert user.next_send_time == 1000
        assert user.max_turns == 3


class TestSessionCountStopping:
    @pytest.mark.asyncio
    async def test_sessions_greater_than_users_completes(
        self, create_orchestrator_harness, two_turn_conversations
    ) -> None:
        num_users = 3
        num_sessions = 6

        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * 10,
            user_centric_rate=100.0,
            num_users=num_users,
            num_sessions=num_sessions,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) >= num_sessions

    @pytest.mark.parametrize(
        "num_users,num_sessions",  # fmt: skip
        [
            (2, 4),
            (2, 6),
            (3, 6),
            (3, 9),
            (4, 8),
            (5, 10),
            (5, 15),
        ],
    )
    @pytest.mark.asyncio
    async def test_various_session_user_ratios(
        self,
        create_orchestrator_harness,
        two_turn_conversations,
        num_users,
        num_sessions,
    ) -> None:
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * (num_sessions + 5),
            user_centric_rate=100.0,
            num_users=num_users,
            num_sessions=num_sessions,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) >= num_sessions
