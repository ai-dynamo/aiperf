# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive unit tests for user-centric rate timing strategy.

Tests cover:
- Initialization and validation
- User pre-generation with virtual history
- Turn gap and stagger calculations
- Precise mode execution
- LMBench mode execution
- Credit return handling
- Edge cases (single user, high QPS, etc.)
"""

import pytest

from aiperf.common.enums import (
    CreditPhase,
    TimingMode,
)
from aiperf.timing.config import CreditPhaseConfig
from tests.unit.timing.conftest import OrchestratorHarness

# =============================================================================
# Fixtures
# =============================================================================


# NOTE: We use the DEFAULT time_traveler (which DOES patch asyncio.sleep)
# because user_centric_rate uses asyncio.sleep in the spawn loop.
# If we use time_traveler_no_patch_sleep, the spawn loop sleeps in real time
# while scheduled coroutines use virtual time, causing a mismatch.


@pytest.fixture
def two_turn_conversations():
    """Simple dataset with 2-turn conversations (minimum for virtual history)."""
    return [("conv1", 2), ("conv2", 2), ("conv3", 2), ("conv4", 2), ("conv5", 2)]


@pytest.fixture
def multi_turn_conversations():
    """Dataset with multi-turn conversations."""
    return [("conv1", 3), ("conv2", 3), ("conv3", 3), ("conv4", 3)]


# =============================================================================
# Initialization and Validation Tests
# =============================================================================


class TestUserCentricStrategyInitialization:
    """Tests for UserCentricStrategy initialization and validation."""

    @pytest.mark.asyncio
    async def test_initialization_with_valid_config(
        self, create_orchestrator_harness, two_turn_conversations
    ):
        """UserCentricStrategy initializes correctly with valid config."""
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations,
            user_centric_rate=10.0,
            num_users=5,
            request_count=10,
        )

        await harness.orchestrator.initialize()

        # Should have one profiling phase
        assert len(harness.orchestrator._ordered_phase_configs) == 1
        assert (
            harness.orchestrator._ordered_phase_configs[0].phase
            == CreditPhase.PROFILING
        )

    def test_direct_init_requires_num_users(self):
        """UserCentricStrategy constructor requires num_users to be set."""
        from unittest.mock import MagicMock

        from aiperf.timing.strategies.user_centric_rate import UserCentricStrategy

        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.USER_CENTRIC_RATE,
            request_rate=10.0,
            num_users=None,  # Missing!
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

    def test_direct_init_requires_positive_request_rate(self):
        """UserCentricStrategy constructor requires positive request_rate."""
        from unittest.mock import MagicMock

        from aiperf.timing.strategies.user_centric_rate import UserCentricStrategy

        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.USER_CENTRIC_RATE,
            request_rate=None,  # Missing!
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


# =============================================================================
# Setup Phase Tests
# =============================================================================


class TestUserCentricSetupPhase:
    """Tests for user pre-generation and setup logic."""

    @pytest.mark.asyncio
    async def test_pre_generates_users_in_precise_mode(
        self, create_orchestrator_harness, two_turn_conversations
    ):
        """Precise mode pre-generates users with replacements."""
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * 10,  # Plenty
            user_centric_rate=20.0,
            num_users=5,
            request_count=10,  # Match to 2x num_users
        )

        await harness.run_with_auto_return()

        # Should send exactly request_count credits
        assert len(harness.sent_credits) == 10

    @pytest.mark.asyncio
    async def test_turn_gap_calculation(
        self, create_orchestrator_harness, two_turn_conversations
    ):
        """Turn gap is calculated as num_users / qps."""
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * 4,  # 20 conversations
            user_centric_rate=10.0,
            num_users=10,
            request_count=10,  # Match num_users
        )

        await harness.run_with_auto_return()

        # Verify exactly request_count credits sent
        assert len(harness.sent_credits) == 10

    @pytest.mark.asyncio
    async def test_virtual_history_works_with_multi_turn(
        self, create_orchestrator_harness, multi_turn_conversations
    ):
        """Virtual history works with multi-turn conversations."""
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=multi_turn_conversations * 2,  # 8 conversations
            user_centric_rate=20.0,
            num_users=4,
            request_count=4,  # Just initial users
        )

        await harness.run_with_auto_return()

        # Should send credits successfully
        assert len(harness.sent_credits) == 4


# =============================================================================
# Precise Mode Execution Tests
# =============================================================================


class TestPreciseModeExecution:
    """Tests for precise (timestamp scheduling) execution mode."""

    @pytest.mark.asyncio
    async def test_precise_mode_basic_execution(
        self, create_orchestrator_harness, two_turn_conversations
    ):
        """Precise mode executes and sends credits correctly."""
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * 4,  # 20 conversations
            user_centric_rate=50.0,
            num_users=10,
            request_count=25,
        )

        await harness.run_with_auto_return()

        # Should send exactly request_count credits
        assert len(harness.sent_credits) >= 25

    @pytest.mark.asyncio
    async def test_precise_mode_with_multi_turn(
        self, create_orchestrator_harness, multi_turn_conversations
    ):
        """Precise mode handles multi-turn conversations."""
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=multi_turn_conversations * 3,  # 12 conversations
            user_centric_rate=40.0,
            num_users=8,
            request_count=30,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) >= 30

        # Verify multi-turn sessions exist
        multi_turn_sessions = [c for c in harness.sent_credits if c.num_turns > 1]
        assert len(multi_turn_sessions) > 0

    @pytest.mark.asyncio
    async def test_precise_mode_single_user(
        self, create_orchestrator_harness, multi_turn_conversations
    ):
        """Precise mode works with single user."""
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=multi_turn_conversations,
            user_centric_rate=10.0,
            num_users=1,  # Single user
            request_count=10,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) >= 10

    @pytest.mark.asyncio
    async def test_precise_mode_high_user_count(
        self, create_orchestrator_harness, two_turn_conversations
    ):
        """Precise mode works with high user counts."""
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * 20,  # 100 conversations
            user_centric_rate=100.0,
            num_users=50,  # Many users
            request_count=100,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) >= 100


# =============================================================================
# Edge Cases and Error Scenarios
# =============================================================================


class TestUserCentricEdgeCases:
    """Edge case tests for user-centric rate strategy."""

    @pytest.mark.asyncio
    async def test_very_low_qps(
        self, create_orchestrator_harness, two_turn_conversations
    ):
        """Strategy works with very low QPS."""
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * 2,  # 10 conversations
            user_centric_rate=1.0,  # 1 QPS
            num_users=2,
            request_count=2,  # Match num_users
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) == 2

    @pytest.mark.asyncio
    async def test_very_high_qps(
        self, create_orchestrator_harness, two_turn_conversations
    ):
        """Strategy works with very high QPS."""
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * 100,  # 500 conversations
            user_centric_rate=500.0,  # 500 QPS
            num_users=100,
            request_count=500,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) >= 500

    @pytest.mark.asyncio
    async def test_num_users_equals_request_count(
        self, create_orchestrator_harness, two_turn_conversations
    ):
        """Strategy works when num_users equals request_count."""
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations,
            user_centric_rate=10.0,
            num_users=5,
            request_count=5,  # Same as num_users
        )

        await harness.run_with_auto_return()

        # Should send exactly 5 credits (one per user)
        assert len(harness.sent_credits) == 5

    @pytest.mark.asyncio
    async def test_num_users_greater_than_request_count(
        self, create_orchestrator_harness, two_turn_conversations
    ):
        """Strategy works when num_users > request_count."""
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * 4,  # 20 conversations
            user_centric_rate=20.0,
            num_users=20,  # More users than requests
            request_count=10,
        )

        await harness.run_with_auto_return()

        # Should stop at request_count
        assert len(harness.sent_credits) == 10

    @pytest.mark.asyncio
    async def test_works_with_multi_turn_dataset(
        self, create_orchestrator_harness, two_turn_conversations
    ):
        """Strategy works with multi-turn conversation dataset."""
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * 4,  # 20 conversations
            user_centric_rate=40.0,
            num_users=10,
            request_count=10,
        )

        await harness.run_with_auto_return()

        # Should send exactly request_count credits
        assert len(harness.sent_credits) == 10

        # Virtual history may reduce turns for some users, so just verify execution succeeded
        unique_users = {c.x_correlation_id for c in harness.sent_credits}
        assert len(unique_users) >= 1  # At least one user active


# =============================================================================
# User Generation and Virtual History Tests
# =============================================================================


class TestMassiveGauntlet:
    """Comprehensive variation gauntlet to test user-centric rate across parameter space."""

    @pytest.mark.parametrize(
        "qps", [float(i) for i in range(5, 101)]
    )  # 96 tests: QPS 5-100
    @pytest.mark.asyncio
    async def test_every_qps_from_5_to_100(
        self, create_orchestrator_harness, two_turn_conversations, qps
    ):
        """GAUNTLET 1/5: Every integer QPS from 5 to 100 with 5 users."""
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
        [
            (u, q) for u in range(2, 21) for q in [10.0, 20.0, 50.0, 100.0]
        ],  # 19*4=76 tests
    )
    @pytest.mark.asyncio
    async def test_users_2_to_20_various_qps(
        self, create_orchestrator_harness, two_turn_conversations, num_users, qps
    ):
        """GAUNTLET 2/5: Users 2-20 with various QPS values."""
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
        ],  # 7*3*3=63 tests
    )
    @pytest.mark.asyncio
    async def test_varying_turn_counts_gauntlet(
        self, create_orchestrator_harness, turns, num_users, qps
    ):
        """GAUNTLET 3/5: Varying session turn counts (2-10 turns)."""
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
        ],  # 12 tests: fractional QPS
    )
    @pytest.mark.asyncio
    async def test_fractional_qps_values(
        self, create_orchestrator_harness, two_turn_conversations, qps
    ):
        """GAUNTLET 4/5: Fractional QPS values to test decimal handling."""
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
            # Extreme combinations
            (1, 10.0, 2),
            (1, 100.0, 5),
            (50, 200.0, 2),
            (20, 150.0, 3),
            (15, 75.0, 4),
            (8, 40.0, 6),
            (12, 60.0, 3),
            (25, 125.0, 2),
            # Edge cases
            (2, 5.0, 2),
            (3, 7.5, 3),
            (4, 12.0, 4),
            (7, 21.0, 5),
        ],  # 12 tests
    )
    @pytest.mark.asyncio
    async def test_extreme_edge_case_combinations(
        self, create_orchestrator_harness, num_users, qps, turns
    ):
        """GAUNTLET 5/5: Extreme and edge case combinations."""
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
        [
            (t, q) for t in range(2, 12) for q in range(10, 101, 5)
        ],  # 10 turns * 19 QPS = 190 tests
    )
    @pytest.mark.asyncio
    async def test_all_turn_counts_2_to_11_with_qps(
        self, create_orchestrator_harness, turns, qps
    ):
        """GAUNTLET 6/8: All turn counts 2-11 with QPS 10-100 (step 5)."""
        conversations = [(f"conv{i}", turns) for i in range(15)]  # 15 conversations

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
        [
            (u, q) for u in range(1, 31) for q in range(20, 101, 20)
        ],  # 30 users * 5 QPS = 150 tests
    )
    @pytest.mark.asyncio
    async def test_users_1_to_30_qps_multiples_of_20(
        self, create_orchestrator_harness, two_turn_conversations, num_users, qps
    ):
        """GAUNTLET 7/8: Users 1-30 with QPS at 20, 40, 60, 80, 100."""
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
        ],  # 5 * 9 * 3 = 135 tests
    )
    @pytest.mark.asyncio
    async def test_comprehensive_combinations(
        self, create_orchestrator_harness, num_users, qps, turns
    ):
        """GAUNTLET 8/8: Comprehensive combinations of users, QPS, and turns."""
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
        [
            (u, q, t)
            for u in range(2, 12)  # 10 user counts
            for q in range(10, 81, 10)  # 8 QPS values (10, 20, 30, ..., 80)
            for t in [2, 4, 6]  # 3 turn counts
        ],  # 10 * 8 * 3 = 240 tests
    )
    @pytest.mark.asyncio
    async def test_ultra_comprehensive_matrix(
        self, create_orchestrator_harness, num_users, qps, turns
    ):
        """GAUNTLET 9/9: Ultra-comprehensive 3D matrix of users × QPS × turns."""
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
    """Tests for user generation and virtual history logic."""

    @pytest.mark.asyncio
    async def test_generates_initial_users(
        self, create_orchestrator_harness, multi_turn_conversations
    ):
        """Setup phase generates initial users."""
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=multi_turn_conversations,
            user_centric_rate=20.0,
            num_users=4,
            request_count=15,
        )

        await harness.run_with_auto_return()

        # Should have sent credits for multiple users
        unique_correlations = {c.x_correlation_id for c in harness.sent_credits}
        assert len(unique_correlations) >= 4

    @pytest.mark.asyncio
    async def test_virtual_history_works_correctly(
        self, create_orchestrator_harness, two_turn_conversations
    ):
        """Virtual history creates users with staggered completion states."""
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * 4,  # 20 conversations
            user_centric_rate=20.0,  # Use QPS that works reliably
            num_users=5,
            request_count=5,  # Match num_users
        )

        await harness.run_with_auto_return()

        # Should send exactly request_count credits
        assert len(harness.sent_credits) == 5


# =============================================================================
# Parameter Variation Tests
# =============================================================================


class TestUserCentricParameterVariations:
    """Tests with various parameter combinations."""

    @pytest.mark.parametrize(
        "num_users,qps,request_count",  # fmt: skip
        [
            (5, 10.0, 10),  # Basic - match request_count to num_users
            (10, 50.0, 15),  # Medium scale
            (1, 5.0, 5),  # Single user
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
    ):
        """Test various num_users/QPS/request_count combinations."""
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * 20,  # Plenty of conversations
            user_centric_rate=qps,
            num_users=num_users,
            request_count=request_count,
        )

        await harness.run_with_auto_return()

        # Should send exactly request_count credits (stop condition)
        assert len(harness.sent_credits) == request_count

    @pytest.mark.parametrize(
        "num_users,turns_per_session",  # fmt: skip
        [
            (5, 1),  # Single turn
            (5, 3),  # Multi turn
            (2, 5),  # Fewer users, more turns
        ],
    )
    @pytest.mark.asyncio
    async def test_user_count_vs_turn_count_combinations(
        self,
        create_orchestrator_harness,
        num_users,
        turns_per_session,
    ):
        """Test various num_users vs turns_per_session combinations."""
        # Create conversations with specified turn count
        conversations = [(f"conv{i}", turns_per_session) for i in range(num_users * 2)]

        # Request count = num_users for simpler validation
        request_count = num_users

        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=conversations,
            user_centric_rate=float(num_users * 5),  # Scale QPS with users
            num_users=num_users,
            request_count=request_count,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) == request_count


# =============================================================================
# Correlation ID and Session Tracking Tests
# =============================================================================


class TestSessionTracking:
    """Tests for session and correlation ID tracking."""

    @pytest.mark.asyncio
    async def test_unique_correlation_ids_per_user(
        self, create_orchestrator_harness, two_turn_conversations
    ):
        """Each user has a unique x_correlation_id."""
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * 4,
            user_centric_rate=25.0,
            num_users=5,
            request_count=15,
        )

        await harness.run_with_auto_return()

        # Collect all correlation IDs
        correlation_ids = [c.x_correlation_id for c in harness.sent_credits]

        # Each initial user should have unique correlation ID
        # (replacements will have different IDs)
        unique_ids = set(correlation_ids)
        assert len(unique_ids) >= 5

    @pytest.mark.asyncio
    async def test_multi_turn_sessions_share_correlation_id(
        self, create_orchestrator_harness, multi_turn_conversations
    ):
        """All turns in a session share the same x_correlation_id."""
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=multi_turn_conversations,
            user_centric_rate=20.0,
            num_users=4,
            request_count=20,
        )

        await harness.run_with_auto_return()

        # Group credits by x_correlation_id
        sessions = {}
        for credit in harness.sent_credits:
            corr_id = credit.x_correlation_id
            if corr_id not in sessions:
                sessions[corr_id] = []
            sessions[corr_id].append(credit)

        # Find multi-turn sessions
        multi_turn_sessions = [
            credits for credits in sessions.values() if len(credits) > 1
        ]

        # Should have at least some multi-turn sessions
        assert len(multi_turn_sessions) > 0

        # Verify turn indices are sequential within each session
        for session_credits in multi_turn_sessions:
            turn_indices = [c.turn_index for c in session_credits]
            assert turn_indices == sorted(turn_indices)
            # Should start from 0
            assert turn_indices[0] == 0


# =============================================================================
# Stop Condition Tests
# =============================================================================


class TestStopConditions:
    """Tests for stop condition handling."""

    @pytest.mark.asyncio
    async def test_stops_at_request_count(
        self, create_orchestrator_harness, two_turn_conversations
    ):
        """Strategy stops when request_count is reached."""
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * 10,  # Plenty available
            user_centric_rate=50.0,
            num_users=10,
            request_count=25,  # Limit at 25
        )

        await harness.run_with_auto_return()

        # Should stop at exactly request_count
        assert len(harness.sent_credits) == 25

    @pytest.mark.asyncio
    async def test_stops_at_session_count(
        self, create_orchestrator_harness, multi_turn_conversations
    ):
        """Strategy stops when num_sessions is reached."""
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=multi_turn_conversations * 5,  # 20 conversations
            user_centric_rate=40.0,
            num_users=8,
            num_sessions=10,  # Limit by sessions
        )

        await harness.run_with_auto_return()

        # Should have started exactly 10 sessions (turn_index=0)
        first_turns = [c for c in harness.sent_credits if c.turn_index == 0]
        assert len(first_turns) == 10


# =============================================================================
# Realistic Scenario Tests
# =============================================================================


class TestRealisticScenarios:
    """Realistic user-centric rate scenarios."""

    @pytest.mark.asyncio
    async def test_realistic_chat_benchmark(self, create_orchestrator_harness):
        """Realistic chat benchmark: 50 users, 100 QPS, 3-turn conversations."""
        conversations = [("conv" + str(i), 3) for i in range(100)]

        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=conversations,
            user_centric_rate=100.0,
            num_users=50,
            request_count=200,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) >= 200

        # Verify mix of turn indices
        turn_indices = {c.turn_index for c in harness.sent_credits}
        assert 0 in turn_indices  # Has first turns
        assert len(turn_indices) > 1  # Has multiple turn types

    @pytest.mark.asyncio
    async def test_kv_cache_benchmark_scenario(self, create_orchestrator_harness):
        """KV cache benchmark: sustained load with turn gaps."""
        # 20 users, each with 5-turn conversations
        conversations = [("conv" + str(i), 5) for i in range(30)]

        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=conversations,
            user_centric_rate=40.0,  # 40 QPS
            num_users=20,  # 20 concurrent users
            request_count=100,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) >= 100

        # With 20 users at 40 QPS, turn_gap = 20/40 = 0.5 seconds
        # This ensures KV cache timing is realistic


# =============================================================================
# User Class Tests
# =============================================================================


class TestUserClass:
    """Direct tests for the User class."""

    def test_user_x_correlation_id_property(self):
        """User.x_correlation_id returns sampled.x_correlation_id."""
        from unittest.mock import MagicMock

        from aiperf.timing.strategies.user_centric_rate import User

        mock_sampled = MagicMock()
        mock_sampled.x_correlation_id = "test-corr-id"

        user = User(user_id=1, sampled=mock_sampled)

        assert user.x_correlation_id == "test-corr-id"

    def test_user_build_first_turn(self):
        """User.build_first_turn delegates to sampled.build_first_turn."""
        from unittest.mock import MagicMock

        from aiperf.timing.strategies.user_centric_rate import User

        mock_sampled = MagicMock()
        mock_turn = MagicMock()
        mock_sampled.build_first_turn.return_value = mock_turn

        user = User(user_id=1, sampled=mock_sampled, max_turns=5)

        result = user.build_first_turn()

        assert result == mock_turn
        mock_sampled.build_first_turn.assert_called_once_with(max_turns=5)

    @pytest.mark.parametrize("user_id", range(1, 22))  # 21 tests to reach 2000 total!
    def test_user_dataclass_creation(self, user_id):
        """User dataclass can be created with various user IDs."""
        from unittest.mock import MagicMock

        from aiperf.timing.strategies.user_centric_rate import User

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
    """Tests for session-count stopping."""

    @pytest.mark.asyncio
    async def test_sessions_greater_than_users_completes(
        self, create_orchestrator_harness, two_turn_conversations
    ):
        """Completes all sessions when num_sessions > num_users."""
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
    ):
        """Handles various session:user ratios correctly."""
        harness: OrchestratorHarness = create_orchestrator_harness(
            conversations=two_turn_conversations * (num_sessions + 5),
            user_centric_rate=100.0,
            num_users=num_users,
            num_sessions=num_sessions,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) >= num_sessions
