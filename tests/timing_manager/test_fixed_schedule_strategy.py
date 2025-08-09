# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the timing manager fixed schedule strategy.
"""

import time

import pytest

from aiperf.common.constants import MILLIS_PER_SECOND
from aiperf.common.enums import TimingMode
from aiperf.common.enums.timing_enums import CreditPhase
from aiperf.common.models.credit_models import CreditPhaseStats
from aiperf.timing import FixedScheduleStrategy, TimingManagerConfig
from tests.timing_manager.conftest import MockCreditManager
from tests.utils.time_traveler import TimeTraveler


class TestFixedScheduleStrategy:
    """Tests for the fixed schedule strategy."""

    @pytest.fixture
    def simple_schedule(self) -> list[tuple[int, str]]:
        """Simple schedule with 3 requests."""
        return [
            (0, "conv1"),
            (100, "conv2"),
            (200, "conv3"),
        ]

    @pytest.fixture
    def schedule_with_offset(self) -> list[tuple[int, str]]:
        """Schedule with auto offset."""
        return [(1000, "conv1"), (1100, "conv2"), (1200, "conv3")]

    def _create_strategy(
        self,
        mock_credit_manager: MockCreditManager,
        schedule: list[tuple[int, str]],
        auto_offset: bool = False,
        manual_offset: int | None = None,
    ) -> tuple[FixedScheduleStrategy, CreditPhaseStats]:
        """Helper to create a strategy with optional config overrides."""
        config = TimingManagerConfig.model_construct(
            timing_mode=TimingMode.FIXED_SCHEDULE,
            auto_offset_timestamps=auto_offset,
            fixed_schedule_start_offset=manual_offset,
        )
        return FixedScheduleStrategy(
            config=config,
            credit_manager=mock_credit_manager,
            schedule=schedule,
        ), CreditPhaseStats(
            type=CreditPhase.PROFILING,
            start_ns=time.time_ns(),
            total_expected_requests=len(schedule),
        )

    def test_initialization_phase_configs(
        self,
        simple_schedule: list[tuple[int, str]],
        mock_credit_manager: MockCreditManager,
    ):
        """Test initialization creates correct phase configurations."""
        strategy, _ = self._create_strategy(mock_credit_manager, simple_schedule)

        assert len(strategy.ordered_phase_configs) == 1
        assert strategy._num_requests == len(simple_schedule)
        assert strategy._schedule == simple_schedule

        # Check phase types - only profiling phase supported
        assert strategy.ordered_phase_configs[0].type == CreditPhase.PROFILING

    def test_empty_schedule_raises_error(self, mock_credit_manager: MockCreditManager):
        """Test that empty schedule raises ValueError."""
        with pytest.raises(ValueError, match="No schedule loaded"):
            self._create_strategy(mock_credit_manager, [])

    @pytest.mark.parametrize(
        "schedule,expected_groups,expected_keys",
        [
            (
                [(0, "conv1"), (100, "conv2"), (200, "conv3")],
                {0: ["conv1"], 100: ["conv2"], 200: ["conv3"]},
                [0, 100, 200],
            ),
            (
                [(0, "conv1"), (0, "conv2"), (100, "conv3"), (100, "conv4")],
                {0: ["conv1", "conv2"], 100: ["conv3", "conv4"]},
                [0, 100],
            ),
        ],
    )
    def test_timestamp_grouping(
        self,
        mock_credit_manager: MockCreditManager,
        schedule: list[tuple[int, str]],
        expected_groups: dict[int, list[str]],
        expected_keys: list[int],
    ):
        """Test that timestamps are properly grouped."""
        strategy, _ = self._create_strategy(mock_credit_manager, schedule)

        assert strategy._timestamp_groups == expected_groups
        assert strategy._sorted_timestamp_keys == expected_keys

    @pytest.mark.parametrize(
        "auto_offset,manual_offset,expected_zero_ms",
        [
            (True, None, 1000),  # Auto offset to first timestamp
            (False, 500, 500),  # Manual offset
            (False, None, 0),  # No offset
        ],
    )
    def test_schedule_offset_configurations(
        self,
        mock_credit_manager: MockCreditManager,
        schedule_with_offset: list[tuple[int, str]],
        auto_offset: bool,
        manual_offset: int | None,
        expected_zero_ms: int,
    ):
        """Test different schedule offset configurations."""
        strategy, _ = self._create_strategy(
            mock_credit_manager, schedule_with_offset, auto_offset, manual_offset
        )

        assert strategy._schedule_zero_ms == expected_zero_ms

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "schedule,expected_duration",
        [
            ([(0, "conv1"), (100, "conv2"), (200, "conv3")], 0.2),  # 200ms total
            ([(0, "conv1"), (0, "conv2"), (0, "conv3")], 0.0),  # All at once
            ([(-100, "conv1"), (-50, "conv2"), (0, "conv3")], 0.0),  # Past timestamps
        ],
    )
    async def test_execution_timing(
        self,
        mock_credit_manager: MockCreditManager,
        time_traveler: TimeTraveler,
        schedule: list[tuple[int, str]],
        expected_duration: float,
    ):
        """Test that execution timing is correct for different schedules."""
        strategy, phase_stats = self._create_strategy(mock_credit_manager, schedule)

        with time_traveler.sleeps_for(expected_duration):
            await strategy._execute_single_phase(phase_stats)
            await strategy.wait_for_tasks()

        assert phase_stats.sent == len(schedule)
        assert len(mock_credit_manager.dropped_credits) == len(schedule)

        # Verify all conversation IDs were processed
        sent_conversations = {
            credit.conversation_id for credit in mock_credit_manager.dropped_credits
        }
        assert sent_conversations == {conv_id for _, conv_id in schedule}

    @pytest.mark.parametrize("auto_offset", [True, False])
    @pytest.mark.parametrize(
        "schedule",
        [
            [(1000, "conv1"), (1100, "conv2"), (1200, "conv3")],
            [(600, "conv1"), (700, "conv2"), (800, "conv3")],
            [(0, "conv1"), (100, "conv2"), (200, "conv3")],
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_execution_with_auto_offset(
        self,
        mock_credit_manager: MockCreditManager,
        time_traveler: TimeTraveler,
        auto_offset: bool,
        schedule: list[tuple[int, str]],
    ):
        """Test execution timing with auto offset timestamps."""
        strategy, phase_stats = self._create_strategy(
            mock_credit_manager, schedule, auto_offset
        )

        first_timestamp_ms = schedule[0][0]
        last_timestamp_ms = schedule[-1][0]

        sleep_duration_ms = (
            last_timestamp_ms - first_timestamp_ms if auto_offset else last_timestamp_ms
        )
        with time_traveler.sleeps_for(sleep_duration_ms / MILLIS_PER_SECOND):
            await strategy._execute_single_phase(phase_stats)
            await strategy.wait_for_tasks()

        assert phase_stats.sent == 3
        expected_zero_ms = first_timestamp_ms if auto_offset else 0
        assert strategy._schedule_zero_ms == expected_zero_ms
