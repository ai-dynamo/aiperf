# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the timing manager fixed schedule strategy.
"""

import time

import pytest

from aiperf.common.constants import MILLIS_PER_SECOND
from aiperf.common.enums import CreditPhase, TimingMode
from aiperf.common.models import CreditPhaseStats
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
        speedup: float | None = None,
    ) -> tuple[FixedScheduleStrategy, CreditPhaseStats]:
        """Helper to create a strategy with optional config overrides."""
        config = TimingManagerConfig.model_construct(
            timing_mode=TimingMode.FIXED_SCHEDULE,
            auto_offset_timestamps=auto_offset,
            fixed_schedule_start_offset=manual_offset,
            fixed_schedule_speedup=speedup,
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

    @pytest.mark.parametrize(
        "speedup,expected_time_scale",
        [
            (None, 1.0),   # Default behavior (no speedup)
            (1.0, 1.0),    # No speedup
            (2.0, 0.5),    # 2x faster
            (0.5, 2.0),    # 2x slower
            (10.0, 0.1),   # 10x faster
            (0.1, 10.0),   # 10x slower
        ],
    )  # fmt: skip
    def test_speedup_time_scale_calculation(
        self,
        simple_schedule: list[tuple[int, str]],
        mock_credit_manager: MockCreditManager,
        speedup: float | None,
        expected_time_scale: float,
    ):
        """Test that speedup parameter correctly calculates time scale."""
        strategy, _ = self._create_strategy(
            mock_credit_manager, simple_schedule, speedup=speedup
        )

        assert strategy._time_scale == expected_time_scale

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "speedup,schedule",
        [
            # 2x faster - should take half the time
            (2.0, [(0, "conv1"), (100, "conv2"), (200, "conv3")]),
            # 4x faster - should take quarter the time
            (4.0, [(0, "conv1"), (100, "conv2"), (200, "conv3")]),
            # 2x slower - should take double the time
            (0.5, [(0, "conv1"), (100, "conv2"), (200, "conv3")]),
            # Different schedule with larger gaps
            (2.0, [(0, "conv1"), (500, "conv2"), (1000, "conv3")]),
            # Edge case: all at same timestamp should still be instant
            (2.0, [(0, "conv1"), (0, "conv2"), (0, "conv3")]),
            (0.5, [(0, "conv1"), (0, "conv2"), (0, "conv3")]),
        ],
    )  # fmt: skip
    async def test_speedup_execution_timing(
        self,
        mock_credit_manager: MockCreditManager,
        time_traveler: TimeTraveler,
        speedup: float,
        schedule: list[tuple[int, str]],
    ):
        """Test that speedup parameter affects actual execution timing."""
        strategy, phase_stats = self._create_strategy(
            mock_credit_manager, schedule, speedup=speedup
        )

        # Calculate expected duration: (last_timestamp - first_timestamp) / 1000 / speedup
        # Since auto_offset is default True, we use the relative duration
        first_timestamp_ms = schedule[0][0]
        last_timestamp_ms = schedule[-1][0]
        base_duration_sec = (last_timestamp_ms - first_timestamp_ms) / MILLIS_PER_SECOND
        expected_duration = base_duration_sec / speedup

        with time_traveler.sleeps_for(expected_duration):
            await strategy._execute_single_phase(phase_stats)
            await strategy.wait_for_tasks()

        assert phase_stats.sent == len(schedule)
        assert len(mock_credit_manager.dropped_credits) == len(schedule)

    @pytest.mark.asyncio
    async def test_speedup_with_auto_offset(
        self,
        mock_credit_manager: MockCreditManager,
        time_traveler: TimeTraveler,
    ):
        """Test speedup works correctly with auto offset timestamps."""
        # Schedule starts at 1000ms with 200ms total duration
        schedule = [(1000, "conv1"), (1100, "conv2"), (1200, "conv3")]
        speedup = 2.0  # 2x faster

        strategy, phase_stats = self._create_strategy(
            mock_credit_manager, schedule, auto_offset=True, speedup=speedup
        )

        # With auto offset, the duration should be (1200-1000)ms = 200ms
        # At 2x speed: 200ms / 2 = 100ms = 0.1s
        base_duration_sec = (1200 - 1000) / MILLIS_PER_SECOND  # 0.2s
        expected_duration = base_duration_sec / speedup  # 0.1s

        with time_traveler.sleeps_for(expected_duration):
            await strategy._execute_single_phase(phase_stats)
            await strategy.wait_for_tasks()

        assert phase_stats.sent == 3
        assert strategy._schedule_zero_ms == 1000  # First timestamp

    @pytest.mark.asyncio
    async def test_speedup_with_manual_offset(
        self,
        mock_credit_manager: MockCreditManager,
        time_traveler: TimeTraveler,
    ):
        """Test speedup works correctly with manual offset."""
        schedule = [(1000, "conv1"), (1100, "conv2"), (1200, "conv3")]
        manual_offset = 500
        speedup = 0.5  # 2x slower

        strategy, phase_stats = self._create_strategy(
            mock_credit_manager, schedule, manual_offset=manual_offset, speedup=speedup
        )

        # With manual offset of 500, effective duration is (1200-500) = 700ms
        # At 0.5x speed: 700ms / 0.5 = 1400ms = 1.4s
        base_duration_sec = (1200 - manual_offset) / MILLIS_PER_SECOND  # 0.7s
        expected_duration = base_duration_sec / speedup  # 1.4s

        with time_traveler.sleeps_for(expected_duration):
            await strategy._execute_single_phase(phase_stats)
            await strategy.wait_for_tasks()

        assert phase_stats.sent == 3
        assert strategy._schedule_zero_ms == manual_offset

    @pytest.mark.asyncio
    async def test_speedup_with_negative_timestamps(
        self,
        mock_credit_manager: MockCreditManager,
        time_traveler: TimeTraveler,
    ):
        """Test speedup behavior with negative timestamps (past events)."""
        # All timestamps are in the past, should execute immediately
        schedule = [(-100, "conv1"), (-50, "conv2"), (0, "conv3")]
        speedup = 2.0  # Even with speedup, past events should be immediate

        strategy, phase_stats = self._create_strategy(
            mock_credit_manager, schedule, speedup=speedup
        )

        # Should still take no time since all timestamps are <= 0
        with time_traveler.sleeps_for(0.0):
            await strategy._execute_single_phase(phase_stats)
            await strategy.wait_for_tasks()

        assert phase_stats.sent == 3

    def test_speedup_edge_cases(
        self,
        simple_schedule: list[tuple[int, str]],
        mock_credit_manager: MockCreditManager,
    ):
        """Test edge cases for speedup parameter."""
        # Test very small speedup (very slow execution)
        strategy_slow, _ = self._create_strategy(
            mock_credit_manager, simple_schedule, speedup=0.001
        )
        assert strategy_slow._time_scale == 1000.0

        # Test very large speedup (very fast execution)
        strategy_fast, _ = self._create_strategy(
            mock_credit_manager, simple_schedule, speedup=1000.0
        )
        assert strategy_fast._time_scale == 0.001
