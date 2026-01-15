# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the timing manager fixed schedule strategy."""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from aiperf.common.constants import MILLIS_PER_SECOND
from aiperf.common.enums import (
    CreditPhase,
    DatasetSamplingStrategy,
    TimingMode,
)
from aiperf.common.loop_scheduler import LoopScheduler
from aiperf.common.models import (
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.credit.structs import Credit
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.conversation_source import ConversationSource
from aiperf.timing.strategies.fixed_schedule import FixedScheduleStrategy
from tests.unit.timing.conftest import (
    OrchestratorHarness,
    create_mock_dataset_sampler,
)


@pytest.fixture
async def time_traveler(time_traveler_no_patch_sleep):
    return time_traveler_no_patch_sleep


@pytest.fixture
def mock_scheduler():
    scheduler = MagicMock(spec=LoopScheduler)
    scheduler.schedule_at_perf_ns = MagicMock()
    scheduler.schedule_later = MagicMock()
    scheduler.schedule_soon = MagicMock()
    return scheduler


@pytest.fixture
def mock_stop_checker():
    checker = MagicMock()
    checker.can_send_any_turn = MagicMock(return_value=True)
    checker.can_start_new_session = MagicMock(return_value=True)
    return checker


@pytest.fixture
def mock_credit_issuer():
    issuer = MagicMock()

    async def mock_issue_credit(*args, **kwargs):
        return True

    issuer.issue_credit = mock_issue_credit
    return issuer


@pytest.fixture
def mock_lifecycle():
    lifecycle = MagicMock()
    lifecycle.started_at_perf_ns = 1_000_000_000
    return lifecycle


def make_dataset_from_schedule(
    schedule: list[tuple[int, str]],
) -> DatasetMetadata:
    """Create DatasetMetadata from a schedule for fixed schedule testing."""
    conv_timestamps: dict[str, list[int]] = {}
    for timestamp, conv_id in schedule:
        if conv_id not in conv_timestamps:
            conv_timestamps[conv_id] = []
        conv_timestamps[conv_id].append(timestamp)

    conversations = []
    for conv_id, timestamps_list in conv_timestamps.items():
        turns = []
        for i, timestamp in enumerate(timestamps_list):
            if i == 0:
                turns.append(TurnMetadata(timestamp_ms=timestamp, delay_ms=None))
            else:
                delay = timestamp - timestamps_list[i - 1]
                turns.append(TurnMetadata(timestamp_ms=timestamp, delay_ms=delay))
        conversations.append(ConversationMetadata(conversation_id=conv_id, turns=turns))

    return DatasetMetadata(
        conversations=conversations,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


class TestFixedScheduleStrategy:
    @pytest.fixture
    def simple_schedule(self) -> list[tuple[int, str]]:
        return [
            (0, "conv1"),
            (100, "conv2"),
            (200, "conv3"),
        ]

    @pytest.fixture
    def schedule_with_offset(self) -> list[tuple[int, str]]:
        return [(1000, "conv1"), (1100, "conv2"), (1200, "conv3")]

    @pytest.mark.asyncio
    async def test_initialization_phase_configs(
        self,
        simple_schedule: list[tuple[int, str]],
        create_orchestrator_harness,
    ) -> None:
        harness: OrchestratorHarness = create_orchestrator_harness(
            schedule=simple_schedule,
        )
        await harness.orchestrator.initialize()

        assert len(harness.orchestrator._ordered_phase_configs) == 1
        assert (
            harness.orchestrator._ordered_phase_configs[0].phase
            == CreditPhase.PROFILING
        )

    @pytest.mark.asyncio
    async def test_empty_schedule_raises_error(
        self, create_orchestrator_harness
    ) -> None:
        with pytest.raises(ValidationError, match="greater than 0"):
            create_orchestrator_harness(schedule=[])

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
    @pytest.mark.asyncio
    async def test_timestamp_grouping(
        self,
        create_orchestrator_harness,
        schedule: list[tuple[int, str]],
        expected_groups: dict[int, list[str]],
        expected_keys: list[int],
    ) -> None:
        harness: OrchestratorHarness = create_orchestrator_harness(
            schedule=schedule,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) == len(schedule)

        sent_conversations = {credit.conversation_id for credit in harness.sent_credits}
        expected_conversations = {conv_id for _, conv_id in schedule}
        assert sent_conversations == expected_conversations

    @pytest.mark.parametrize(
        "auto_offset,manual_offset,expected_zero_ms",
        [
            (True, None, 1000),
            (False, 500, 500),
            (False, None, 0),
        ],
    )
    @pytest.mark.asyncio
    async def test_schedule_offset_configurations(
        self,
        mock_scheduler,
        mock_stop_checker,
        mock_credit_issuer,
        mock_lifecycle,
        schedule_with_offset: list[tuple[int, str]],
        auto_offset: bool,
        manual_offset: int | None,
        expected_zero_ms: int,
    ) -> None:
        dataset = make_dataset_from_schedule(schedule_with_offset)
        sampler = create_mock_dataset_sampler(
            conversation_ids=[conv.conversation_id for conv in dataset.conversations],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
        conversation_source = ConversationSource(dataset, sampler)

        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.FIXED_SCHEDULE,
            total_expected_requests=len(schedule_with_offset),
            auto_offset_timestamps=auto_offset,
            fixed_schedule_start_offset=manual_offset,
        )

        strategy = FixedScheduleStrategy(
            config=config,
            conversation_source=conversation_source,
            scheduler=mock_scheduler,
            stop_checker=mock_stop_checker,
            credit_issuer=mock_credit_issuer,
            lifecycle=mock_lifecycle,
        )

        await strategy.setup_phase()

        assert strategy._schedule_zero_ms == expected_zero_ms

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "schedule,expected_duration",
        [
            ([(0, "conv1"), (100, "conv2"), (200, "conv3")], 0.2),
            ([(0, "conv1"), (0, "conv2"), (0, "conv3")], 0.0),
            ([(-100, "conv1"), (-50, "conv2"), (0, "conv3")], 0.0),
        ],
    )
    async def test_execution_timing(
        self,
        create_orchestrator_harness,
        schedule: list[tuple[int, str]],
        expected_duration: float,
    ) -> None:
        harness: OrchestratorHarness = create_orchestrator_harness(
            schedule=schedule,
        )

        await harness.run_with_auto_return()

        assert len(harness.sent_credits) == len(schedule)

        sent_conversations = {credit.conversation_id for credit in harness.sent_credits}
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
        create_orchestrator_harness,
        time_traveler,
        auto_offset: bool,
        schedule: list[tuple[int, str]],
    ) -> None:
        harness: OrchestratorHarness = create_orchestrator_harness(
            schedule=schedule,
            auto_offset_timestamps=auto_offset,
        )

        first_timestamp_ms = schedule[0][0]
        last_timestamp_ms = schedule[-1][0]

        sleep_duration_ms = (
            last_timestamp_ms - first_timestamp_ms if auto_offset else last_timestamp_ms
        )
        with time_traveler.travels_for(
            sleep_duration_ms / MILLIS_PER_SECOND, tolerance=0.02
        ):
            await harness.run_with_auto_return()

        assert len(harness.sent_credits) == 3


@pytest.fixture
def fixed_schedule_strategy_factory(
    mock_scheduler, mock_stop_checker, mock_credit_issuer, mock_lifecycle
):
    async def _create_strategy(
        schedule: list[tuple[int, str]],
        auto_offset: bool = True,
        manual_offset: int | None = None,
    ):
        dataset = make_dataset_from_schedule(schedule)
        sampler = create_mock_dataset_sampler(
            conversation_ids=[conv.conversation_id for conv in dataset.conversations],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
        conversation_source = ConversationSource(dataset, sampler)

        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.FIXED_SCHEDULE,
            total_expected_requests=len(schedule),
            auto_offset_timestamps=auto_offset,
            fixed_schedule_start_offset=manual_offset,
        )

        strategy = FixedScheduleStrategy(
            config=config,
            conversation_source=conversation_source,
            scheduler=mock_scheduler,
            stop_checker=mock_stop_checker,
            credit_issuer=mock_credit_issuer,
            lifecycle=mock_lifecycle,
        )

        await strategy.setup_phase()
        return strategy

    return _create_strategy


@pytest.mark.asyncio
class TestFixedScheduleStrategySetup:
    async def test_setup_builds_sorted_schedule(
        self, fixed_schedule_strategy_factory
    ) -> None:
        schedule = [(200, "conv3"), (0, "conv1"), (100, "conv2")]
        strategy = await fixed_schedule_strategy_factory(schedule)

        timestamps = [ts for ts, _ in strategy._absolute_schedule]
        assert timestamps == sorted(timestamps)

    async def test_setup_with_auto_offset(
        self, fixed_schedule_strategy_factory
    ) -> None:
        schedule = [(1000, "conv1"), (1100, "conv2"), (1200, "conv3")]
        strategy = await fixed_schedule_strategy_factory(schedule, auto_offset=True)

        assert strategy._schedule_zero_ms == 1000

    async def test_setup_with_manual_offset(
        self, fixed_schedule_strategy_factory
    ) -> None:
        schedule = [(1000, "conv1"), (1100, "conv2"), (1200, "conv3")]
        strategy = await fixed_schedule_strategy_factory(
            schedule, auto_offset=False, manual_offset=500
        )

        assert strategy._schedule_zero_ms == 500

    async def test_setup_without_offset(self, fixed_schedule_strategy_factory) -> None:
        schedule = [(1000, "conv1"), (1100, "conv2"), (1200, "conv3")]
        strategy = await fixed_schedule_strategy_factory(
            schedule, auto_offset=False, manual_offset=None
        )

        assert strategy._schedule_zero_ms == 0

    async def test_setup_stores_lifecycle(
        self, fixed_schedule_strategy_factory
    ) -> None:
        schedule = [(0, "conv1")]
        strategy = await fixed_schedule_strategy_factory(schedule)

        assert strategy._lifecycle is not None
        assert strategy._lifecycle.started_at_perf_ns == 1_000_000_000


@pytest.mark.asyncio
class TestFixedScheduleStrategyExecutePhase:
    async def test_execute_phase_schedules_all_first_turns(
        self, fixed_schedule_strategy_factory, mock_scheduler
    ) -> None:
        schedule = [(0, "conv1"), (100, "conv2"), (200, "conv3")]
        strategy = await fixed_schedule_strategy_factory(schedule)

        await strategy.execute_phase()

        assert mock_scheduler.schedule_at_perf_sec.call_count == 3

    async def test_execute_phase_raises_without_started_at_perf_ns(
        self, mock_scheduler, mock_stop_checker, mock_credit_issuer
    ) -> None:
        schedule = [(0, "conv1")]
        dataset = make_dataset_from_schedule(schedule)
        sampler = create_mock_dataset_sampler(
            conversation_ids=["conv1"],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
        conversation_source = ConversationSource(dataset, sampler)

        mock_lifecycle = MagicMock()
        mock_lifecycle.started_at_perf_ns = None

        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.FIXED_SCHEDULE,
            total_expected_requests=1,
            auto_offset_timestamps=True,
        )

        strategy = FixedScheduleStrategy(
            config=config,
            conversation_source=conversation_source,
            scheduler=mock_scheduler,
            stop_checker=mock_stop_checker,
            credit_issuer=mock_credit_issuer,
            lifecycle=mock_lifecycle,
        )

        await strategy.setup_phase()

        with pytest.raises(RuntimeError, match="started_at_perf_ns is not set"):
            await strategy.execute_phase()


@pytest.mark.asyncio
class TestFixedScheduleStrategyCreditReturn:
    async def test_final_turn_returns_immediately(
        self, fixed_schedule_strategy_factory, mock_scheduler
    ) -> None:
        schedule = [(0, "conv1"), (100, "conv1")]
        strategy = await fixed_schedule_strategy_factory(schedule)

        credit = Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="conv1",
            x_correlation_id="corr-conv1",
            turn_index=1,
            num_turns=2,
            issued_at_ns=1000,
        )

        await strategy.handle_credit_return(credit)

        mock_scheduler.schedule_at_perf_sec.assert_not_called()
        mock_scheduler.schedule_later.assert_not_called()
        mock_scheduler.schedule_soon.assert_not_called()

    async def test_credit_return_with_timestamp_schedules_at_perf_sec(
        self, fixed_schedule_strategy_factory, mock_scheduler
    ) -> None:
        schedule = [(0, "conv1"), (100, "conv1")]
        strategy = await fixed_schedule_strategy_factory(schedule)

        credit = Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="conv1",
            x_correlation_id="corr-conv1",
            turn_index=0,
            num_turns=2,
            issued_at_ns=1000,
        )

        await strategy.handle_credit_return(credit)

        mock_scheduler.schedule_at_perf_sec.assert_called_once()


@pytest.mark.asyncio
class TestFixedScheduleTimestampConversion:
    async def test_timestamp_to_perf_sec_with_auto_offset(
        self, fixed_schedule_strategy_factory
    ) -> None:
        schedule = [(1000, "conv1"), (1100, "conv2")]
        strategy = await fixed_schedule_strategy_factory(schedule, auto_offset=True)

        expected = strategy._lifecycle.started_at_perf_sec + (100 / MILLIS_PER_SECOND)
        actual = strategy._timestamp_to_perf_sec(1100)

        assert actual == expected

    async def test_timestamp_to_perf_sec_without_offset(
        self, fixed_schedule_strategy_factory
    ) -> None:
        schedule = [(100, "conv1"), (200, "conv2")]
        strategy = await fixed_schedule_strategy_factory(
            schedule, auto_offset=False, manual_offset=None
        )

        expected = strategy._lifecycle.started_at_perf_sec + (100 / MILLIS_PER_SECOND)
        actual = strategy._timestamp_to_perf_sec(100)

        assert actual == expected


@pytest.mark.asyncio
class TestFixedScheduleStrategyEdgeCases:
    async def test_missing_first_turn_timestamp_raises_error(
        self, mock_scheduler, mock_stop_checker, mock_credit_issuer, mock_lifecycle
    ) -> None:
        dataset = DatasetMetadata(
            conversations=[
                ConversationMetadata(
                    conversation_id="conv1",
                    turns=[TurnMetadata(timestamp_ms=None)],
                )
            ],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
        sampler = create_mock_dataset_sampler(
            conversation_ids=["conv1"],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
        conversation_source = ConversationSource(dataset, sampler)

        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.FIXED_SCHEDULE,
            total_expected_requests=1,
            auto_offset_timestamps=True,
        )

        strategy = FixedScheduleStrategy(
            config=config,
            conversation_source=conversation_source,
            scheduler=mock_scheduler,
            stop_checker=mock_stop_checker,
            credit_issuer=mock_credit_issuer,
            lifecycle=mock_lifecycle,
        )

        with pytest.raises(ValueError, match="missing timestamp_ms"):
            await strategy.setup_phase()

    async def test_empty_conversations_raises_error(
        self, mock_scheduler, mock_stop_checker, mock_credit_issuer, mock_lifecycle
    ) -> None:
        dataset = DatasetMetadata(
            conversations=[
                ConversationMetadata(conversation_id="conv1", turns=[]),
            ],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
        sampler = create_mock_dataset_sampler(
            conversation_ids=["conv1"],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
        conversation_source = ConversationSource(dataset, sampler)

        config = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.FIXED_SCHEDULE,
            total_expected_requests=1,
            auto_offset_timestamps=True,
        )

        strategy = FixedScheduleStrategy(
            config=config,
            conversation_source=conversation_source,
            scheduler=mock_scheduler,
            stop_checker=mock_stop_checker,
            credit_issuer=mock_credit_issuer,
            lifecycle=mock_lifecycle,
        )

        with pytest.raises(ValueError, match="No conversations with valid"):
            await strategy.setup_phase()

    async def test_single_conversation_works(
        self, fixed_schedule_strategy_factory
    ) -> None:
        schedule = [(0, "conv1")]
        strategy = await fixed_schedule_strategy_factory(schedule)

        assert len(strategy._absolute_schedule) == 1
        assert strategy._absolute_schedule[0][0] == 0
