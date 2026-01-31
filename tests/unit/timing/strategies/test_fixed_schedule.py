# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from aiperf.common.constants import MILLIS_PER_SECOND
from aiperf.common.enums import CreditPhase
from aiperf.common.models import ConversationMetadata, DatasetMetadata, TurnMetadata
from aiperf.credit.structs import Credit
from aiperf.plugin.enums import DatasetSamplingStrategy, TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.conversation_source import ConversationSource
from aiperf.timing.strategies.fixed_schedule import FixedScheduleStrategy
from tests.unit.timing.conftest import OrchestratorHarness, make_sampler


@pytest.fixture
async def time_traveler(time_traveler_no_patch_sleep):
    return time_traveler_no_patch_sleep


def make_dataset(schedule: list[tuple[int, str]]) -> DatasetMetadata:
    conv_ts: dict[str, list[int]] = {}
    for ts, cid in schedule:
        conv_ts.setdefault(cid, []).append(ts)
    convs = []
    for cid, ts_list in conv_ts.items():
        turns = [TurnMetadata(timestamp_ms=ts_list[0], delay_ms=None)]
        turns.extend(
            TurnMetadata(timestamp_ms=ts, delay_ms=ts - ts_list[i])
            for i, ts in enumerate(ts_list[1:])
        )
        convs.append(ConversationMetadata(conversation_id=cid, turns=turns))
    return DatasetMetadata(
        conversations=convs, sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL
    )


def make_strategy(
    schedule: list[tuple[int, str]],
    auto_offset: bool = True,
    manual_offset: int | None = None,
) -> tuple[FixedScheduleStrategy, MagicMock, MagicMock]:
    scheduler = MagicMock()
    scheduler.schedule_at_perf_ns = MagicMock()
    scheduler.schedule_later = MagicMock()
    scheduler.execute_async = MagicMock()
    stop_checker = MagicMock()
    stop_checker.can_send_any_turn = MagicMock(return_value=True)
    stop_checker.can_start_new_session = MagicMock(return_value=True)
    issuer = MagicMock()
    issuer.issue_credit = lambda *a, **k: True
    lifecycle = MagicMock()
    lifecycle.started_at_perf_ns = 1_000_000_000
    ds = make_dataset(schedule)
    sampler = make_sampler(
        [c.conversation_id for c in ds.conversations],
        DatasetSamplingStrategy.SEQUENTIAL,
    )
    src = ConversationSource(ds, sampler)
    cfg = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.FIXED_SCHEDULE,
        total_expected_requests=len(schedule),
        auto_offset_timestamps=auto_offset,
        fixed_schedule_start_offset=manual_offset,
    )
    strategy = FixedScheduleStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=stop_checker,
        credit_issuer=issuer,
        lifecycle=lifecycle,
    )
    return strategy, scheduler, lifecycle


@pytest.mark.asyncio
class TestFixedScheduleStrategy:
    async def test_empty_schedule_raises(self, create_orchestrator_harness) -> None:
        with pytest.raises(ValidationError, match="greater than 0"):
            create_orchestrator_harness(schedule=[])

    @pytest.mark.parametrize(
        "schedule",
        [
            [(0, "c1"), (100, "c2"), (200, "c3")],
            [(0, "c1"), (0, "c2"), (100, "c3"), (100, "c4")],
            [(0, "c1"), (0, "c2"), (0, "c3")],
            [(-100, "c1"), (-50, "c2"), (0, "c3")],
        ],
    )  # fmt: skip
    async def test_sends_credits_for_all_conversations(
        self, create_orchestrator_harness, schedule
    ) -> None:
        """Verify all conversations in schedule receive credits."""
        expected_convs = {cid for _, cid in schedule}
        h: OrchestratorHarness = create_orchestrator_harness(schedule=schedule)
        await h.run_with_auto_return()
        assert len(h.sent_credits) == len(schedule)
        assert {c.conversation_id for c in h.sent_credits} == expected_convs

    @pytest.mark.parametrize(
        "auto_offset,manual_offset,expected_zero",
        [(True, None, 1000), (False, 500, 500), (False, None, 0)],
    )  # fmt: skip
    async def test_offset_configs(
        self, auto_offset, manual_offset, expected_zero
    ) -> None:
        """Verify schedule_zero_ms is computed correctly from offset settings."""
        schedule = [(1000, "c1"), (1100, "c2"), (1200, "c3")]
        strategy, _, _ = make_strategy(schedule, auto_offset, manual_offset)
        await strategy.setup_phase()
        assert strategy._schedule_zero_ms == expected_zero

    @pytest.mark.parametrize("auto_offset", [True, False])
    @pytest.mark.parametrize(
        "schedule",
        [
            [(1000, "c1"), (1100, "c2"), (1200, "c3")],
            [(600, "c1"), (700, "c2"), (800, "c3")],
            [(0, "c1"), (100, "c2"), (200, "c3")],
        ],
    )  # fmt: skip
    async def test_auto_offset_affects_execution_duration(
        self, create_orchestrator_harness, time_traveler, auto_offset, schedule
    ) -> None:
        """Verify auto_offset=True reduces execution time by skipping initial gap."""
        h: OrchestratorHarness = create_orchestrator_harness(
            schedule=schedule, auto_offset_timestamps=auto_offset
        )
        first_ts, last_ts = schedule[0][0], schedule[-1][0]
        sleep_ms = last_ts - first_ts if auto_offset else last_ts
        # Use wider tolerance (50ms) for CI variance
        with time_traveler.travels_for(sleep_ms / MILLIS_PER_SECOND, tolerance=0.05):
            await h.run_with_auto_return()
        assert len(h.sent_credits) == 3


@pytest.mark.asyncio
class TestFixedScheduleSetup:
    async def test_builds_sorted_schedule(self) -> None:
        """Verify schedule entries are sorted by timestamp regardless of input order."""
        strategy, _, _ = make_strategy([(200, "c3"), (0, "c1"), (100, "c2")])
        await strategy.setup_phase()
        timestamps = [ts for ts, _ in strategy._absolute_schedule]
        assert timestamps == sorted(timestamps)


@pytest.mark.asyncio
class TestFixedScheduleExecutePhase:
    async def test_schedules_all_first_turns(self) -> None:
        """Verify execute_phase schedules one call per first turn."""
        strategy, scheduler, _ = make_strategy([(0, "c1"), (100, "c2"), (200, "c3")])
        await strategy.setup_phase()
        await strategy.execute_phase()
        assert scheduler.schedule_at_perf_sec.call_count == 3

    async def test_raises_without_started_at_perf_ns(self) -> None:
        """Verify execute_phase raises RuntimeError if lifecycle not started."""
        scheduler = MagicMock()
        stop_checker = MagicMock()
        stop_checker.can_send_any_turn = MagicMock(return_value=True)
        stop_checker.can_start_new_session = MagicMock(return_value=True)
        issuer = MagicMock()
        issuer.issue_credit = lambda *a, **k: True
        lifecycle = MagicMock()
        lifecycle.started_at_perf_ns = None
        ds = make_dataset([(0, "c1")])
        sampler = make_sampler(["c1"], DatasetSamplingStrategy.SEQUENTIAL)
        src = ConversationSource(ds, sampler)
        cfg = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.FIXED_SCHEDULE,
            total_expected_requests=1,
            auto_offset_timestamps=True,
        )
        strategy = FixedScheduleStrategy(
            config=cfg,
            conversation_source=src,
            scheduler=scheduler,
            stop_checker=stop_checker,
            credit_issuer=issuer,
            lifecycle=lifecycle,
        )
        await strategy.setup_phase()
        with pytest.raises(RuntimeError, match="started_at_perf_ns is not set"):
            await strategy.execute_phase()


@pytest.mark.asyncio
class TestFixedScheduleCreditReturn:
    async def test_final_turn_does_not_schedule(self) -> None:
        """Verify final turn of conversation does not schedule any follow-up."""
        strategy, scheduler, _ = make_strategy([(0, "c1"), (100, "c1")])
        await strategy.setup_phase()
        credit = Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="c1",
            x_correlation_id="corr-c1",
            turn_index=1,
            num_turns=2,
            issued_at_ns=1000,
        )
        await strategy.handle_credit_return(credit)
        scheduler.schedule_at_perf_sec.assert_not_called()
        scheduler.schedule_later.assert_not_called()
        scheduler.execute_async.assert_not_called()

    async def test_non_final_turn_schedules_next(self) -> None:
        """Verify non-final turn schedules the next turn in the conversation."""
        strategy, scheduler, _ = make_strategy([(0, "c1"), (100, "c1")])
        await strategy.setup_phase()
        credit = Credit(
            id=1,
            phase=CreditPhase.PROFILING,
            conversation_id="c1",
            x_correlation_id="corr-c1",
            turn_index=0,
            num_turns=2,
            issued_at_ns=1000,
        )
        await strategy.handle_credit_return(credit)
        scheduler.schedule_at_perf_sec.assert_called_once()


@pytest.mark.asyncio
class TestFixedScheduleTimestampConversion:
    @pytest.mark.parametrize(
        "auto_offset,manual_offset",
        [(True, None), (False, None)],
    )  # fmt: skip
    async def test_timestamp_to_perf_sec(self, auto_offset, manual_offset) -> None:
        """Verify timestamp conversion accounts for schedule_zero_ms offset."""
        schedule = (
            [(100, "c1"), (200, "c2")]
            if not auto_offset
            else [(1000, "c1"), (1100, "c2")]
        )
        strategy, _, _ = make_strategy(schedule, auto_offset, manual_offset)
        await strategy.setup_phase()
        expected = strategy._lifecycle.started_at_perf_sec + (100 / MILLIS_PER_SECOND)
        actual = strategy._timestamp_to_perf_sec(1100 if auto_offset else 100)
        assert actual == expected


@pytest.mark.asyncio
class TestFixedScheduleEdgeCases:
    async def test_missing_first_turn_timestamp_raises(self) -> None:
        """Verify setup_phase raises ValueError when first turn lacks timestamp."""
        scheduler, stop_checker, issuer, lifecycle = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        stop_checker.can_send_any_turn = MagicMock(return_value=True)
        stop_checker.can_start_new_session = MagicMock(return_value=True)
        issuer.issue_credit = lambda *a, **k: True
        lifecycle.started_at_perf_ns = 1_000_000_000
        ds = DatasetMetadata(
            conversations=[
                ConversationMetadata(
                    conversation_id="c1", turns=[TurnMetadata(timestamp_ms=None)]
                )
            ],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
        sampler = make_sampler(["c1"], DatasetSamplingStrategy.SEQUENTIAL)
        src = ConversationSource(ds, sampler)
        cfg = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.FIXED_SCHEDULE,
            total_expected_requests=1,
            auto_offset_timestamps=True,
        )
        strategy = FixedScheduleStrategy(
            config=cfg,
            conversation_source=src,
            scheduler=scheduler,
            stop_checker=stop_checker,
            credit_issuer=issuer,
            lifecycle=lifecycle,
        )
        with pytest.raises(ValueError, match="missing timestamp_ms"):
            await strategy.setup_phase()

    async def test_empty_conversations_raises(self) -> None:
        """Verify setup_phase raises ValueError when no valid timestamps found."""
        scheduler, stop_checker, issuer, lifecycle = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        stop_checker.can_send_any_turn = MagicMock(return_value=True)
        stop_checker.can_start_new_session = MagicMock(return_value=True)
        issuer.issue_credit = lambda *a, **k: True
        lifecycle.started_at_perf_ns = 1_000_000_000
        ds = DatasetMetadata(
            conversations=[ConversationMetadata(conversation_id="c1", turns=[])],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )
        sampler = make_sampler(["c1"], DatasetSamplingStrategy.SEQUENTIAL)
        src = ConversationSource(ds, sampler)
        cfg = CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.FIXED_SCHEDULE,
            total_expected_requests=1,
            auto_offset_timestamps=True,
        )
        strategy = FixedScheduleStrategy(
            config=cfg,
            conversation_source=src,
            scheduler=scheduler,
            stop_checker=stop_checker,
            credit_issuer=issuer,
            lifecycle=lifecycle,
        )
        with pytest.raises(ValueError, match="No conversations with valid"):
            await strategy.setup_phase()

    async def test_single_conversation_works(self) -> None:
        """Verify strategy handles single-conversation schedule correctly."""
        strategy, _, _ = make_strategy([(0, "c1")])
        await strategy.setup_phase()
        assert len(strategy._absolute_schedule) == 1
        assert strategy._absolute_schedule[0][0] == 0
