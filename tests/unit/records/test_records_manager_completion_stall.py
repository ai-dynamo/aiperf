# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.models import CreditPhaseStats
from aiperf.common.models.record_models import MetricRecordMetadata
from aiperf.records.records_manager import RecordsManager
from aiperf.records.records_tracker import RecordsTracker

DEADLINE = 120.0


def _tracker(final_requests_completed: int, records: int) -> RecordsTracker:
    """Build a real tracker for one profiling phase with a record deficit."""
    tracker = RecordsTracker()
    tracker.update_phase_info(
        CreditPhaseStats(
            phase=CreditPhase.PROFILING,
            phase_index=1,
            profiling_index=0,
            phase_name="load",
            phase_kind="profiling",
            start_ns=100,
            requests_end_ns=300,
            baseline_start_ns=90,
            baseline_end_ns=310,
            final_requests_completed=final_requests_completed,
        )
    )
    for session_num in range(records):
        tracker.update_from_request(
            MetricRecordMetadata(
                session_num=session_num,
                request_start_ns=101,
                request_end_ns=110,
                worker_id="worker",
                record_processor_id="processor",
                benchmark_phase=CreditPhase.PROFILING,
                phase_index=1,
            ),
            None,
        )
    return tracker


def _manager(tracker: RecordsTracker) -> RecordsManager:
    manager = RecordsManager.__new__(RecordsManager)
    manager._records_tracker = tracker
    manager._complete_credit_phases = {CreditPhase.PROFILING}
    manager._credits_complete_received = True
    manager._all_records_received_phases = set()
    manager._completion_stall_state = {}
    manager._completion_stall_last_log = {}
    manager._handle_all_records_received_once = AsyncMock()
    manager.error = MagicMock()
    manager.notice = MagicMock()
    manager.debug = MagicMock()
    return manager


class TestCompletionStallWatchdog:
    """The event-driven completion barrier is force-released after a stall.

    Regression tests for the scenario where a credit completes without ever
    emitting a record: no further record message arrives, so the barrier
    predicate is never re-evaluated and the phase waits forever.
    """

    @pytest.mark.asyncio
    async def test_first_tick_arms_without_forcing(self) -> None:
        manager = _manager(_tracker(final_requests_completed=3, records=2))
        await manager._check_completion_stall(now=100.0, deadline=DEADLINE)
        assert CreditPhase.PROFILING in manager._completion_stall_state
        manager._handle_all_records_received_once.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stall_past_deadline_forces_finalization(self) -> None:
        manager = _manager(_tracker(final_requests_completed=3, records=2))
        await manager._check_completion_stall(now=0.0, deadline=DEADLINE)
        await manager._check_completion_stall(now=DEADLINE, deadline=DEADLINE)
        manager._handle_all_records_received_once.assert_awaited_once_with(
            CreditPhase.PROFILING
        )
        manager.error.assert_called_once()
        assert CreditPhase.PROFILING not in manager._completion_stall_state

    @pytest.mark.asyncio
    async def test_progress_resets_the_clock(self) -> None:
        tracker = _tracker(final_requests_completed=4, records=2)
        manager = _manager(tracker)
        await manager._check_completion_stall(now=0.0, deadline=DEADLINE)
        # A third record arrives: the deficit persists but progress was made.
        tracker.update_from_request(
            MetricRecordMetadata(
                session_num=2,
                request_start_ns=101,
                request_end_ns=110,
                worker_id="worker",
                record_processor_id="processor",
                benchmark_phase=CreditPhase.PROFILING,
                phase_index=1,
            ),
            None,
        )
        await manager._check_completion_stall(now=DEADLINE, deadline=DEADLINE)
        manager._handle_all_records_received_once.assert_not_awaited()
        # Only a full deadline of no progress after the last change forces.
        await manager._check_completion_stall(now=DEADLINE * 2, deadline=DEADLINE)
        manager._handle_all_records_received_once.assert_awaited_once_with(
            CreditPhase.PROFILING
        )

    @pytest.mark.asyncio
    async def test_zero_deadline_disables_the_backstop(self) -> None:
        manager = _manager(_tracker(final_requests_completed=3, records=2))
        await manager._check_completion_stall(now=0.0, deadline=0.0)
        await manager._check_completion_stall(now=1e9, deadline=0.0)
        assert manager._completion_stall_state == {}
        manager._handle_all_records_received_once.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_deficit_never_arms(self) -> None:
        manager = _manager(_tracker(final_requests_completed=2, records=2))
        await manager._check_completion_stall(now=0.0, deadline=DEADLINE)
        assert manager._completion_stall_state == {}
        manager._handle_all_records_received_once.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_waiting_notice_is_logged_while_stalled(self) -> None:
        manager = _manager(_tracker(final_requests_completed=3, records=2))
        await manager._check_completion_stall(now=0.0, deadline=DEADLINE)
        await manager._check_completion_stall(now=31.0, deadline=DEADLINE)
        manager.notice.assert_called_once()
        manager._handle_all_records_received_once.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_already_finalized_phase_is_ignored(self) -> None:
        manager = _manager(_tracker(final_requests_completed=3, records=2))
        manager._all_records_received_phases = {CreditPhase.PROFILING}
        await manager._check_completion_stall(now=0.0, deadline=DEADLINE)
        await manager._check_completion_stall(now=1e9, deadline=DEADLINE)
        assert manager._completion_stall_state == {}
        manager._handle_all_records_received_once.assert_not_awaited()
