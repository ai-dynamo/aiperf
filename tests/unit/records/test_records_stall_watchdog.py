# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The records-completion barrier must fail loudly rather than hang forever.

``check_and_set_all_records_received`` requires
``success + error >= final_requests_completed`` and is driven purely by record
arrivals. A request that completes without ever emitting a record leaves the
barrier permanently short, and nothing re-triggers it -- observed live as a run
frozen at 24 of 1200 records with no timeout. These tests pin the watchdog that
bounds that.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import CreditPhase
from aiperf.common.environment import Environment
from aiperf.common.models import MetricRecordMetadata
from aiperf.records.records_manager import RecordsManager
from aiperf.records.records_tracker import RecordsTracker


def _manager(
    *,
    credits_complete: bool,
    total_records: int,
    already_handled: bool = False,
) -> MagicMock:
    """Build a stand-in exposing only what the watchdog touches."""
    mgr = MagicMock(spec=RecordsManager)
    mgr._credits_complete_received = credits_complete
    mgr._all_records_received_phases = (
        {CreditPhase.PROFILING} if already_handled else set()
    )
    mgr._stall_last_total_records = -1
    mgr._stall_last_progress_ns = 0
    mgr._records_tracker = MagicMock()
    mgr._records_tracker.total_records_for_phase.return_value = total_records
    stats = MagicMock()
    stats.final_requests_completed = 1200
    mgr._records_tracker.create_aggregate_stats_for_phase.return_value = stats
    mgr.error = MagicMock()
    mgr._handle_all_records_received_once = AsyncMock()
    return mgr


async def _tick(mgr: MagicMock) -> None:
    await RecordsManager._watch_for_record_stall(mgr)


class TestRecordStallWatchdog:
    @pytest.mark.asyncio
    async def test_does_not_fire_before_credits_complete(self) -> None:
        """Records legitimately lag credits; only a post-credits stall counts."""
        mgr = _manager(credits_complete=False, total_records=24)
        await _tick(mgr)
        mgr._handle_all_records_received_once.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_does_not_fire_when_already_finalized(self) -> None:
        mgr = _manager(credits_complete=True, total_records=24, already_handled=True)
        await _tick(mgr)
        mgr._handle_all_records_received_once.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_progress_resets_the_stall_timer(self) -> None:
        """Slow-but-advancing aggregation must never be cut short."""
        mgr = _manager(credits_complete=True, total_records=24)
        await _tick(mgr)
        first_mark = mgr._stall_last_progress_ns
        assert first_mark > 0

        # A record lands: the timer restarts even though it is long overdue.
        mgr._stall_last_progress_ns = 1
        mgr._records_tracker.total_records_for_phase.return_value = 25
        await _tick(mgr)

        assert mgr._stall_last_total_records == 25
        assert mgr._stall_last_progress_ns > 1
        mgr._handle_all_records_received_once.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_indexed_named_phase_progress_arms_stall_timer(self) -> None:
        """The watchdog must see records stored under a concrete phase index."""
        mgr = _manager(credits_complete=True, total_records=0)
        tracker = RecordsTracker()
        tracker.update_from_request(
            MetricRecordMetadata(
                session_num=1,
                request_start_ns=1,
                request_end_ns=2,
                worker_id="worker",
                record_processor_id="processor",
                benchmark_phase=CreditPhase.PROFILING,
                phase_index=7,
            ),
            None,
        )
        mgr._records_tracker = tracker

        await _tick(mgr)

        assert mgr._stall_last_total_records == 1
        assert mgr._stall_last_progress_ns > 0
        assert (CreditPhase.PROFILING, None) not in tracker._phase_trackers

    @pytest.mark.asyncio
    async def test_finalizes_after_the_stall_timeout(self) -> None:
        """No progress for the whole budget -> finalize, loudly."""
        mgr = _manager(credits_complete=True, total_records=24)
        await _tick(mgr)  # establish the baseline mark

        # Backdate the mark past the budget with the count unchanged.
        overdue = int(
            (Environment.RECORD.COMPLETION_STALL_TIMEOUT + 1) * NANOS_PER_SECOND
        )
        mgr._stall_last_progress_ns -= overdue
        await _tick(mgr)

        mgr._handle_all_records_received_once.assert_awaited_once_with(
            CreditPhase.PROFILING
        )
        assert mgr.error.called, "a stalled run must say so"

    @pytest.mark.asyncio
    async def test_timeout_zero_disables_the_watchdog(self, monkeypatch) -> None:
        mgr = _manager(credits_complete=True, total_records=24)
        monkeypatch.setattr(Environment.RECORD, "COMPLETION_STALL_TIMEOUT", 0.0)
        await _tick(mgr)
        mgr._stall_last_progress_ns -= 10**15
        await _tick(mgr)
        mgr._handle_all_records_received_once.assert_not_awaited()
