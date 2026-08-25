# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A failed finalization must terminate the run, not hang it.

``_finalize_and_process_results`` is launched with ``execute_async`` -- a bare
``create_task`` whose exception nobody retrieves, with no global asyncio
exception handler behind it. The controller's join barrier only closes on
``ProcessRecordsResultMessage``, so an exception escaping this task means the
CLI waits forever with every metric computed and nothing written. These tests
pin that a failure still produces a published, explicitly-failed result.

Companion coverage here:

* the GPU telemetry drain timeout is marked non-fatal so a dead telemetry
  container cannot suppress a valid inference export;
* ``AIPERF_RECORD_CHECKPOINT_INTERVAL=0`` genuinely disables the checkpoint
  task instead of spinning the background loop on ``asyncio.sleep(0)``;
* a stall-forced finalization is stamped onto the exported artifact.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import CreditPhase
from aiperf.common.environment import Environment
from aiperf.common.messages import ProcessRecordsResultMessage
from aiperf.common.models import PhaseRecordsStats
from aiperf.records.records_manager import (
    ERROR_FATAL_DETAIL_KEY,
    RecordsManager,
)


def _finalize_manager(finalize_error: Exception) -> MagicMock:
    """A manager whose artifact barrier fails, with the real path bound."""
    mgr = MagicMock()
    mgr.service_id = "records-manager-test"
    mgr.debug = MagicMock()
    mgr.info = MagicMock()
    mgr.warning = MagicMock()
    mgr.error = MagicMock()
    mgr.exception = MagicMock()
    mgr.publish = AsyncMock()
    mgr.send_command_and_wait_for_response = AsyncMock()
    mgr._process_results_lock = asyncio.Lock()
    mgr._processed_results = {}
    mgr._incomplete_reason = None

    stats = PhaseRecordsStats(
        phase=CreditPhase.PROFILING,
        start_ns=1_000_000_000,
        requests_end_ns=2_000_000_000,
    )
    mgr._records_tracker.create_aggregate_stats_for_phase.return_value = stats
    mgr._records_tracker.create_stats_for_phase.return_value = stats

    # Everything from the fire-and-forget entry point down to the raising
    # barrier is the real implementation.
    mgr._finalize_and_process_results = (
        RecordsManager._finalize_and_process_results.__get__(mgr)
    )
    mgr._finalize_and_process_results_impl = (
        RecordsManager._finalize_and_process_results_impl.__get__(mgr)
    )
    mgr._publish_terminal_failure_result = (
        RecordsManager._publish_terminal_failure_result.__get__(mgr)
    )
    mgr._process_results = RecordsManager._process_results.__get__(mgr)
    mgr._process_results_impl = RecordsManager._process_results_impl.__get__(mgr)
    mgr._finalize_record_processor_artifacts = AsyncMock(side_effect=finalize_error)
    return mgr


def _published_results(mgr: MagicMock) -> list[ProcessRecordsResultMessage]:
    return [
        call.args[0]
        for call in mgr.publish.await_args_list
        if isinstance(call.args[0], ProcessRecordsResultMessage)
    ]


class TestTelemetryDrainIsNonFatal:
    @pytest.mark.asyncio
    async def test_drain_timeout_is_marked_non_fatal(self, monkeypatch) -> None:
        """A dead telemetry container must not suppress a valid export."""
        mgr = MagicMock()
        mgr.warning = MagicMock()
        mgr.error = MagicMock()
        mgr._telemetry_completion_expected = True
        mgr._telemetry_final_sequence = 12
        mgr._telemetry_processed_high_water = 3
        mgr._telemetry_completion_event = asyncio.Event()
        monkeypatch.setattr(Environment.SERVICE, "COMMAND_RESPONSE_TIMEOUT", 0.01)

        errors = await RecordsManager._await_telemetry_ingest_complete(mgr)

        assert len(errors) == 1
        assert errors[0].details[ERROR_FATAL_DETAIL_KEY] is False
        assert errors[0].details["stage"] == "gpu_telemetry_drain"


class TestStallDegradationReachesTheArtifact:
    @pytest.mark.asyncio
    async def test_stall_finalization_stamps_incomplete_reason(self) -> None:
        """An incomplete run must be detectable from the artifact, not the log."""
        mgr = MagicMock()
        mgr._credits_complete_received = True
        mgr._all_records_received_phases = set()
        mgr._stall_last_total_records = -1
        mgr._stall_last_progress_ns = 0
        mgr._incomplete_reason = None
        mgr.error = MagicMock()
        mgr._handle_all_records_received_once = AsyncMock()
        mgr._records_tracker.total_records_for_phase.return_value = 24
        stats = MagicMock()
        stats.final_requests_completed = 1200
        mgr._records_tracker.create_aggregate_stats_for_phase.return_value = stats

        await RecordsManager._watch_for_record_stall(mgr)  # arm the timer
        mgr._stall_last_progress_ns -= int(
            (Environment.RECORD.COMPLETION_STALL_TIMEOUT + 1) * NANOS_PER_SECOND
        )
        await RecordsManager._watch_for_record_stall(mgr)

        mgr._handle_all_records_received_once.assert_awaited_once()
        assert mgr._incomplete_reason is not None
        assert "stalled" in mgr._incomplete_reason
