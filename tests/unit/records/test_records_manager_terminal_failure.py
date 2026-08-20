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
from aiperf.common.mixins.task_manager_mixin import TaskManagerMixin
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


class TestFinalizationFailureTerminatesTheRun:
    @pytest.mark.asyncio
    async def test_barrier_failure_still_publishes_a_result(self) -> None:
        """The join barrier must close even when finalization blows up."""
        mgr = _finalize_manager(RuntimeError("artifact barrier failed"))

        # Must not escape: an escaping exception is exactly the hang.
        await mgr._finalize_and_process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        published = _published_results(mgr)
        assert len(published) == 1, "the run must terminate, not hang"

    @pytest.mark.asyncio
    async def test_published_failure_is_not_mistakable_for_success(self) -> None:
        """Fail-closed: no records, marked incomplete, error marked fatal."""
        mgr = _finalize_manager(RuntimeError("artifact barrier failed"))

        await mgr._finalize_and_process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        result = _published_results(mgr)[0].results
        assert result.results.records is None
        assert result.results.completed == 0
        assert result.results.is_complete is False
        assert "artifact barrier failed" in result.results.incomplete_reason
        assert len(result.errors) == 1
        assert result.errors[0].details[ERROR_FATAL_DETAIL_KEY] is True
        assert "artifact barrier failed" in result.errors[0].message

    @pytest.mark.asyncio
    async def test_failure_is_logged_with_context(self) -> None:
        mgr = _finalize_manager(RuntimeError("artifact barrier failed"))

        await mgr._finalize_and_process_results(
            phase=CreditPhase.PROFILING, cancelled=False
        )

        assert mgr.exception.called
        logged = str(mgr.exception.call_args.args[0])
        assert "artifact barrier failed" in logged

    @pytest.mark.asyncio
    async def test_a_real_result_is_never_overwritten(self) -> None:
        """A late failure must not clobber results that already went out."""
        mgr = _finalize_manager(RuntimeError("artifact barrier failed"))
        sentinel = MagicMock()
        mgr._processed_results[CreditPhase.PROFILING] = sentinel

        returned = await mgr._publish_terminal_failure_result(
            CreditPhase.PROFILING, False, RuntimeError("late")
        )

        assert returned is sentinel
        assert _published_results(mgr) == []


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


class TestCheckpointIntervalZeroDisablesTheTask:
    @pytest.mark.asyncio
    async def test_zero_interval_ends_the_background_loop(self, monkeypatch) -> None:
        """``interval=0`` means disabled, not "spin on asyncio.sleep(0)"."""
        monkeypatch.setattr(Environment.RECORD, "CHECKPOINT_INTERVAL", 0.0)
        mgr = MagicMock()
        mgr.debug = MagicMock()
        mgr.exception = MagicMock()
        mgr._is_kubernetes_run = MagicMock(return_value=False)

        calls = 0

        async def body() -> None:
            nonlocal calls
            calls += 1
            await RecordsManager._write_partial_checkpoint_task(mgr)

        task = asyncio.create_task(
            TaskManagerMixin._background_task_loop(
                mgr,
                body,
                interval=lambda self: Environment.RECORD.CHECKPOINT_INTERVAL,
                immediate=False,
            )
        )
        try:
            for _ in range(50):
                await asyncio.sleep(0)
            assert task.done(), f"loop still spinning after {calls} iterations"
            assert calls == 1
        finally:
            task.cancel()


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
