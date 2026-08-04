# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.accuracy.models import AccuracyRecordsData, AccuracySummary
from aiperf.common.enums import CreditPhase
from aiperf.common.messages import (
    ProcessAccuracyResultMessage,
    RecordsMessage,
)
from aiperf.common.models.record_models import MetricRecordMetadata
from aiperf.records.records_manager import RecordsManager


def _make_accuracy_record(session_num: int = 0) -> AccuracyRecordsData:
    return AccuracyRecordsData(
        session_num=session_num,
        worker_id="w1",
        benchmark_phase=CreditPhase.PROFILING,
        timestamp_ns=1_000,
        task=None,
        grader_name="multiple_choice",
        passed=True,
        unparsed=False,
        confidence=1.0,
        expected="A",
        actual="A",
        explanation="ok",
    )


def _metadata() -> MetricRecordMetadata:
    return MetricRecordMetadata(
        session_num=0,
        request_start_ns=1_000,
        request_end_ns=2_000,
        worker_id="w1",
        record_processor_id="rp",
        benchmark_phase=CreditPhase.PROFILING,
    )


def _records_manager_for_dispatch(dispatch_result: list) -> MagicMock:
    mgr = MagicMock()
    mgr.debug = MagicMock()
    mgr.trace = MagicMock()
    mgr.is_trace_enabled = False
    mgr._dataset_configured_event = asyncio.Event()
    mgr._dataset_configured_event.set()
    mgr._records_tracker = MagicMock()
    mgr._records_tracker.check_and_set_all_records_received_for_phase.return_value = (
        False
    )
    mgr._error_tracker = MagicMock()
    mgr._complete_credit_phases = set()
    mgr._dispatch_record = AsyncMock(return_value=dispatch_result)
    mgr._warned_missing_cache_reporting = False
    mgr._failed_request_threshold = None
    mgr._maybe_trigger_failed_request_abort = AsyncMock()
    mgr._on_records = RecordsManager._on_records.__get__(mgr)
    return mgr


class TestOnAccuracyRecords:
    """Accuracy records ride the generic RecordsMessage envelope and are
    dispatched by ``_on_records`` without any accuracy-specific handler."""

    @pytest.mark.asyncio
    async def test_dispatches_each_record(self) -> None:
        mgr = _records_manager_for_dispatch([])

        records = [_make_accuracy_record(0), _make_accuracy_record(1)]
        await mgr._on_records(
            RecordsMessage(service_id="rp", metadata=_metadata(), records=records)
        )

        assert mgr._dispatch_record.await_count == 2
        dispatched = [c.args[0] for c in mgr._dispatch_record.await_args_list]
        assert dispatched == records

    @pytest.mark.asyncio
    async def test_dispatch_errors_are_tracked_not_raised(self) -> None:
        mgr = _records_manager_for_dispatch([ValueError("boom")])

        await mgr._on_records(
            RecordsMessage(
                service_id="rp", metadata=_metadata(), records=[_make_accuracy_record()]
            )
        )

        assert mgr._error_tracker.increment_error_count_for_phase.called


class TestPublishAccuracyResults:
    @pytest.mark.asyncio
    async def test_publishes_summary_from_accumulator(self) -> None:
        summary = AccuracySummary(
            total_evaluated=3,
            total_passed=2,
            accuracy_rate=2 / 3,
            overall_unparsed=1,
            grader_name="multiple_choice",
        )
        accumulator = MagicMock()
        accumulator.export_results = AsyncMock(return_value=summary)

        mgr = MagicMock()
        mgr.service_id = "rm"
        mgr.publish = AsyncMock()
        mgr._accuracy_accumulator = accumulator
        mgr._publish_accuracy_results = (
            RecordsManager._publish_accuracy_results.__get__(mgr)
        )

        await mgr._publish_accuracy_results(CreditPhase.PROFILING)

        accumulator.export_results.assert_awaited_once()
        ctx = accumulator.export_results.await_args.args[0]
        assert ctx.phase == CreditPhase.PROFILING

        mgr.publish.assert_awaited_once()
        msg = mgr.publish.await_args.args[0]
        assert isinstance(msg, ProcessAccuracyResultMessage)
        assert msg.accuracy_result.results == summary

    @pytest.mark.asyncio
    async def test_publishes_none_summary(self) -> None:
        accumulator = MagicMock()
        accumulator.export_results = AsyncMock(return_value=None)

        mgr = MagicMock()
        mgr.service_id = "rm"
        mgr.publish = AsyncMock()
        mgr._accuracy_accumulator = accumulator
        mgr._publish_accuracy_results = (
            RecordsManager._publish_accuracy_results.__get__(mgr)
        )

        await mgr._publish_accuracy_results(CreditPhase.PROFILING)

        mgr.publish.assert_awaited_once()
        msg = mgr.publish.await_args.args[0]
        assert msg.accuracy_result.results is None

    @pytest.mark.asyncio
    async def test_no_accumulator_publishes_terminal_none(self) -> None:
        """No accumulator must still publish exactly one terminal ``results=None``.

        The SystemController clears ``_should_wait_for_accuracy`` only on this
        message, so a bare early-return would hang shutdown forever when the
        accumulator failed to construct while accuracy is config-enabled.
        """
        mgr = MagicMock()
        mgr.service_id = "rm"
        mgr.publish = AsyncMock()
        mgr._accuracy_accumulator = None
        mgr._publish_accuracy_results = (
            RecordsManager._publish_accuracy_results.__get__(mgr)
        )

        await mgr._publish_accuracy_results(CreditPhase.PROFILING)

        mgr.publish.assert_awaited_once()
        msg = mgr.publish.await_args.args[0]
        assert isinstance(msg, ProcessAccuracyResultMessage)
        assert msg.accuracy_result.results is None

    @pytest.mark.asyncio
    async def test_export_raises_still_publishes_terminal_none(self) -> None:
        """An ``export_results`` failure logs and still publishes ``results=None``.

        Guarantees the exactly-once terminal message so shutdown never hangs on a
        summary-computation error.
        """
        accumulator = MagicMock()
        accumulator.export_results = AsyncMock(side_effect=RuntimeError("boom"))

        mgr = MagicMock()
        mgr.service_id = "rm"
        mgr.publish = AsyncMock()
        mgr.exception = MagicMock()
        mgr._accuracy_accumulator = accumulator
        mgr._publish_accuracy_results = (
            RecordsManager._publish_accuracy_results.__get__(mgr)
        )

        await mgr._publish_accuracy_results(CreditPhase.PROFILING)

        assert mgr.exception.called
        mgr.publish.assert_awaited_once()
        msg = mgr.publish.await_args.args[0]
        assert isinstance(msg, ProcessAccuracyResultMessage)
        assert msg.accuracy_result.results is None
