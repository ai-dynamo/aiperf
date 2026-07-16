# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.accuracy.models import AccuracyRecordsData, AccuracySummary
from aiperf.common.enums import CreditPhase
from aiperf.common.messages import (
    AccuracyRecordsMessage,
    ProcessAccuracyResultMessage,
)
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
        reasoning="ok",
    )


class TestOnAccuracyRecords:
    @pytest.mark.asyncio
    async def test_dispatches_each_record(self) -> None:
        mgr = MagicMock()
        mgr.debug = MagicMock()
        mgr._dispatch_record = AsyncMock(return_value=[])
        mgr._on_accuracy_records = RecordsManager._on_accuracy_records.__get__(mgr)

        records = [_make_accuracy_record(0), _make_accuracy_record(1)]
        await mgr._on_accuracy_records(
            AccuracyRecordsMessage(service_id="rp", records=records)
        )

        assert mgr._dispatch_record.await_count == 2
        dispatched = [c.args[0] for c in mgr._dispatch_record.await_args_list]
        assert dispatched == records

    @pytest.mark.asyncio
    async def test_dispatch_errors_are_logged_not_raised(self) -> None:
        mgr = MagicMock()
        mgr.debug = MagicMock()
        mgr._dispatch_record = AsyncMock(return_value=[ValueError("boom")])
        mgr._on_accuracy_records = RecordsManager._on_accuracy_records.__get__(mgr)

        await mgr._on_accuracy_records(
            AccuracyRecordsMessage(service_id="rp", records=[_make_accuracy_record()])
        )

        assert mgr.debug.called


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
    async def test_no_accumulator_is_noop(self) -> None:
        mgr = MagicMock()
        mgr.publish = AsyncMock()
        mgr._accuracy_accumulator = None
        mgr._publish_accuracy_results = (
            RecordsManager._publish_accuracy_results.__get__(mgr)
        )

        await mgr._publish_accuracy_results(CreditPhase.PROFILING)

        mgr.publish.assert_not_awaited()
