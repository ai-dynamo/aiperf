# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for FinalResultsMixin."""

import pytest

from aiperf.common.enums import MessageType
from aiperf.common.messages import ProcessRecordsResultMessage
from aiperf.common.mixins.final_results_mixin import FinalResultsMixin
from aiperf.common.models.record_models import (
    MetricResult,
    ProcessRecordsResult,
    ProfileResults,
)


def _make_results() -> ProcessRecordsResult:
    """Create a minimal ProcessRecordsResult for testing."""
    metric = MetricResult(
        tag="request_latency",
        header="Request Latency",
        unit="ms",
        avg=42.0,
        count=10,
    )
    profile = ProfileResults(
        records=[metric],
        completed=10,
        start_ns=1_000_000_000,
        end_ns=2_000_000_000,
    )
    return ProcessRecordsResult(results=profile)


class TestFinalResultsMixinInit:
    """Test initialization defaults."""

    def test_initial_state_has_no_results(self) -> None:
        mixin = FinalResultsMixin.__new__(FinalResultsMixin)
        mixin._final_results = None
        mixin._benchmark_complete = False

        assert mixin._final_results is None
        assert mixin._benchmark_complete is False


class TestFinalResultsMixinOnMessage:
    """Test the message handler stores results."""

    @pytest.mark.asyncio
    async def test_stores_results_on_message(self) -> None:
        mixin = FinalResultsMixin.__new__(FinalResultsMixin)
        mixin._final_results = None
        mixin._benchmark_complete = False

        results = _make_results()
        message = ProcessRecordsResultMessage(
            service_id="records-manager-1",
            results=results,
        )

        await mixin._on_process_records_result(message)

        assert mixin._final_results is results
        assert mixin._benchmark_complete is True

    @pytest.mark.asyncio
    async def test_overwrites_previous_results(self) -> None:
        mixin = FinalResultsMixin.__new__(FinalResultsMixin)
        mixin._final_results = None
        mixin._benchmark_complete = False

        results1 = _make_results()
        results2 = _make_results()

        msg1 = ProcessRecordsResultMessage(service_id="rm-1", results=results1)
        msg2 = ProcessRecordsResultMessage(service_id="rm-1", results=results2)

        await mixin._on_process_records_result(msg1)
        await mixin._on_process_records_result(msg2)

        assert mixin._final_results is results2

    @pytest.mark.asyncio
    async def test_handler_is_decorated_with_on_message(self) -> None:
        from aiperf.common.hooks import AIPerfHook

        handler = FinalResultsMixin._on_process_records_result
        assert handler.__aiperf_hook_type__ == AIPerfHook.ON_MESSAGE
        assert MessageType.PROCESS_RECORDS_RESULT in handler.__aiperf_hook_params__
