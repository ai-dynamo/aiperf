# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component-integration regression tests for the record/credit lockstep"""

from __future__ import annotations

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.messages import RecordsMessage
from aiperf.credit.messages import CreditReturn
from aiperf.records.inference_result_parser import InferenceResultParser
from aiperf.workers.inference_client import InferenceClient
from tests.harness import (
    FakeCommunication,  # noqa: F401
    FakeServiceManager,  # noqa: F401
    FakeTransport,  # noqa: F401
)
from tests.harness.utils import AIPerfCLI

pytestmark = pytest.mark.component_integration

_PROFILE_CMD = """
    aiperf profile \
        --model gpt2 \
        --endpoint-type chat \
        --request-count 12 \
        --concurrency 4 \
        --osl 2 \
        --isl 2 \
        --extra-inputs ignore_eos:true \
        --workers-max 2 \
        --random-seed 42 \
        --ui simple \
        --streaming
"""


def _assert_lockstep_with_injected_errors(result) -> None:
    """Assert the lockstep invariant held across the full run."""
    rr = result.runner_result
    credit_returns = rr.messages(CreditReturn, sent=True)
    metric_records = rr.messages(RecordsMessage, sent=True)
    non_cancelled = [c for c in credit_returns if not c.cancelled]

    assert len(metric_records) > 0, "no records reached the RecordsManager"
    assert len(metric_records) == len(non_cancelled), (
        f"lockstep broken: {len(non_cancelled)} non-cancelled credits returned "
        f"but {len(metric_records)} records were forwarded"
    )
    error_records = [m for m in metric_records if m.error is not None]
    assert len(error_records) >= 1, (
        "expected the injected fault to surface as a forwarded error record"
    )


@pytest.mark.component_integration
class TestRecordCreditLockstepIntegration:
    """The full pipeline must not hang when records fail to parse or requests"""

    def test_record_parse_failure_does_not_hang_run(self, cli: AIPerfCLI, monkeypatch):
        """RecordProcessor parse failures are forwarded as error records, so the"""
        original = InferenceResultParser.parse_request_record

        async def flaky_parse(self, request_record):
            info = request_record.request_info
            if (
                info is not None
                and info.credit_phase == CreditPhase.PROFILING
                and info.credit_num % 3 == 0
            ):
                raise ValueError("injected parse failure (over-context simulation)")
            return await original(self, request_record)

        monkeypatch.setattr(InferenceResultParser, "parse_request_record", flaky_parse)

        result = cli.run_sync(_PROFILE_CMD, assert_success=False)
        _assert_lockstep_with_injected_errors(result)

    def test_worker_send_failure_does_not_hang_run(self, cli: AIPerfCLI, monkeypatch):
        """Worker failures before the request is sent are forwarded as error"""
        original = InferenceClient.send_request

        async def flaky_send(self, request_info, first_token_callback=None):
            if (
                request_info.credit_phase == CreditPhase.PROFILING
                and request_info.credit_num % 3 == 0
            ):
                raise ValueError("injected send failure (pre-request)")
            return await original(
                self, request_info, first_token_callback=first_token_callback
            )

        monkeypatch.setattr(InferenceClient, "send_request", flaky_send)

        result = cli.run_sync(_PROFILE_CMD, assert_success=False)
        _assert_lockstep_with_injected_errors(result)
