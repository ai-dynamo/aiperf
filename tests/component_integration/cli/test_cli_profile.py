# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration test that runs the aiperf profile command with mocked services."""

import pytest

from aiperf.credit.messages import CreditReturn
from aiperf.credit.structs import Credit
from tests.component_integration.conftest import AIPerfRunnerResultWithSharedBus
from tests.harness import (
    FakeCommunication,  # noqa: F401
    FakeServiceManager,  # noqa: F401
    FakeTransport,  # noqa: F401
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestCLIProfile:
    """Tests for CLI profile command."""

    def test_profile_command_completes_with_credit_accounting(self, cli: AIPerfCLI):
        """Verify profile command completes and credits are properly accounted for.

        This test validates the end-to-end flow:
        1. CLI profile command runs to completion
        2. All issued credits are eventually returned (no credit leaks)
        3. Request cancellation works correctly with the credit system
        """
        concurrency = 13
        workers_max = 3
        request_count = 100
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model gpt2 \
                --endpoint-type chat \
                --request-count {request_count} \
                --concurrency {concurrency} \
                --osl 2 \
                --isl 2 \
                --extra-inputs ignore_eos:true \
                --request-cancellation-rate 50 \
                --workers-max {workers_max} \
                --random-seed 42 \
                --ui simple \
                --streaming
            """
        )

        # Verify benchmark completed with expected request count
        # With seed 42 and 50% cancellation rate, 47 requests complete
        assert result.request_count == 47

        # Verify credit accounting: all credits issued must be returned
        runner_result: AIPerfRunnerResultWithSharedBus = result.runner_result
        assert runner_result.shared_bus is not None

        credits = list(runner_result.payloads_by_type(Credit, sent=True))
        credit_returns = list(runner_result.payloads_by_type(CreditReturn, sent=True))

        # Every credit issued must have a corresponding return (no leaks)
        assert len(credits) == len(credit_returns), (
            f"Credit leak detected: {len(credits)} credits issued but "
            f"{len(credit_returns)} returned"
        )
