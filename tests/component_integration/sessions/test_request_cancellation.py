# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for single-turn request cancellation functionality.

This module tests basic request cancellation behavior without multi-turn sessions.
For multi-turn cancellation tests, see test_request_cancellation_multi_turn.py.
"""

import pytest

from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.component_integration.conftest import (
    count_cancelled_requests,
    validate_cancellation_errors,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestRequestCancellation:
    """Tests for single-turn request cancellation functionality."""

    def test_single_turn_cancellation_pipeline_integrity(self, cli: AIPerfCLI):
        """Cancelled requests are properly recorded with correct error codes.

        Verifies that:
        1. Cancelled requests have error code 499 and type RequestCancellationError
        2. The benchmark completes successfully (was_cancelled=False for the run)
        3. Error summary correctly aggregates cancellation errors
        """
        result = cli.run_sync(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 50 \
                --concurrency 5 \
                --random-seed 42 \
                --osl 10 \
                --request-cancellation-rate 30 \
                --request-cancellation-delay 0 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=120.0,
        )

        # Validate cancelled requests have proper error details
        validate_cancellation_errors(result.jsonl)

        # With 30% cancellation rate on 50 requests, expect meaningful cancellations
        cancelled_count = count_cancelled_requests(result.jsonl)
        assert cancelled_count > 5, (
            f"Expected >5 cancellations with 30% rate, got {cancelled_count}"
        )

        # Benchmark itself should complete successfully (not be cancelled)
        assert result.json.was_cancelled is False

        # Error summary should capture the cancellation errors
        assert result.json.error_summary is not None
        cancellation_summary = next(
            (e for e in result.json.error_summary if e.error_details.code == 499),
            None,
        )
        assert cancellation_summary is not None, (
            "No cancellation error in error_summary"
        )
        assert cancellation_summary.count > 0
        assert cancellation_summary.error_details.type == "RequestCancellationError"
