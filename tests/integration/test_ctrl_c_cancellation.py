# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Ctrl+C (SIGINT) benchmark cancellation functionality.

These tests verify the graceful cancellation via Ctrl+C:
- First Ctrl+C stops issuing new credits and waits for in-flight requests
- Results are written to files with was_cancelled=True
"""

import pytest

from tests.harness.utils import AIPerfMockServer
from tests.integration.conftest import AIPerfSignalCLI
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
class TestCtrlCCancellation:
    """Tests for Ctrl+C (SIGINT) benchmark cancellation functionality."""

    async def test_ctrl_c_graceful_cancel_writes_results(
        self, signal_cli: AIPerfSignalCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Ctrl+C triggers graceful cancellation, writes all output files, and sets was_cancelled=True."""
        result = await signal_cli.run_with_sigint(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --streaming \
                --benchmark-duration 20 \
                --concurrency 5 \
                --random-seed 42 \
                --osl 100 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            sigint_delay=2.0,  # Wait 2s after profiling starts
            wait_for_profiling=True,  # Wait for "AIPerf is PROFILING" log first
        )

        # All output files should be written
        assert result.json is not None, "JSON export should exist"
        assert result.csv, "CSV export should exist"
        assert result.jsonl is not None, "JSONL records should exist"
        assert result.inputs is not None, "Inputs file should exist"

        # was_cancelled flag should be True
        assert result.json.was_cancelled is True, (
            "was_cancelled flag should be True after Ctrl+C"
        )

        # Should have some completed requests (partial results)
        assert len(result.jsonl) > 0, "Should have some completed requests"

        # Each record should have metrics computed
        for record in result.jsonl:
            assert record.metrics is not None, "Record should have metrics"
