# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for request cancellation functionality."""

import pytest

from aiperf.common.constants import IS_WINDOWS
from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
class TestRequestCancellation:
    """Tests for request cancellation functionality."""

    async def test_request_cancellation(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Request cancellation doesn't break pipeline."""
        # Heavier per-stream throughput is required on Windows for the same
        # workload (~3.5M tokens streamed across 35 surviving requests of 100k
        # OSL each). The Linux 120s budget is too tight on Windows VDI even
        # after the SO_SNDBUF Auto-Tuning fix — async I/O via the Proactor
        # event loop runs ~2x slower under sustained streaming concurrency.
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-count 50 \
                --concurrency 5 \
                --random-seed 42 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --osl 100_000 \
                --request-cancellation-rate 30 \
                --request-cancellation-delay 0 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=300.0 if IS_WINDOWS else 120.0,
        )
        for request in result.jsonl:
            if request.metadata.was_cancelled:
                assert request.error is not None
                assert request.error.code == 499
                assert request.error.type == "RequestCancellationError"
                # A delay-0 cancellation can happen before the request is
                # acknowledged, so no per-request latency/token metric is
                # required. The aggregate cancellation counter is authoritative.

        assert result.json.was_cancelled is False  # This is not a cancellation error
        assert result.json.cancelled_request_count is not None
        assert result.json.cancelled_request_count["avg"] > 0
        assert result.json.error_summary is not None
        assert len(result.json.error_summary) > 0
        assert result.json.error_summary[0].count > 0
        assert result.json.error_summary[0].error_details.code == 499
        assert (
            result.json.error_summary[0].error_details.type
            == "RequestCancellationError"
        )
