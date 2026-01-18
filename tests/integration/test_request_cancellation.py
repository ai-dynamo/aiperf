# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for request cancellation functionality."""

import pytest

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
            timeout=120.0,
        )
        for request in result.jsonl:
            if request.metadata.was_cancelled:
                assert request.error is not None
                assert request.error.code == 499
                assert request.error.type == "RequestCancellationError"
                # Make sure that error_isl is still computed for errors
                assert request.metrics.get("error_isl") is not None
                assert request.metrics.get("error_isl").value > 0

        assert result.json.was_cancelled is False  # This is not a cancellation error
        assert result.json.error_summary is not None
        assert len(result.json.error_summary) > 0
        assert result.json.error_summary[0].count > 0
        assert result.json.error_summary[0].error_details.code == 499
        assert (
            result.json.error_summary[0].error_details.type
            == "RequestCancellationError"
        )
