# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for different output export formats."""

import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.conftest import IntegrationTestDefaults as defaults
from tests.integration.models import AIPerfMockServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestOutputFormats:
    """Tests for different output export formats."""

    async def test_csv_export(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """CSV export format validation."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model Qwen/Qwen2.5-Coder-32B-Instruct \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert "Metric" in result.csv
        assert "Request Latency" in result.csv

    async def test_json_export(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """JSON export format validation."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model microsoft/Phi-4-reasoning \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """
        )
        assert result.json is not None
        assert result.json.request_count is not None
        assert result.json.request_latency is not None
