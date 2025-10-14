# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for different output export formats."""

import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.models import FakeAIServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestOutputFormats:
    """Tests for different output export formats."""

    async def test_csv_export(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """CSV export format validation."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model Qwen/Qwen2.5-Coder-32B-Instruct \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-count 10 \
                --concurrency 2 \
                --workers-max 1 \
                --ui simple
            """
        )
        assert "Metric" in result.csv
        assert "Request Latency" in result.csv

    async def test_json_export(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """JSON export format validation."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model microsoft/Phi-4-reasoning \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --workers-max 1 \
                --ui simple
            """
        )
        assert result.json is not None
        assert result.json.request_count is not None
        assert result.json.request_latency is not None
