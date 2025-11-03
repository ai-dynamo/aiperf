# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for different output export formats."""

import platform

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

    @pytest.mark.skipif(
        platform.system() == "Darwin",
        reason="This test is flaky on macOS in Github Actions.",
    )
    async def test_raw_export_level(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test that raw records are properly created using --export-level raw."""
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
                --export-level raw \
                --ui {defaults.ui}
            """
        )

        # Verify raw records file exists
        assert result.raw_records is not None
        assert len(result.raw_records) > 0
        assert len(result.raw_records) == defaults.request_count

        # Validate raw record structure and content
        for record in result.raw_records:
            # Verify metadata exists and has required fields
            assert record.metadata is not None
            assert isinstance(record.metadata.turn_index, int)
            assert isinstance(record.metadata.request_start_ns, int)
            assert record.metadata.worker_id is not None
            assert record.metadata.record_processor_id is not None
            assert record.metadata.benchmark_phase is not None

            # Verify raw record fields exist
            assert isinstance(record.start_perf_ns, int)
            assert record.payload is not None
            assert isinstance(record.payload, dict)

            # Verify payload has expected structure for chat endpoint
            assert "messages" in record.payload
            assert isinstance(record.payload["messages"], list)
            assert len(record.payload["messages"]) > 0

            # Verify status code exists and is valid
            assert record.status is not None
            assert isinstance(record.status, int)
            assert 200 <= record.status < 300

            # Verify responses exist (should have at least one for streaming)
            assert record.responses is not None
            assert isinstance(record.responses, list)

            # Verify error is None for successful requests
            assert record.error is None

            # Verify request headers exist
            assert record.request_headers is not None
            assert isinstance(record.request_headers, dict)

        # Validate all Pydantic models are properly loaded
        result.validate_pydantic_models()

        # Verify standard exports still exist
        assert result.json is not None
        assert result.csv
        assert result.jsonl is not None
