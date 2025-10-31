# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for high concurrency and performance scenarios."""

import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.conftest import IntegrationTestDefaults as defaults
from tests.integration.models import AIPerfMockServer


@pytest.mark.stress
@pytest.mark.integration
@pytest.mark.asyncio
class TestStressScenarios:
    """Tests for high concurrency and stress scenarios."""

    async def test_high_concurrency_multimodal(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """High concurrency (1000) with streaming and multimodal inputs."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
                --url {aiperf_mock_server.url} \
                --gpu-telemetry {" ".join(aiperf_mock_server.dcgm_urls)} \
                --endpoint-type chat \
                --streaming \
                --warmup-request-count 1000 \
                --request-count 1000 \
                --concurrency 1000 \
                --request-rate 1000 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --workers-max 5 \
                --record-processors 5 \
                --ui {defaults.ui}
            """,
            timeout=180.0,
        )
        assert result.request_count == 1000
        assert result.has_streaming_metrics

    async def test_high_worker_count_streaming(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """High worker count (100 workers) with streaming."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --gpu-telemetry {" ".join(aiperf_mock_server.dcgm_urls)} \
                --endpoint-type chat \
                --concurrency 2000 \
                --request-count 4000 \
                --osl 50 \
                --workers-max 100 \
                --streaming \
                --ui dashboard
            """,
            timeout=180.0,
        )
        assert result.request_count == 4000
        assert result.has_streaming_metrics
