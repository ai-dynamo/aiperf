# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for GPU telemetry collection and reporting."""

import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.models import AIPerfMockServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestGpuTelemetry:
    """Tests for GPU telemetry collection and reporting."""

    async def test_gpu_telemetry(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """GPU telemetry collection with DCGM endpoint."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {aiperf_mock_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --gpu-telemetry {" ".join(aiperf_mock_server.dcgm_urls)} \
                --streaming \
                --request-count 100 \
                --concurrency 2 \
                --workers-max 2 \
                --ui dashboard
            """
        )
        assert result.request_count == 100
        assert result.has_gpu_telemetry
        assert result.json.telemetry_data.endpoints is not None
        assert len(result.json.telemetry_data.endpoints) > 0

        for dcgm_url in result.json.telemetry_data.endpoints:
            assert result.json.telemetry_data.endpoints[dcgm_url].gpus is not None
            assert len(result.json.telemetry_data.endpoints[dcgm_url].gpus) > 0

            for gpu_data in result.json.telemetry_data.endpoints[
                dcgm_url
            ].gpus.values():
                assert gpu_data.metrics is not None
                assert len(gpu_data.metrics) > 0

                for metric_value in gpu_data.metrics.values():
                    assert metric_value is not None
                    assert metric_value.avg is not None
                    assert metric_value.min is not None
                    assert metric_value.max is not None
                    assert metric_value.unit is not None
