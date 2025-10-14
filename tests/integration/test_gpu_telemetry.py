# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for GPU telemetry collection and reporting."""

import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.models import FakeAIServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestGpuTelemetry:
    """Tests for GPU telemetry collection and reporting."""

    async def test_gpu_telemetry(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """GPU telemetry collection with DCGM endpoint."""
        dcgm_url = f"{fakeai_server.url}/dcgm"
        result = await cli.run(
            f"""
            aiperf profile \
                --model nvidia/llama-3.1-nemotron-70b-instruct \
                --url {fakeai_server.url} \
                --tokenizer gpt2 \
                --endpoint-type chat \
                --gpu-telemetry {dcgm_url} \
                --streaming \
                --request-count 100 \
                --concurrency 10 \
                --workers-max 2 \
                --ui dashboard
            """
        )
        dcgm_url = dcgm_url.replace("http://", "")
        assert result.request_count == 100
        assert result.has_gpu_telemetry
        assert result.json.telemetry_data.endpoints is not None
        assert len(result.json.telemetry_data.endpoints) > 0
        assert result.json.telemetry_data.endpoints[dcgm_url].gpus is not None
        assert len(result.json.telemetry_data.endpoints[dcgm_url].gpus) > 0
        assert (
            result.json.telemetry_data.endpoints[dcgm_url].gpus["gpu_0"].metrics
            is not None
        )
        assert (
            len(result.json.telemetry_data.endpoints[dcgm_url].gpus["gpu_0"].metrics)
            > 0
        )
