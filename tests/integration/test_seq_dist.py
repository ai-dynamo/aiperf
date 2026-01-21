# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for sequence length distribution.
"""

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
class TestSequenceLengthDistribution:
    """Test SequenceLengthDistribution functionality."""

    async def test_sequence_length_distribution(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test sequence length distribution."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model Qwen/Qwen3-0.6B \
                --endpoint-type chat \
                --endpoint /v1/chat/completions \
                --streaming \
                --url {aiperf_mock_server.url} \
                --sequence-distribution "64|10,32|8:70;256|40,128|20:20;1024|100,512|50:10" \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=120.0,
        )
        for request in result.jsonl:
            assert request.metrics.get("input_sequence_length") is not None
            assert request.metrics.get("output_sequence_length") is not None
            assert request.metrics.get("input_sequence_length").value > 0
            assert request.metrics.get("output_sequence_length").value > 0
