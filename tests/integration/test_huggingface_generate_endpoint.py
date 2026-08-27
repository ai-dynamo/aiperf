# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
class TestHuggingFaceGenerateEndpoint:
    """Integration tests for huggingface_generate endpoint."""

    @pytest.mark.parametrize(
        "streaming,expect_streaming_metrics",
        [
            (False, False),
            (True, True),
        ],
        ids=["non_streaming", "streaming"],
    )
    async def test_huggingface_generate(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        streaming: bool,
        expect_streaming_metrics: bool,
    ):
        """Test huggingface_generate endpoint with streaming and non-streaming modes."""
        stream_flag = "--streaming" if streaming else ""

        result = await cli.run(
            f"""
            aiperf profile \
                --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
                --url {aiperf_mock_server.url} \
                --endpoint-type huggingface_generate \
                {stream_flag} \
                --request-count {defaults.request_count}
            """
        )

        assert result.request_count == defaults.request_count
        assert result.has_streaming_metrics == expect_streaming_metrics
