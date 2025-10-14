# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for warmup phase functionality."""

import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.models import FakeAIServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestWarmup:
    """Tests for warmup phase functionality."""

    async def test_warmup_phase(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Warmup requests excluded from profiling metrics."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-120b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --warmup-request-count 5 \
                --request-count 15 \
                --concurrency 2 \
                --workers-max 1 \
                --ui simple
            """
        )
        assert result.request_count == 15

    async def test_warmup_with_streaming(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """Warmup with streaming enabled."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-120b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --streaming \
                --warmup-request-count 10 \
                --request-count 20 \
                --concurrency 4 \
                --workers-max 1 \
                --ui simple
            """
        )
        assert result.request_count == 20
        assert result.has_streaming_metrics
