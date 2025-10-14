# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for request cancellation functionality."""

import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.models import FakeAIServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestRequestCancellation:
    """Tests for request cancellation functionality."""

    async def test_request_cancellation(
        self, cli: AIPerfCLI, fakeai_server: FakeAIServer
    ):
        """Request cancellation doesn't break pipeline."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-120b \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --streaming \
                --request-count 50 \
                --concurrency 5 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --request-cancellation-rate 0.3 \
                --request-cancellation-delay 0.5 \
                --workers-max 1 \
                --ui simple
            """,
            timeout=120.0,
        )
        assert result.request_count >= 30
