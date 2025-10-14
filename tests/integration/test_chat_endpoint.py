# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for /v1/chat/completions endpoint."""

import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.models import FakeAIServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestChatEndpoint:
    """Tests for /v1/chat/completions endpoint."""

    async def test_basic_chat(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Basic non-streaming chat completion."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model microsoft/phi-4 \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --ui simple
            """
        )
        assert result.request_count == 10
