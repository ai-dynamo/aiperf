# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for different UI modes."""

import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.models import FakeAIServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestUIOptions:
    """Tests for different UI modes."""

    async def test_none_ui(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """None UI mode (no interactive output)."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model deepseek-ai/DeepSeek-R1 \
                --url {fakeai_server.url} \
                --endpoint-type chat \
                --request-count 10 \
                --concurrency 2 \
                --workers-max 1 \
                --ui none
            """
        )
        assert result.request_count == 10
        assert result.has_all_outputs
