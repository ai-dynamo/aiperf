# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for default behavior."""

import pytest

from tests.integration.conftest import AIPerfCLI
from tests.integration.models import FakeAIServer


@pytest.mark.integration
@pytest.mark.asyncio
class TestDefaultBehavior:
    """Tests for default behavior."""

    async def test_default_behavior(self, cli: AIPerfCLI, fakeai_server: FakeAIServer):
        """Test that only providing the model and nothing else still works.

        NOTE: We still have to provide the server's url due to the nature of it being on a non-default port.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model openai/gpt-oss-120b \
                --url {fakeai_server.url}
            """
        )
        assert result.request_count == 10
        assert result.exit_code == 0
