# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
class TestDefaultBehavior:
    """Tests for default behavior."""

    async def test_default_behavior(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """Test that only providing the model and nothing else still works.

        NOTE: We still have to provide the server's url due to the nature of it being on a non-default port.
        """
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url}
            """
        )
        assert result.request_count == defaults.request_count
        assert result.exit_code == 0
