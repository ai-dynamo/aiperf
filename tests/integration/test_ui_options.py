# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for different UI modes."""

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
class TestUIOptions:
    """Tests for different UI modes."""

    async def test_none_ui(self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer):
        """None UI mode (no interactive output)."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --endpoint-type chat \
                --request-count {defaults.request_count} \
                --concurrency {defaults.concurrency} \
                --workers-max {defaults.workers_max} \
                --ui none
            """
        )
        assert result.request_count == defaults.request_count
        assert result.has_all_outputs
