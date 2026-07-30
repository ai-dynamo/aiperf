# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end regression test for live failed-request threshold aborts."""

from __future__ import annotations

import pytest

from tests.harness.utils import AIPerfCLI
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.integration
@pytest.mark.asyncio
async def test_failed_request_threshold_aborts_nonzero(
    cli: AIPerfCLI,
    mock_server_factory,
) -> None:
    """An unhealthy inference server terminates profiling with a failure."""
    async with mock_server_factory(fast=True, error_rate=100) as server:
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {server.url} \
                --endpoint-type chat \
                --request-count 1000 \
                --concurrency {defaults.concurrency} \
                --failed-request-threshold 0.1 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=30.0,
            assert_success=False,
        )

    assert result.exit_code != 0
    log = result.log or ""
    assert "--failed-request-threshold exceeded: 10/10" in log
    assert "Run aborted (failed_request_threshold)" in log
