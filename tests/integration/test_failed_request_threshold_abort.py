# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression test: --failed-request-threshold abort exits NON-ZERO.

The abort itself already existed (records broadcasts ProfileCancelCommand when
the profiling failure rate exceeds the threshold), but it used to exit 0 -- an
invalid run looked like a success to CI/automation. The exit-code polish makes
both abort paths (this one and the warmup-failure path) exit non-zero. Real
process, real mock server returning 100% errors during PROFILING.
"""

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
    """Profiling failure rate over the threshold -> abort -> non-zero exit.

    concurrency=2 => grace floor max(2,10)=10, so --request-count 30 with a
    100%-error server crosses the floor and trips the abort."""
    async with mock_server_factory(fast=True, error_rate=100) as server:
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {server.url} \
                --endpoint-type chat \
                --request-count 30 \
                --concurrency {defaults.concurrency} \
                --failed-request-threshold 0.1 \
                --workers-max {defaults.workers_max} \
                --ui {defaults.ui}
            """,
            timeout=120.0,
            assert_success=False,
        )

    assert result.exit_code != 0, (
        f"threshold abort must exit non-zero, got {result.exit_code}\n"
        f"{(result.log or '')[-1500:]}"
    )
    log = result.log or ""
    assert "Run aborted (failed_request_threshold)" in log, (
        f"missing threshold abort-reason marker (exit-code path)\n{log[-2000:]}"
    )
