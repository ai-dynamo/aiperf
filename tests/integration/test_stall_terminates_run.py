# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A stalled benchmark must end the process, not just report a stall.

The watchdog's unit tests stub ``_request_profile_cancel``, so they prove the
stall is noticed and prove nothing about whether the run then stops. It did
not: publishing a terminal result fills only the ``profile`` result domain,
and the controller keeps waiting on ``server_metrics``, whose result cannot
arrive until the profiling phase completes -- which the stuck request is
exactly what prevents. The run hung until it was killed.

Server metrics are enabled by default whenever the endpoint exposes
``/metrics``, which the mock server does, so this must run without
``--no-server-metrics`` to cover the default configuration at all.
"""

import pytest

from tests.harness.utils import AIPerfCLI
from tests.integration.conftest import IntegrationTestDefaults as defaults

# Long enough that the request never returns on its own, so the only way the
# process can exit is the watchdog.
_NEVER_RETURNS_MS = 600_000
_STALL_TIMEOUT_S = 5


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_stalled_run_exits_instead_of_waiting_on_server_metrics(
    cli: AIPerfCLI, mock_server_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AIPERF_RECORD_PROGRESS_STALL_TIMEOUT", str(_STALL_TIMEOUT_S))
    monkeypatch.setenv("AIPERF_RECORD_PROGRESS_STALL_CHECK_INTERVAL", "1")

    async with mock_server_factory(ttft=_NEVER_RETURNS_MS) as server:
        # A timeout here raises rather than returning, so a run that hangs
        # fails this test -- which is the regression being guarded.
        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --tokenizer builtin \
                --url {server.url} \
                --endpoint-type chat \
                --streaming \
                --concurrency 1 \
                --request-count 5 \
                --ui none
            """,
            timeout=90.0,
            assert_success=False,
        )

    output = result.stdout + result.stderr
    assert result.exit_code == 1, output[-2000:]
    assert "Benchmark stalled" in output, output[-2000:]
    # Without this the test could pass vacuously: if server metrics never
    # engaged, the profile domain would be the only one to finalize and the
    # original defect would be invisible here.
    assert "server_metrics" in output, (
        f"server metrics never engaged, so this run did not cover the "
        f"configuration the defect needs: {output[-2000:]}"
    )
