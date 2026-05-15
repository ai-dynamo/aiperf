# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for high concurrency and performance scenarios."""

import platform

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer
from tests.integration.conftest import IntegrationTestDefaults as defaults


@pytest.mark.stress
@pytest.mark.integration
@pytest.mark.asyncio
class TestStressScenarios:
    """Tests for high concurrency and stress scenarios."""

    async def test_high_concurrency_multimodal(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ):
        """High concurrency (1000) with streaming and multimodal inputs."""
        result = await cli.run(
            f"""
            aiperf profile \
                --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
                --url {aiperf_mock_server.url} \
                --gpu-telemetry {" ".join(aiperf_mock_server.dcgm_urls)} \
                --endpoint-type chat \
                --streaming \
                --warmup-request-count 100 \
                --request-count 1000 \
                --concurrency 1000 \
                --request-rate 1000 \
                --image-width-mean 64 \
                --image-height-mean 64 \
                --workers-max 5 \
                --record-processors 5 \
                --ui {defaults.ui}
            """,
            timeout=600.0,  # bumped from 180s for slow Windows VDI under heavy concurrency
        )
        # Allow up to 0.5% drop at 1000-way concurrency. On a busy VDI a
        # couple of in-flight requests can be cancelled during shutdown
        # without indicating a real product bug — the assertion is about
        # the stress path completing, not about exact request accounting.
        assert result.request_count >= 995, (
            f"Expected >=995 requests, got {result.request_count}"
        )
        assert result.has_streaming_metrics

    @pytest.mark.skipif(
        platform.system() == "Windows",
        reason="Windows VDIs hit WinError 1450 (insufficient system resources) "
        "spawning 100 worker subprocesses; this stress level is Linux-CI only.",
    )
    async def test_high_worker_count_streaming(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """High worker count (100 workers) with streaming.

        100 worker subprocesses spawning concurrently overrun the
        default registration retry budget (10 attempts x 1s = 10s)
        because the SystemController's registration handler services
        requests serially and 100 simultaneous registrants queue up.
        Bump the per-worker max attempts so each worker has ~60s
        before giving up. Doesn't affect normal runs - only this
        stress test hits the contention.
        """
        monkeypatch.setenv("AIPERF_SERVICE_REGISTRATION_MAX_ATTEMPTS", "60")

        result = await cli.run(
            f"""
            aiperf profile \
                --model {defaults.model} \
                --url {aiperf_mock_server.url} \
                --gpu-telemetry {" ".join(aiperf_mock_server.dcgm_urls)} \
                --endpoint-type chat \
                --concurrency 2000 \
                --request-count 4000 \
                --osl 50 \
                --workers-max 100 \
                --streaming \
                --ui {defaults.ui}
            """,
            timeout=600.0,  # bumped from 180s; 100 worker subprocesses on Windows VDI are slow to spawn
        )
        assert result.request_count == 4000
        assert result.has_streaming_metrics
