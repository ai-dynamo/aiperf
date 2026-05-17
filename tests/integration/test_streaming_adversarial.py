# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial integration tests for endpoint streaming metric shape.

Focuses on:
- real ``aiperf profile`` subprocesses against the in-repo mock server
- stream-capable OpenAI chat, OpenAI completions, and Hugging Face generate endpoints
- honest metric shape: streaming runs expose TTFT/ITL-style metrics, while a
  non-streaming control does not report streaming-only metrics

Out of scope: endpoint parser unit edge cases and raw record wire-codec behavior,
which are covered by ``tests/unit/records/`` and ``tests/unit/common/messages/``.
"""

from __future__ import annotations

import pytest
from pytest import param

from tests.harness.utils import AIPerfCLI, AIPerfMockServer, AIPerfResults

_REQUEST_COUNT = 2
_CONCURRENCY = 1
_WORKERS_MAX = 1
_UI = "none"


# ============================================================================
# Helpers
# ============================================================================


def _profile_command(
    *,
    endpoint_type: str,
    model: str,
    server_url: str,
    streaming: bool,
) -> str:
    """Build a small real profile command for the mock server."""
    stream_flag = "--streaming" if streaming else ""
    return f"""
        aiperf profile \
            --model {model} \
            --url {server_url} \
            --endpoint-type {endpoint_type} \
            {stream_flag} \
            --request-count {_REQUEST_COUNT} \
            --concurrency {_CONCURRENCY} \
            --workers-max {_WORKERS_MAX} \
            --ui {_UI}
    """


def _assert_completed_expected_requests(result: AIPerfResults) -> None:
    assert result.request_count == _REQUEST_COUNT


# ============================================================================
# Streaming endpoint behavior
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestStreamingEndpointMetricsShape:
    """Verify stream-capable endpoints expose streaming metrics only when streaming."""

    @pytest.mark.parametrize(
        "endpoint_type,model",
        [
            param("chat", "Qwen/Qwen2.5-32B-Instruct", id="chat-streaming"),
            param("completions", "openai/gpt-oss-120b", id="completions-streaming"),
        ],
    )  # fmt: skip
    async def test_profile_streaming_openai_endpoint_reports_streaming_metrics(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        endpoint_type: str,
        model: str,
    ) -> None:
        result = await cli.run(
            _profile_command(
                endpoint_type=endpoint_type,
                model=model,
                server_url=aiperf_mock_server.url,
                streaming=True,
            )
        )

        _assert_completed_expected_requests(result)
        assert result.has_streaming_metrics

    async def test_profile_streaming_huggingface_generate_reports_streaming_metrics(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ) -> None:
        result = await cli.run(
            _profile_command(
                endpoint_type="huggingface_generate",
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                server_url=aiperf_mock_server.url,
                streaming=True,
            )
        )

        _assert_completed_expected_requests(result)
        assert result.has_streaming_metrics

    async def test_profile_non_streaming_chat_omits_streaming_metrics(
        self, cli: AIPerfCLI, aiperf_mock_server: AIPerfMockServer
    ) -> None:
        result = await cli.run(
            _profile_command(
                endpoint_type="chat",
                model="Qwen/Qwen2.5-32B-Instruct",
                server_url=aiperf_mock_server.url,
                streaming=False,
            )
        )

        _assert_completed_expected_requests(result)
        assert not result.has_streaming_metrics
