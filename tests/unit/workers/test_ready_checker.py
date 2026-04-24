# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the endpoint readiness checker.

These cover the new multi-URL / multi-model / mode-aware signature. Deep
HTTP behavior is exercised via integration tests against the mock server;
here we focus on the dispatch logic and failure paths.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from aiperf.workers.ready_checker import wait_for_endpoint


def _record(status: int | None, text: str = "") -> SimpleNamespace:
    """Minimal RequestRecord-shaped object for patching AioHttpClient.get/post."""
    responses = [SimpleNamespace(text=text)] if text else []
    return SimpleNamespace(status=status, responses=responses, error=None)


class TestWaitForEndpointSkipConditions:
    @pytest.mark.asyncio
    async def test_skips_when_timeout_zero(self) -> None:
        with patch("aiperf.workers.ready_checker.AioHttpClient") as MockClient:
            await wait_for_endpoint(["http://x"], ["m"], timeout=0.0)
            MockClient.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_no_urls(self) -> None:
        with patch("aiperf.workers.ready_checker.AioHttpClient") as MockClient:
            await wait_for_endpoint([], ["m"], timeout=10.0)
            MockClient.assert_not_called()

    @pytest.mark.asyncio
    async def test_models_mode_requires_model_names(self) -> None:
        with pytest.raises(ValueError, match="requires at least one model name"):
            await wait_for_endpoint(["http://x"], [], mode="models", timeout=10.0)


class TestModelsMode:
    @pytest.mark.asyncio
    async def test_success_when_model_in_payload(self) -> None:
        body = '{"object":"list","data":[{"id":"m-1","object":"model"}]}'
        client = AsyncMock()
        client.get_request = AsyncMock(return_value=_record(200, body))
        client.close = AsyncMock()

        with patch("aiperf.workers.ready_checker.AioHttpClient", return_value=client):
            await wait_for_endpoint(["http://x"], ["m-1"], mode="models", timeout=10.0)

        client.get_request.assert_awaited_once()
        client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_404_falls_back_to_base_url(self) -> None:
        client = AsyncMock()
        client.get_request = AsyncMock(side_effect=[_record(404), _record(200, "ok")])
        client.close = AsyncMock()

        with patch("aiperf.workers.ready_checker.AioHttpClient", return_value=client):
            await wait_for_endpoint(["http://x"], ["m-1"], mode="models", timeout=10.0)

        assert client.get_request.await_count == 2
        first_url = client.get_request.await_args_list[0].args[0]
        second_url = client.get_request.await_args_list[1].args[0]
        assert first_url.endswith("/v1/models")
        assert second_url == "http://x"

    @pytest.mark.asyncio
    async def test_timeout_when_model_never_appears(self) -> None:
        empty_body = '{"object":"list","data":[]}'
        client = AsyncMock()
        client.get_request = AsyncMock(return_value=_record(200, empty_body))
        client.close = AsyncMock()

        with (
            patch("aiperf.workers.ready_checker.AioHttpClient", return_value=client),
            patch("aiperf.workers.ready_checker.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(TimeoutError, match="Timed out"),
        ):
            await wait_for_endpoint(
                ["http://x"],
                ["m-1"],
                mode="models",
                timeout=0.1,
                interval=0.01,
            )


class TestInferenceMode:
    @pytest.mark.asyncio
    async def test_success_on_first_post(self) -> None:
        client = AsyncMock()
        client.post_request = AsyncMock(return_value=_record(200))
        client.close = AsyncMock()

        with patch("aiperf.workers.ready_checker.AioHttpClient", return_value=client):
            await wait_for_endpoint(
                ["http://x"], ["m-1"], mode="inference", timeout=10.0
            )

        client.post_request.assert_awaited_once()
        post_url = client.post_request.await_args.args[0]
        assert post_url.endswith("/v1/chat/completions")

    @pytest.mark.asyncio
    async def test_4xx_counts_as_ready(self) -> None:
        client = AsyncMock()
        client.post_request = AsyncMock(return_value=_record(401))
        client.close = AsyncMock()

        with patch("aiperf.workers.ready_checker.AioHttpClient", return_value=client):
            await wait_for_endpoint(
                ["http://x"], ["m-1"], mode="inference", timeout=10.0
            )

        client.post_request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retries_on_5xx(self) -> None:
        client = AsyncMock()
        client.post_request = AsyncMock(side_effect=[_record(503), _record(200)])
        client.close = AsyncMock()

        with (
            patch("aiperf.workers.ready_checker.AioHttpClient", return_value=client),
            patch("aiperf.workers.ready_checker.asyncio.sleep", new_callable=AsyncMock),
        ):
            await wait_for_endpoint(
                ["http://x"],
                ["m-1"],
                mode="inference",
                timeout=10.0,
                interval=0.01,
            )

        assert client.post_request.await_count == 2


class TestMultiURLMultiModel:
    @pytest.mark.asyncio
    async def test_iterates_all_urls_and_models(self) -> None:
        body = '{"data":[{"id":"m-1"},{"id":"m-2"}]}'
        client = AsyncMock()
        client.get_request = AsyncMock(return_value=_record(200, body))
        client.close = AsyncMock()

        with patch("aiperf.workers.ready_checker.AioHttpClient", return_value=client):
            await wait_for_endpoint(
                ["http://a", "http://b"],
                ["m-1", "m-2"],
                mode="models",
                timeout=10.0,
            )

        # 2 URLs x 2 models = 4 probes
        assert client.get_request.await_count == 4

    @pytest.mark.asyncio
    async def test_both_mode_runs_models_then_inference_per_url(self) -> None:
        body = '{"data":[{"id":"m-1"}]}'
        client = AsyncMock()
        client.get_request = AsyncMock(return_value=_record(200, body))
        client.post_request = AsyncMock(return_value=_record(200))
        client.close = AsyncMock()

        with patch("aiperf.workers.ready_checker.AioHttpClient", return_value=client):
            await wait_for_endpoint(
                ["http://a", "http://b"],
                ["m-1"],
                mode="both",
                timeout=10.0,
            )

        # 2 URLs: models x 2 + inference x 2
        assert client.get_request.await_count == 2
        assert client.post_request.await_count == 2
