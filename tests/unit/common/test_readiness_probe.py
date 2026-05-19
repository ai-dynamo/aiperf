# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

import orjson
import pytest

from aiperf.common import readiness_probe
from aiperf.config.flags.cli_config import CLIConfig


class _FakeRecord:
    status: int = 400
    error: None = None


class _FakeClient:
    def __init__(self) -> None:
        self.posted_urls: list[str] = []
        self.payloads: list[dict[str, Any]] = []

    async def post_request(
        self,
        request_url: str,
        payload: bytes,
        headers: dict[str, str],
        timeout: object,
    ) -> _FakeRecord:
        del headers, timeout
        decoded_payload = orjson.loads(payload)
        assert isinstance(decoded_payload, dict)
        self.posted_urls.append(request_url)
        self.payloads.append(decoded_payload)
        return _FakeRecord()


def test_wait_inference_warns_on_endpoint_type_fallback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="aiperf.common.readiness_probe")
    client = _FakeClient()

    asyncio.run(
        readiness_probe._wait_inference(
            client=cast(Any, client),
            url="http://server",
            model_name="model-a",
            endpoint_type="responses",
            custom_endpoint=None,
            timeout_s=1.0,
            interval_s=0.1,
            headers={},
        )
    )

    assert "endpoint type 'responses'" in caplog.text
    assert "may not prove model readiness" in caplog.text
    assert client.posted_urls == ["http://server/v1/chat/completions"]
    assert client.payloads[0]["model"] == "model-a"
    assert "messages" in client.payloads[0]


def test_wait_inference_dedicated_template_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="aiperf.common.readiness_probe")
    client = _FakeClient()

    asyncio.run(
        readiness_probe._wait_inference(
            client=cast(Any, client),
            url="http://server",
            model_name="embedder",
            endpoint_type="embeddings",
            custom_endpoint=None,
            timeout_s=1.0,
            interval_s=0.1,
            headers={},
        )
    )

    assert "no dedicated request template" not in caplog.text
    assert client.posted_urls == ["http://server/v1/embeddings"]
    assert client.payloads == [{"input": "Lo", "model": "embedder"}]


class _FakeReadyRecord:
    """A get_request response that the probe treats as 'server live'."""

    status: int = 200
    error: None = None
    responses: list[Any]

    def __init__(self, body: str) -> None:
        text_resp = type("_Resp", (), {"text": body})()
        self.responses = [text_resp]


class _FakeMultiClient:
    """Captures every URL passed to get_request/post_request for assertion."""

    def __init__(self, models_payload: bytes | None = None) -> None:
        self.urls: list[str] = []
        self._models_payload = models_payload or orjson.dumps(
            {"data": [{"id": "served-model"}]}
        )

    async def get_request(
        self, url: str, headers: dict[str, str], timeout: object
    ) -> _FakeReadyRecord:
        del headers, timeout
        self.urls.append(url)
        return _FakeReadyRecord(self._models_payload.decode("utf-8"))

    async def post_request(
        self,
        request_url: str,
        payload: bytes,
        headers: dict[str, str],
        timeout: object,
    ) -> _FakeRecord:
        del payload, headers, timeout
        self.urls.append(request_url)
        return _FakeRecord()

    async def close(self) -> None:
        return None


def test_wait_for_endpoint_receives_normalized_urls_from_endpoint_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: the readiness probe must see scheme-prefixed URLs.

    Before the fix, `EndpointConfig(urls=["localhost:8000"]).urls` returned
    the raw string and `wait_for_endpoint` passed `localhost:8000/v1/models`
    to aiohttp, which raised NonHttpUrlClientError.

    This test wires `EndpointConfig` to a fake aiohttp client and asserts
    every URL the client sees is well-formed (starts with `http://` or
    `https://`).
    """

    fake = _FakeMultiClient(
        models_payload=orjson.dumps({"data": [{"id": "served-model"}]})
    )

    # `wait_for_endpoint` constructs `AioHttpClient` internally — patch the
    # import so it returns our fake instead.
    monkeypatch.setattr(
        "aiperf.transports.aiohttp_client.AioHttpClient",
        lambda *args, **kwargs: fake,
    )

    config = CLIConfig(model_names=["served-model"], urls=["localhost:8000"])
    assert config.urls == ["http://localhost:8000"], (
        "EndpointConfig must prepend http:// to scheme-less URLs"
    )

    asyncio.run(
        readiness_probe.wait_for_endpoint(
            urls=config.urls,
            model_names=config.model_names,
            mode="models",
            endpoint_type="chat",
            custom_endpoint=None,
            timeout_s=2.0,
            interval_s=0.1,
            headers={},
        )
    )

    assert fake.urls, "wait_for_endpoint should have made at least one request"
    for url in fake.urls:
        assert url.startswith(("http://", "https://")), (
            f"URL {url!r} reached the HTTP client without a scheme — "
            f"EndpointConfig normalization is broken"
        )
    assert fake.urls[0] == "http://localhost:8000/v1/models"
