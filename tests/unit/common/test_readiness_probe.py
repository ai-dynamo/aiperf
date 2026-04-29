# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

import orjson
import pytest

from aiperf.common import readiness_probe


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
