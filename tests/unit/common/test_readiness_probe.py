# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

import orjson
import pytest

from aiperf.auth.base_signer import SignedRequest
from aiperf.common import readiness_probe
from aiperf.config.flags.cli_config import CLIConfig


class _FakeRecord:
    status: int = 400
    error: None = None


class _FakeClient:
    def __init__(self, status: int = 400) -> None:
        self.posted_urls: list[str] = []
        self.payloads: list[dict[str, Any]] = []
        self._status = status

    async def post_request(
        self,
        request_url: str,
        payload: bytes,
        headers: dict[str, str],
        timeout: object,
        **kwargs: Any,
    ) -> _FakeRecord:
        del headers, timeout, kwargs
        decoded_payload = orjson.loads(payload)
        assert isinstance(decoded_payload, dict)
        self.posted_urls.append(request_url)
        self.payloads.append(decoded_payload)
        record = _FakeRecord()
        record.status = self._status
        return record


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
            signer=None,
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
            signer=None,
        )
    )

    assert "no dedicated request template" not in caplog.text
    assert client.posted_urls == ["http://server/v1/embeddings"]
    assert client.payloads == [{"input": "Lo", "model": "embedder"}]


def test_wait_inference_messages_uses_dedicated_template(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="aiperf.common.readiness_probe")
    client = _FakeClient()

    asyncio.run(
        readiness_probe._wait_inference(
            client=cast(Any, client),
            url="http://server",
            model_name="claude-sonnet-4-20250514",
            endpoint_type="messages",
            custom_endpoint=None,
            timeout_s=1.0,
            interval_s=0.1,
            headers={},
            signer=None,
        )
    )

    # The Messages endpoint has a dedicated template, so no chat fallback
    # warning and the probe hits /v1/messages with a Messages-shaped body
    # (max_tokens is required by the Anthropic API).
    assert "no dedicated request template" not in caplog.text
    assert client.posted_urls == ["http://server/v1/messages"]
    assert client.payloads == [
        {
            "messages": [{"role": "user", "content": "Lo"}],
            "max_tokens": 1,
            "model": "claude-sonnet-4-20250514",
        }
    ]


class _FakeSigner:
    """Minimal stand-in for a configured RequestSignerProtocol."""

    async def sign(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> SignedRequest:
        return SignedRequest(headers=headers)


@pytest.mark.parametrize("status", [401, 403])
def test_wait_inference_signed_401_403_raises_instead_of_ready(status: int) -> None:
    """A signed probe rejected with 401/403 must not be treated as ready.

    Unlike the unsigned case (any status < 500 counts as ready), a rejected
    *signed* request almost always means the signature itself is
    misconfigured (bad credentials/region/service), so the probe should
    fail fast with a clear error rather than silently reporting ready.
    """
    client = _FakeClient(status=status)

    with pytest.raises(RuntimeError, match=str(status)):
        asyncio.run(
            readiness_probe._wait_inference(
                client=cast(Any, client),
                url="http://server",
                model_name="model-a",
                endpoint_type="chat",
                custom_endpoint=None,
                timeout_s=1.0,
                interval_s=0.1,
                headers={},
                signer=cast(Any, _FakeSigner()),
            )
        )


def test_wait_inference_unsigned_403_still_counts_as_ready() -> None:
    """Without a signer, 401/403 keeps the documented < 500 behavior."""
    client = _FakeClient(status=403)

    asyncio.run(
        readiness_probe._wait_inference(
            client=cast(Any, client),
            url="http://server",
            model_name="model-a",
            endpoint_type="chat",
            custom_endpoint=None,
            timeout_s=1.0,
            interval_s=0.1,
            headers={},
            signer=None,
        )
    )

    assert client.posted_urls == ["http://server/v1/chat/completions"]


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
        self, url: str, headers: dict[str, str], timeout: object, **kwargs: Any
    ) -> _FakeReadyRecord:
        del headers, timeout, kwargs
        self.urls.append(url)
        return _FakeReadyRecord(self._models_payload.decode("utf-8"))

    async def post_request(
        self,
        request_url: str,
        payload: bytes,
        headers: dict[str, str],
        timeout: object,
        **kwargs: Any,
    ) -> _FakeRecord:
        del payload, headers, timeout, kwargs
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
            signer=None,
        )
    )

    assert fake.urls, "wait_for_endpoint should have made at least one request"
    for url in fake.urls:
        assert url.startswith(("http://", "https://")), (
            f"URL {url!r} reached the HTTP client without a scheme — "
            f"EndpointConfig normalization is broken"
        )
    assert fake.urls[0] == "http://localhost:8000/v1/models"


@pytest.mark.parametrize(
    "mode, base_path, custom_endpoint, expected_paths",
    [
        pytest.param(
            "both", "", None, ["/v1/models", "/v1/chat/completions"], id="default"
        ),
        pytest.param(
            "inference",
            "/v1/chat/completions/",
            None,
            ["/v1/chat/completions"],
            id="existing-path",
        ),
        pytest.param(
            "both",
            "/proxy/",
            "/generate",
            ["/proxy/v1/models", "/proxy/generate"],
            id="custom-path",
        ),
    ],
)
def test_readiness_preserves_query_while_appending_endpoint(
    mode: str, base_path: str, custom_endpoint: str | None, expected_paths: list[str]
) -> None:
    """Probe paths must precede repeated queries, including trailing slashes in values."""
    client = _FakeMultiClient()
    query = "?tag=first&mode=strict&tag=&tag=last&prefix=%2Fraw/"
    common = dict(
        client=cast(Any, client),
        url="http://server" + base_path + query,
        model_name="served-model",
        timeout_s=1.0,
        interval_s=0.1,
        headers={},
        signer=None,
    )
    if mode == "both":
        asyncio.run(readiness_probe._wait_models(**common))
    asyncio.run(
        readiness_probe._wait_inference(
            **common,
            endpoint_type="chat",
            custom_endpoint=custom_endpoint,
        )
    )
    assert client.urls == ["http://server" + path + query for path in expected_paths]


class _KwargCapturingClient:
    """Records the keyword arguments each probe request was issued with."""

    def __init__(self, models_body: str | None = None) -> None:
        self._models_body = models_body or orjson.dumps(
            {"data": [{"id": "model-a"}]}
        ).decode("utf-8")
        self.get_kwargs: list[dict[str, Any]] = []
        self.post_kwargs: list[dict[str, Any]] = []

    async def get_request(
        self, url: str, headers: dict[str, str], **kwargs: Any
    ) -> _FakeReadyRecord:
        del url, headers
        self.get_kwargs.append(kwargs)
        return _FakeReadyRecord(self._models_body)

    async def post_request(
        self, request_url: str, payload: bytes, headers: dict[str, str], **kwargs: Any
    ) -> _FakeRecord:
        del request_url, payload, headers
        self.post_kwargs.append(kwargs)
        return _FakeRecord()

    async def close(self) -> None:
        return None


class TestSignedProbesDoNotFollowRedirects:
    """These signed probe requests exist only because SigV4 support was added,
    so the exposure is introduced here rather than inherited. A redirect would
    hand ``x-amz-security-token`` -- a bearer credential -- to whatever origin
    the benchmarked endpoint names.
    """

    def test_signed_models_probe_refuses_redirects(self) -> None:
        client = _KwargCapturingClient()

        asyncio.run(
            readiness_probe._wait_models(
                client=cast(Any, client),
                url="http://server",
                model_name="model-a",
                timeout_s=1.0,
                interval_s=0.1,
                headers={},
                signer=cast(Any, _FakeSigner()),
            )
        )

        assert client.get_kwargs
        assert all(k.get("allow_redirects") is False for k in client.get_kwargs)

    def test_signed_inference_probe_refuses_redirects(self) -> None:
        client = _KwargCapturingClient()

        asyncio.run(
            readiness_probe._wait_inference(
                client=cast(Any, client),
                url="http://server",
                model_name="model-a",
                endpoint_type="chat",
                custom_endpoint=None,
                timeout_s=1.0,
                interval_s=0.1,
                headers={},
                signer=cast(Any, _FakeSigner()),
            )
        )

        assert client.post_kwargs
        assert all(k.get("allow_redirects") is False for k in client.post_kwargs)

    def test_unsigned_probes_still_follow_redirects(self) -> None:
        client = _KwargCapturingClient()

        asyncio.run(
            readiness_probe._wait_inference(
                client=cast(Any, client),
                url="http://server",
                model_name="model-a",
                endpoint_type="chat",
                custom_endpoint=None,
                timeout_s=1.0,
                interval_s=0.1,
                headers={},
                signer=None,
            )
        )

        assert client.post_kwargs
        assert all("allow_redirects" not in k for k in client.post_kwargs)


class _RejectingRecord:
    """A probe response carrying an upstream rejection body."""

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.error = type("_Err", (), {"message": message})()
        self.responses: list[Any] = []


class _RejectingClient:
    def __init__(self, status: int, message: str = "") -> None:
        self._status = status
        self._message = message

    async def post_request(
        self, request_url: str, payload: bytes, headers: dict[str, str], **kwargs: Any
    ) -> _RejectingRecord:
        del request_url, payload, headers, kwargs
        return _RejectingRecord(self._status, self._message)


class TestSignedRejectionDiagnosisIsHonest:
    """A 401/403 on a signed probe still stops preflight -- retrying fixes none
    of the plausible causes -- but the message must not assert a signature
    mismatch. API Gateway answers an unrouted path with 403 ``Missing
    Authentication Token``, and ``execute-api`` is this feature's headline
    target, so a correctly signed run can land here with nothing whatsoever
    wrong with its signature.
    """

    def test_a_plain_403_lists_alternatives_rather_than_asserting_a_cause(
        self,
    ) -> None:
        client = _RejectingClient(status=403)

        with pytest.raises(RuntimeError) as excinfo:
            asyncio.run(
                readiness_probe._wait_inference(
                    client=cast(Any, client),
                    url="http://server",
                    model_name="model-a",
                    endpoint_type="chat",
                    custom_endpoint=None,
                    timeout_s=1.0,
                    interval_s=0.1,
                    headers={},
                    signer=cast(Any, _FakeSigner()),
                )
            )

        message = str(excinfo.value)
        assert "may mean" in message
        assert "IAM" in message
        assert "does not route" in message

    def test_api_gateway_unrouted_path_is_named_as_such(self) -> None:
        client = _RejectingClient(status=403, message="Missing Authentication Token")

        with pytest.raises(RuntimeError) as excinfo:
            asyncio.run(
                readiness_probe._wait_inference(
                    client=cast(Any, client),
                    url="http://server",
                    model_name="model-a",
                    endpoint_type="chat",
                    custom_endpoint=None,
                    timeout_s=1.0,
                    interval_s=0.1,
                    headers={},
                    signer=cast(Any, _FakeSigner()),
                )
            )

        message = str(excinfo.value)
        assert "does not route" in message
        # The signature is fine in this case; do not send the user chasing it.
        assert "--aws-region" not in message
