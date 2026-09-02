# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for URL validation on EndpointConfig.

A bare ``:18765`` (host-less URL) was being silently accepted at config parse
time and only surfaced as 20x InvalidUrlClientError at request time, making
the user think the server was unreachable when in reality their URL was
malformed. These tests pin the fail-fast behavior at config validation.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.config.endpoint import EndpointConfig


@pytest.mark.parametrize(
    ("url", "expected_fragment"),
    [
        param(":18765", "missing scheme or host", id="hostless-port-only"),
        param("/path/only", "missing scheme or host", id="path-only"),
        param("ftp://host:21", "unsupported scheme", id="ftp-scheme"),
    ],
)
def test_endpoint_config_rejects_invalid_urls(url: str, expected_fragment: str) -> None:
    """Invalid URLs raise ValidationError at config parse time, not at request time."""
    with pytest.raises(ValidationError) as exc_info:
        EndpointConfig(urls=[url])
    assert expected_fragment in str(exc_info.value)
    assert repr(url) in str(exc_info.value)


def test_endpoint_config_normalizes_schemeless_localhost() -> None:
    """``localhost:18765`` is normalized to ``http://localhost:18765``."""
    cfg = EndpointConfig(urls=["localhost:18765"])
    assert cfg.urls == ["http://localhost:18765"]


@pytest.mark.parametrize(
    ("url", "endpoint_type"),
    [
        param("http://localhost:18765", "chat", id="http-localhost-port"),
        param("https://api.example.com/v1", "chat", id="https-with-path"),
        param("http://10.0.0.1:8000", "chat", id="http-ip-port"),
        param("ws://localhost:19000", "responses", id="ws-localhost-port"),
        param("wss://api.example.com/v1/responses", "responses", id="wss-with-path"),
    ],
)  # fmt: skip
def test_endpoint_config_accepts_valid_urls(url: str, endpoint_type: str) -> None:
    """Standard http(s) and ws(s) URLs with explicit host pass validation.

    WebSocket URLs pair with endpoint type 'responses' since the WebSocket
    transport only speaks the Responses API contract.
    """
    cfg = EndpointConfig(urls=[url], type=endpoint_type)
    assert cfg.urls == [url]


def test_endpoint_config_rejects_http_with_empty_host() -> None:
    """``http://:18765`` (post-normalization of bare ``:18765``) must fail.

    urlparse parses this as scheme=http, netloc=':18765', hostname=None — a
    truthy netloc that fooled the prior validator. This is the exact bug the
    smoke test surfaced.
    """
    with pytest.raises(ValidationError) as exc_info:
        EndpointConfig(urls=["http://:18765"])
    assert "missing scheme or host" in str(exc_info.value)


# --- WebSocket transport requires the Responses API contract -----------------


@pytest.mark.parametrize(
    ("url", "transport"),
    [
        param("ws://localhost:19000", None, id="ws-url-autodetect"),
        param("wss://api.example.com/v1/responses", None, id="wss-url-autodetect"),
        param("http://localhost:8000", "websocket", id="explicit-transport-http-url"),
    ],
)  # fmt: skip
def test_websocket_transport_rejects_non_responses_type(
    url: str, transport: str | None
) -> None:
    """A ws/wss URL or --transport websocket requires endpoint type 'responses'."""
    with pytest.raises(ValidationError, match="only supported for endpoint type"):
        EndpointConfig(urls=[url], transport=transport)  # type defaults to chat


@pytest.mark.parametrize(
    ("url", "transport"),
    [
        param("ws://localhost:19000", None, id="ws-url-autodetect"),
        param("wss://api.example.com/v1/responses", None, id="wss-url-autodetect"),
        param("ws://localhost:19000", "websocket", id="explicit-transport"),
    ],
)  # fmt: skip
def test_websocket_transport_accepts_responses_type(
    url: str, transport: str | None
) -> None:
    """WebSocket paired with endpoint type 'responses' validates cleanly."""
    cfg = EndpointConfig(urls=[url], type="responses", transport=transport)
    assert cfg.urls == [url]


# --- Credential-bearing WebSocket connections require TLS (wss://) -----------


def test_ws_with_api_key_rejected() -> None:
    """An API key over unencrypted ws:// would leak the credential in cleartext."""
    with pytest.raises(ValidationError, match="unencrypted 'ws://'"):
        EndpointConfig(
            urls=["ws://localhost:19000"], type="responses", api_key="secret"
        )


def test_ws_with_auth_header_rejected() -> None:
    """Authentication headers over ws:// are cleartext-exposed."""
    with pytest.raises(ValidationError, match="unencrypted 'ws://'"):
        EndpointConfig(
            urls=["ws://localhost:19000"],
            type="responses",
            headers={"Authorization": "Bearer token"},
        )


def test_wss_with_api_key_allowed() -> None:
    """Credentials are fine over TLS-encrypted wss://."""
    cfg = EndpointConfig(
        urls=["wss://api.example.com/v1/responses"],
        type="responses",
        api_key="secret",
    )
    assert cfg.api_key == "secret"


def test_ws_without_credentials_allowed() -> None:
    """A plain ws:// with no credentials is permitted."""
    cfg = EndpointConfig(urls=["ws://localhost:19000"], type="responses")
    assert cfg.urls == ["ws://localhost:19000"]


def test_ws_with_non_sensitive_header_allowed() -> None:
    """A benign (non-credential) header does not trip the TLS gate."""
    cfg = EndpointConfig(
        urls=["ws://localhost:19000"],
        type="responses",
        headers={"X-Trace-Id": "abc"},
    )
    assert cfg.urls == ["ws://localhost:19000"]
