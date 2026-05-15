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
    "url",
    [
        param("http://localhost:18765", id="http-localhost-port"),
        param("https://api.example.com/v1", id="https-with-path"),
        param("http://10.0.0.1:8000", id="http-ip-port"),
    ],
)
def test_endpoint_config_accepts_valid_urls(url: str) -> None:
    """Standard http(s) URLs with explicit host pass validation."""
    cfg = EndpointConfig(urls=[url])
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
