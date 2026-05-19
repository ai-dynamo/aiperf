# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adversarial regression tests for transport URL building and endpoint validation.

These pin behavior on the silent-wrong-URL bugs surfaced in
``/tmp/adversarial-transport.md``: query/fragment in base URL, the ``/v1`` dedup
gap on the custom-endpoint branch, uppercase scheme bypass, validator gaps
(garbage ports, whitespace, non-http schemes), and the empty-string
``custom_endpoint`` fall-through.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.config.endpoint import EndpointConfig
from aiperf.transports.aiohttp_transport import AioHttpTransport
from tests.unit.transports.conftest import create_model_endpoint_info
from tests.unit.transports.test_aiohttp_transport import create_request_info


def _get_url(base_url: str, custom_endpoint: str | None) -> str:
    """Build a transport, call get_url, and return the result."""
    model_endpoint = create_model_endpoint_info(
        base_url=base_url, custom_endpoint=custom_endpoint
    )
    transport = AioHttpTransport(model_endpoint=model_endpoint)
    request_info = create_request_info(model_endpoint)
    return transport.get_url(request_info)


@pytest.mark.parametrize(
    ("base_url", "custom_endpoint", "expected"),
    [
        param(
            "http://h/v1?key=abc",
            "/v1/chat",
            "http://h/v1/chat?key=abc",
            id="custom-branch-query-preserved",
        ),
        param(
            "http://h/v1/chat/completions?key=abc",
            "/v1/chat/completions",
            "http://h/v1/chat/completions?key=abc",
            id="custom-branch-query-preserved-on-dedup",
        ),
        param(
            "http://h/v1?key=abc",
            None,
            "http://h/v1/chat/completions?key=abc",
            id="metadata-branch-query-preserved",
        ),
    ],
)  # fmt: skip
def test_build_url_preserves_query_in_base(
    base_url: str, custom_endpoint: str | None, expected: str
) -> None:
    """A query string in the base URL must survive path-joining intact."""
    assert _get_url(base_url, custom_endpoint) == expected


@pytest.mark.parametrize(
    ("base_url", "custom_endpoint", "expected"),
    [
        param(
            "http://h/v1#frag",
            "/v1/chat",
            "http://h/v1/chat#frag",
            id="custom-branch-fragment-preserved",
        ),
        param(
            "http://h/v1#frag",
            None,
            "http://h/v1/chat/completions#frag",
            id="metadata-branch-fragment-preserved",
        ),
    ],
)  # fmt: skip
def test_build_url_preserves_fragment_in_base(
    base_url: str, custom_endpoint: str | None, expected: str
) -> None:
    """A fragment in the base URL must survive path-joining intact."""
    assert _get_url(base_url, custom_endpoint) == expected


@pytest.mark.parametrize(
    ("base_url", "custom_endpoint", "expected"),
    [
        # The /v1 dedup logic was previously only applied in the metadata
        # branch; it must apply in the custom-endpoint branch too so users
        # who pass ``/v1`` as the base and ``/v1/<path>`` as the custom
        # endpoint do not get a doubled path.
        param(
            "http://h/v1",
            "/v1/chat",
            "http://h/v1/chat",
            id="v1-base-leading-slash-custom",
        ),
        param(
            "http://h/v1",
            "v1/chat",
            "http://h/v1/chat",
            id="v1-base-no-leading-slash-custom",
        ),
        # Trailing slash on /v1/ base must still dedup against v1/<path>.
        param(
            "http://h/v1/",
            "/v1/chat",
            "http://h/v1/chat",
            id="v1-base-trailing-slash-custom",
        ),
    ],
)  # fmt: skip
def test_build_url_dedups_in_custom_endpoint_branch(
    base_url: str, custom_endpoint: str, expected: str
) -> None:
    """``/v1`` overlap dedup must apply to the custom-endpoint branch too."""
    assert _get_url(base_url, custom_endpoint) == expected


@pytest.mark.parametrize(
    ("base_url", "custom_endpoint"),
    [
        # Uppercase HTTP must be recognized as already-schemed; previously the
        # transport would prepend another http:// and produce
        # "http://HTTP://h:8000/...".
        param("HTTP://h:8000", "/v1/chat", id="uppercase-http"),
        param("HTTPS://h:8000", "/v1/chat", id="uppercase-https"),
        param("Http://h:8000", "/v1/chat", id="mixed-case"),
    ],
)  # fmt: skip
def test_build_url_uppercase_scheme(base_url: str, custom_endpoint: str) -> None:
    """Uppercase/mixed-case schemes must not be doubly-prefixed with http://."""
    url = _get_url(base_url, custom_endpoint)
    # No second scheme tacked on (the bug was http://HTTP://...).
    assert "://HTTP://" not in url
    assert "://Http://" not in url
    # Path must be appended once.
    assert url.endswith("/v1/chat")
    # Either preserves the input case OR normalizes to lowercase — both are
    # acceptable; the only thing forbidden is double-prefixing.
    lowered = url.lower()
    expected_https = base_url.lower().startswith("https")
    assert lowered.startswith("https://" if expected_https else "http://")


def test_build_url_empty_custom_endpoint() -> None:
    """An explicitly-empty ``custom_endpoint`` means "no path append" — not
    "fall through to metadata path". Previously the truthiness check treated
    empty-string the same as ``None``, surprising users who passed ``""`` to
    suppress any path append.
    """
    # Empty custom_endpoint with /v1 base: the base path stays as /v1.
    assert _get_url("http://h:8080/v1", "") == "http://h:8080/v1"
    # Empty custom_endpoint with bare host: nothing appended.
    assert _get_url("http://h:8080", "") == "http://h:8080"
    # Sanity: None still dispatches into the metadata branch.
    assert _get_url("http://h:8080/v1", None) == "http://h:8080/v1/chat/completions"


@pytest.mark.parametrize(
    ("url", "expected_fragment"),
    [
        # Garbage ports.
        param("http://h:abc", "invalid port", id="port-non-numeric"),
        param("http://h:99999", "invalid port", id="port-too-large"),
        param("http://h:-1", "invalid port", id="port-negative"),
        param("http://h:0", "outside the valid range", id="port-zero"),
        # Whitespace.
        param("  http://h:8000  ", "whitespace", id="leading-trailing-spaces"),
        param("http://h:8000\n", "whitespace", id="trailing-newline"),
        param("http://h:8000/path with space", "whitespace", id="space-in-path"),
        param("http://h\t:8000", "whitespace", id="internal-tab"),
        # Non-http(s) schemes.
        param("gopher://h:70", "unsupported scheme", id="gopher-scheme"),
        param("file:///etc/passwd", "missing scheme or host", id="file-scheme"),
        param("javascript:alert(1)", "missing scheme or host", id="javascript-scheme"),
    ],
)
def test_endpoint_validator_rejects_garbage(url: str, expected_fragment: str) -> None:
    """Validator must reject malformed URLs at config parse time."""
    with pytest.raises(ValidationError) as exc_info:
        EndpointConfig(urls=[url])
    assert expected_fragment in str(exc_info.value)
