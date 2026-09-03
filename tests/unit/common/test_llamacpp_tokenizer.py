# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the llama.cpp HTTP tokenizer adapter."""

from __future__ import annotations

import httpx
import pytest

from aiperf.common.exceptions import TokenizerError
from aiperf.common.llamacpp_tokenizer import (
    LlamaCppTokenizerAdapter,
    normalize_llamacpp_base_url,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("http://127.0.0.1:8081", "http://127.0.0.1:8081"),
        (
            "http://127.0.0.1:8081/v1/chat/completions",
            "http://127.0.0.1:8081",
        ),
        ("https://example.test/prefix/tokenize", "https://example.test/prefix"),
        ("HTTP://EXAMPLE.TEST:8080/", "http://EXAMPLE.TEST:8080"),
    ],
)
def test_normalize_llamacpp_base_url_known_paths(value: str, expected: str) -> None:
    assert normalize_llamacpp_base_url(value) == expected


@pytest.mark.parametrize("value", ["qwen3.5", "/tmp/model", "ftp://example.test"])
def test_normalize_llamacpp_base_url_invalid_url_raises(value: str) -> None:
    with pytest.raises(ValueError, match=r"absolute http\(s\) URL"):
        normalize_llamacpp_base_url(value)


def test_adapter_probes_and_tokenizes_with_llamacpp_contract() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/prefix/tokenize":
            return httpx.Response(200, json={"tokens": [1, 2]})
        if request.url.path == "/prefix/detokenize":
            return httpx.Response(200, json={"content": "AIPerf tokenizer probe"})
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = LlamaCppTokenizerAdapter("http://llama.test/prefix", client=client)

    assert adapter.encode("hello") == [1, 2]
    assert adapter.decode([1, 2]) == "AIPerf tokenizer probe"
    assert [request.url.path for request in requests] == [
        "/prefix/tokenize",
        "/prefix/detokenize",
        "/prefix/tokenize",
        "/prefix/detokenize",
    ]
    tokenize_payload = requests[2].read().decode()
    assert '"content":"hello"' in tokenize_payload
    assert '"add_special":false' in tokenize_payload
    assert '"with_pieces":false' in tokenize_payload


def test_adapter_incompatible_endpoint_raises_tokenizer_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    client = httpx.Client(transport=httpx.MockTransport(handler))

    with pytest.raises(TokenizerError, match="/tokenize.*404"):
        LlamaCppTokenizerAdapter("http://not-llama.test", client=client)


def test_adapter_invalid_tokens_shape_raises_tokenizer_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"tokens": [True]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = LlamaCppTokenizerAdapter(
        "http://llama.test", client=client, validate=False
    )

    with pytest.raises(TokenizerError, match="invalid 'tokens'"):
        adapter.encode("hello")
