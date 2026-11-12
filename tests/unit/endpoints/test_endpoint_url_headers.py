# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Requests carry the headers of the URL they are sent to."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.enums import ModelSelectionStrategy
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.endpoints.anthropic_messages import MessagesEndpoint
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_request_info,
)

DP_URL = "http://localhost:8000"


def _model_endpoint(
    endpoint_type: EndpointType = EndpointType.CHAT,
    api_key: str | None = None,
) -> ModelEndpointInfo:
    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="test-model")],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(
            type=endpoint_type,
            base_urls=[DP_URL, DP_URL, DP_URL],
            headers=[("X-Team", "perf"), ("x-data-parallel-rank", "global")],
            url_headers=[
                {"X-data-parallel-rank": "0"},
                {},
                {"X-data-parallel-rank": "2", "X-Pool": "b"},
            ],
            api_key=api_key,
        ),
    )


@pytest.mark.parametrize(
    ("url_index", "expected"),
    [
        param(None, {"X-Team": "perf", "X-data-parallel-rank": "0"}, id="unset-index-uses-first-url"),
        param(0, {"X-Team": "perf", "X-data-parallel-rank": "0"}, id="url-0"),
        param(1, {"X-Team": "perf", "x-data-parallel-rank": "global"}, id="url-without-overrides"),
        param(2, {"X-Team": "perf", "X-data-parallel-rank": "2", "X-Pool": "b"}, id="url-2"),
        param(3, {"X-Team": "perf", "X-data-parallel-rank": "0"}, id="index-wraps-like-get-url"),
    ],
)  # fmt: skip
def test_get_headers_overlays_url_headers_case_insensitively(
    url_index: int | None, expected: dict[str, str]
) -> None:
    assert _model_endpoint().endpoint.get_headers(url_index) == expected


def test_get_headers_without_url_headers_returns_global_headers() -> None:
    info = EndpointInfo(base_urls=[DP_URL, DP_URL], headers=[("X-Team", "perf")])

    assert info.get_headers(0) == info.get_headers(1) == {"X-Team": "perf"}


@pytest.mark.parametrize("url_index", [0, 1, 2])
def test_chat_endpoint_headers_follow_request_url_index(url_index: int) -> None:
    model_endpoint = _model_endpoint(api_key="sk-test")
    endpoint = create_endpoint_with_mock_transport(ChatEndpoint, model_endpoint)
    request_info = create_request_info(model_endpoint=model_endpoint)
    request_info.url_index = url_index

    headers = endpoint.get_endpoint_headers(request_info)

    assert headers == {
        **model_endpoint.endpoint.get_headers(url_index),
        "Authorization": "Bearer sk-test",
    }


def test_messages_endpoint_headers_follow_request_url_index() -> None:
    model_endpoint = _model_endpoint(EndpointType.MESSAGES, api_key="sk-ant")
    endpoint = create_endpoint_with_mock_transport(MessagesEndpoint, model_endpoint)
    request_info = create_request_info(model_endpoint=model_endpoint)
    request_info.url_index = 2

    headers = endpoint.get_endpoint_headers(request_info)

    assert headers["X-data-parallel-rank"] == "2"
    assert headers["X-Pool"] == "b"
    assert "x-data-parallel-rank" not in headers
    assert headers["x-api-key"] == "sk-ant"
    assert headers["anthropic-version"] == "2023-06-01"
