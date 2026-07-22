# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Readiness-probe auth-header selection per endpoint type."""

from __future__ import annotations

from aiperf.cli_runner._preflight import _readiness_auth_headers
from aiperf.config import EndpointConfig


def test_messages_headers_use_x_api_key_and_version() -> None:
    """Messages endpoints must probe with x-api-key + anthropic-version.

    A Bearer header or a missing anthropic-version makes an Anthropic server
    return 4xx, which the probe's "status < 500 == ready" rule would misread
    as ready.
    """
    headers = _readiness_auth_headers(
        EndpointConfig(type="messages", urls=["http://server"], api_key="sk-ant-test")
    )

    assert headers["x-api-key"] == "sk-ant-test"
    assert headers["anthropic-version"]
    assert "Authorization" not in headers


def test_messages_api_key_overrides_preconfigured_x_api_key() -> None:
    """--api-key must override a preconfigured x-api-key on the probe.

    MessagesEndpoint.get_endpoint_headers() hard-assigns x-api-key from
    api_key, so real requests use the CLI key. Readiness must probe the same
    key, otherwise preflight validates a different credential than the run.
    """
    headers = _readiness_auth_headers(
        EndpointConfig(
            type="messages",
            urls=["http://server"],
            api_key="sk-ant-cli",
            headers={"x-api-key": "sk-ant-preconfigured"},
        )
    )

    assert headers["x-api-key"] == "sk-ant-cli"


def test_messages_headers_set_version_without_api_key() -> None:
    """anthropic-version is required even when no api_key is configured."""
    headers = _readiness_auth_headers(
        EndpointConfig(type="messages", urls=["http://server"])
    )

    assert headers["anthropic-version"]
    assert "x-api-key" not in headers


def test_chat_headers_use_bearer() -> None:
    """OpenAI-compatible endpoints keep Authorization: Bearer."""
    headers = _readiness_auth_headers(
        EndpointConfig(type="chat", urls=["http://server"], api_key="sk-test")
    )

    assert headers["Authorization"] == "Bearer sk-test"
    assert "x-api-key" not in headers
