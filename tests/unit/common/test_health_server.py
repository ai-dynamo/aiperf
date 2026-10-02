# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for direct HealthServer configuration."""

from types import SimpleNamespace

import pytest

from aiperf.common.health_server import HealthServer


def test_health_server_uses_configured_host_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An omitted host resolves from the shared service settings at construction."""
    import aiperf.common.health_server as health_server

    monkeypatch.setattr(
        health_server,
        "Environment",
        SimpleNamespace(SERVICE=SimpleNamespace(HEALTH_HOST="192.0.2.10")),
    )

    server = HealthServer(port=8080)

    assert server._host == "192.0.2.10"


def test_health_server_explicit_host_overrides_configured_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit host remains authoritative for callers needing an override."""
    import aiperf.common.health_server as health_server

    monkeypatch.setattr(
        health_server,
        "Environment",
        SimpleNamespace(SERVICE=SimpleNamespace(HEALTH_HOST="192.0.2.10")),
    )

    server = HealthServer(port=8080, host="127.0.0.1")

    assert server._host == "127.0.0.1"
