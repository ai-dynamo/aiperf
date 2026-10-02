# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the dashboard-refresh fire-and-forget call from try_claim_completion."""

from __future__ import annotations

import importlib

import aiohttp
import pytest


class _FakeResponse:
    """Minimal aiohttp response usable as an async context manager."""

    status = 202

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def read(self) -> bytes:
        return b""


@pytest.fixture
def reload_env_with_dashboard_port(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIPERF_DASHBOARD_PORT", "8082")
    monkeypatch.setenv("AIPERF_DASHBOARD_PROXY_ENABLED", "1")
    from aiperf.operator import environment as env_mod

    importlib.reload(env_mod)
    yield env_mod


@pytest.mark.asyncio
async def test_post_dashboard_refresh_succeeds(
    monkeypatch: pytest.MonkeyPatch, reload_env_with_dashboard_port
) -> None:
    """When PORT is set, _post_dashboard_refresh hits localhost:<port>/admin/refresh."""
    from aiperf.operator import client_cache

    seen: dict[str, str] = {}

    def _fake_post(self, url, **_kwargs):
        seen["url"] = url
        return _FakeResponse()

    monkeypatch.setattr(aiohttp.ClientSession, "post", _fake_post)

    await client_cache._post_dashboard_refresh()
    assert seen["url"] == "http://localhost:8082/admin/refresh"


@pytest.mark.asyncio
async def test_post_dashboard_refresh_swallows_errors(
    monkeypatch: pytest.MonkeyPatch, reload_env_with_dashboard_port
) -> None:
    """aiohttp errors must not propagate out of the helper."""
    from aiperf.operator import client_cache

    def _broken_post(self, *_args, **_kwargs):
        raise aiohttp.ClientConnectionError("boom")

    monkeypatch.setattr(aiohttp.ClientSession, "post", _broken_post)

    # Must not raise.
    await client_cache._post_dashboard_refresh()


@pytest.mark.asyncio
async def test_post_dashboard_refresh_skipped_when_port_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AIPERF_DASHBOARD_PORT", raising=False)
    monkeypatch.delenv("AIPERF_DASHBOARD_PROXY_ENABLED", raising=False)

    from aiperf.operator import environment as env_mod

    importlib.reload(env_mod)

    from aiperf.operator import client_cache

    posted = False

    def _fake_post(self, *_args, **_kwargs):
        nonlocal posted
        posted = True
        return _FakeResponse()

    monkeypatch.setattr(aiohttp.ClientSession, "post", _fake_post)
    await client_cache._post_dashboard_refresh()
    assert posted is False
