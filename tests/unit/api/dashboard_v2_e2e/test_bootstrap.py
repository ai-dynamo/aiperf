# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bootstrap smoke coverage for the dashboard-v2 Playwright harness."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from playwright.sync_api import expect

from tests.unit.api.dashboard_v2_e2e import harness as harness_module
from tests.unit.api.dashboard_v2_e2e.harness import (
    DashboardHarness,
    dashboard_harness_for_browser,
)
from tests.unit.api.dashboard_v2_e2e.helpers import dashboard_cfg


class _InlineThread:
    def __init__(self, target: Callable[[], None], *, daemon: bool) -> None:
        self._target = target
        self._alive = False

    def start(self) -> None:
        self._target()

    def join(self, timeout: float | None = None) -> None:
        pass

    def is_alive(self) -> bool:
        return self._alive


class _FakePage:
    def on(self, event: str, callback: Callable[..., None]) -> None:
        pass


class _FakeContext:
    def __init__(self) -> None:
        self.closed = False

    def new_page(self) -> _FakePage:
        return _FakePage()

    def close(self) -> None:
        self.closed = True


class _FakeBrowser:
    def __init__(self) -> None:
        self.context = _FakeContext()

    def new_context(self, *, viewport: dict[str, int]) -> _FakeContext:
        return self.context


class _FailingServerHandle:
    def stop(self) -> None:
        raise RuntimeError("server cleanup failed")


def test_start_server_passes_bound_socket_to_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid xdist races by keeping the allocated socket through uvicorn bind."""
    captured: dict[str, Any] = {}

    class _FakeServer:
        started = True
        should_exit = False

        def __init__(self, config: Any) -> None:
            captured["config"] = config

        def run(self, sockets: list[Any] | None = None) -> None:
            captured["sockets"] = sockets

    monkeypatch.setattr(harness_module.uvicorn, "Server", _FakeServer)
    monkeypatch.setattr(harness_module.threading, "Thread", _InlineThread)

    harness = DashboardHarness(
        page=object(),
        console_errors=[],
        bad_responses=[],
        servers=[],  # type: ignore[arg-type]
    )

    sockets: list[Any] | None = None
    try:
        base_url = harness.start_server()

        sockets = captured["sockets"]
        assert sockets and len(sockets) == 1
        assert base_url == f"http://127.0.0.1:{sockets[0].getsockname()[1]}"
    finally:
        for sock in sockets or captured.get("sockets") or []:
            sock.close()


def test_dashboard_fixture_closes_context_when_server_stop_fails() -> None:
    """Browser context cleanup must not be skipped by server cleanup failures."""
    browser = _FakeBrowser()
    fixture = dashboard_harness_for_browser(browser)
    harness = next(fixture)
    harness.servers.append(_FailingServerHandle())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="server cleanup failed"):
        next(fixture)

    assert browser.context.closed


def test_dashboard_v2_bootstrap_smoke(dashboard: DashboardHarness) -> None:
    """The v2 dashboard boots, connects, renders config, and hides secrets."""
    dashboard.goto_dashboard()
    dashboard.wait_for_boot()

    title = dashboard.page.title()
    body_text = dashboard.page.locator("body").inner_text()
    config_text = dashboard.page.locator("#config-bar").text_content() or ""
    combined = "\n".join([title, body_text, config_text])

    for token in (
        "AIPerf Dashboard",
        "Connected",
        "llama3-8b",
        "llama3-70b",
        "chat (streaming)",
    ):
        assert token in combined
    assert "SHOULD_NOT_LEAK" not in combined

    dashboard.assert_no_console_errors()
    dashboard.assert_no_bad_responses()


def test_dashboard_v2_loads_all_static_assets(dashboard: DashboardHarness) -> None:
    """The browser can load the SPA shell and same-origin static assets."""
    dashboard.goto_dashboard()
    dashboard.wait_for_boot()

    assert dashboard.page.locator(".topbar").is_visible()
    expect(dashboard.page.get_by_text("AIPerf Dashboard").first).to_be_visible()
    bad_dashboard_assets = [
        response
        for response in dashboard.bad_responses
        if "/dashboard-v2/" in response or response.endswith("/dashboard-v2")
    ]
    assert not bad_dashboard_assets
    dashboard.assert_no_console_errors()
    dashboard.assert_no_bad_responses()


def test_browser_never_receives_api_key_in_config(dashboard: DashboardHarness) -> None:
    """The config payload exposed to the browser must redact endpoint secrets."""
    config_payloads: list[dict[str, Any]] = []

    def _capture_config_response(response: Any) -> None:
        if response.url.endswith("/api/config") and response.status == 200:
            config_payloads.append(response.json())

    dashboard.page.on("response", _capture_config_response)
    dashboard.goto_dashboard()
    dashboard.wait_for_boot()

    assert config_payloads
    endpoint = config_payloads[-1]["endpoint"]
    assert "api_key" not in endpoint
    assert endpoint["urls"] == dashboard_cfg().benchmark.endpoint.urls
    assert endpoint["type"] == dashboard_cfg().benchmark.endpoint.type
    assert endpoint["streaming"] is dashboard_cfg().benchmark.endpoint.streaming
    dashboard.assert_no_console_errors()
    dashboard.assert_no_bad_responses()
