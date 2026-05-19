# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared Playwright + uvicorn harness for dashboard-v2 tests."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from starlette.websockets import WebSocket as StarletteWebSocket

from aiperf.api.routers.core import core_router
from aiperf.api.routers.static import static_router
from aiperf.config import AIPerfConfig, BenchmarkRun
from tests.unit.api.dashboard_v2_e2e.helpers import (
    dashboard_cfg,
    server_metrics_response,
)

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Page


RestOverrideBody = dict[str, Any] | list[Any] | str | Response
RestOverride = tuple[int, RestOverrideBody]


@dataclass(slots=True)
class DashboardScenario:
    """Server-side state used by one dashboard-v2 browser scenario."""

    cfg: AIPerfConfig = field(default_factory=dashboard_cfg)
    progress: dict[str, Any] = field(default_factory=lambda: {"phases": {}})
    server_metrics: dict[str, Any] = field(default_factory=server_metrics_response)
    ws_payloads: list[dict[str, Any] | str] = field(default_factory=list)
    rest_overrides: dict[str, RestOverride] = field(default_factory=dict)
    close_ws_after_payloads: bool = False


@dataclass(slots=True)
class _StubAPIService:
    """Surface required by ``core_router`` for ``/api/config``."""

    run: BenchmarkRun
    app: FastAPI | None = None

    def is_healthy(self) -> bool:
        return True

    def is_ready(self) -> bool:
        return True


@dataclass(slots=True)
class _ServerHandle:
    base_url: str
    server: uvicorn.Server
    thread: threading.Thread

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5.0)
        assert not self.thread.is_alive(), (
            f"dashboard-v2 uvicorn thread failed to stop for {self.base_url}"
        )


@dataclass(slots=True)
class DashboardHarness:
    """Per-test dashboard-v2 browser and server handle."""

    page: Page
    console_errors: list[str]
    bad_responses: list[str]
    servers: list[_ServerHandle]

    def start_server(self, scenario: DashboardScenario | None = None) -> str:
        """Start a threaded uvicorn server and return its base URL."""
        app = _build_app(scenario or DashboardScenario())
        sock = _bound_server_socket()
        port = sock.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=port,
                log_level="warning",
                access_log=False,
            )
        )
        thread = threading.Thread(
            target=lambda: server.run(sockets=[sock]), daemon=True
        )
        thread.start()
        deadline = time.monotonic() + 10.0
        while not getattr(server, "started", False):
            if time.monotonic() > deadline:
                server.should_exit = True
                thread.join(timeout=5.0)
                sock.close()
                raise RuntimeError("dashboard-v2 uvicorn did not start within 10 s")
            time.sleep(0.02)
        handle = _ServerHandle(
            base_url=f"http://127.0.0.1:{port}", server=server, thread=thread
        )
        self.servers.append(handle)
        return handle.base_url

    def goto_dashboard(self, scenario: DashboardScenario | None = None) -> Page:
        """Navigate to ``/dashboard-v2`` against a fresh scenario server."""
        base_url = self.start_server(scenario)
        self.page.goto(f"{base_url}/dashboard-v2", wait_until="domcontentloaded")
        return self.page

    def wait_for_boot(self) -> None:
        """Wait until the SPA is connected and its config bar has rendered."""
        self.page.wait_for_selector(".status-dot.connected", timeout=10_000)
        self.page.wait_for_selector("#config-bar.visible", timeout=10_000)

    def assert_no_console_errors(self) -> None:
        """Fail if the browser logged console errors or page exceptions."""
        assert not self.console_errors, "Unexpected console errors:\n  " + "\n  ".join(
            self.console_errors
        )

    def assert_no_bad_responses(self) -> None:
        """Fail if the dashboard observed an HTTP response with status >= 400."""
        assert not self.bad_responses, "Unexpected bad responses:\n  " + "\n  ".join(
            self.bad_responses
        )


def dashboard_harness_for_browser(browser: Browser) -> Iterator[DashboardHarness]:
    """Yield one dashboard harness and always close its Playwright context."""
    context = browser.new_context(viewport={"width": 1600, "height": 1000})
    page = context.new_page()
    console_errors: list[str] = []
    bad_responses: list[str] = []

    page.on(
        "console",
        lambda msg: console_errors.append(f"[{msg.type}] {msg.text}")
        if msg.type in ("error",)
        else None,
    )
    page.on("pageerror", lambda exc: console_errors.append(f"[pageerror] {exc}"))

    def _on_response(response: Any) -> None:
        if response.status >= 400:
            bad_responses.append(
                f"{response.status} {response.request.method} {response.url}"
            )

    page.on("response", _on_response)
    harness = DashboardHarness(
        page=page,
        console_errors=console_errors,
        bad_responses=bad_responses,
        servers=[],
    )
    try:
        yield harness
    finally:
        server_cleanup_error: BaseException | None = None
        try:
            for server in reversed(harness.servers):
                try:
                    server.stop()
                except BaseException as exc:
                    if server_cleanup_error is None:
                        server_cleanup_error = exc
        finally:
            try:
                context.close()
            finally:
                if server_cleanup_error is not None:
                    raise server_cleanup_error


def playwright_ready() -> tuple[bool, str]:
    """Return (available, reason). Available means Chromium can launch."""
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except ImportError:
        return (
            False,
            "playwright not installed (`uv pip install playwright pytest-playwright`)",
        )
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
    except Exception as exc:  # noqa: BLE001 - skip reason should surface raw launch detail
        return (
            False,
            f"Chromium not launchable: {exc!s}. Run `uv run playwright install chromium`.",
        )
    return True, ""


def _bound_server_socket() -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    return sock


def _response_from_override(status: int, body: RestOverrideBody) -> Response:
    if isinstance(body, Response):
        body.status_code = status
        return body
    if isinstance(body, str):
        return PlainTextResponse(body, status_code=status)
    return JSONResponse(body, status_code=status)


def _build_app(scenario: DashboardScenario) -> FastAPI:
    app = FastAPI(title="aiperf-dashboard-v2-e2e")
    run = BenchmarkRun(
        benchmark_id="dashboard-v2-e2e",
        cfg=scenario.cfg.benchmark,
        artifact_dir=Path("/tmp/aiperf-dashboard-v2-e2e"),
    )
    svc = _StubAPIService(run=run)
    svc.app = app
    app.state.service = svc

    @app.middleware("http")
    async def _rest_overrides(request: Request, call_next: Any) -> Response:
        override = scenario.rest_overrides.get(request.url.path)
        if override is not None:
            status, body = override
            return _response_from_override(status, body)
        return await call_next(request)

    app.include_router(static_router)
    app.include_router(core_router)

    @app.get("/api/progress")
    async def _progress() -> dict[str, Any]:
        return scenario.progress

    @app.get("/api/server-metrics")
    async def _server_metrics() -> dict[str, Any]:
        return scenario.server_metrics

    @app.websocket("/ws")
    async def _ws(websocket: StarletteWebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    parsed = orjson.loads(raw)
                except Exception:  # noqa: BLE001 - malformed subscribe payload is ignored
                    parsed = {}
                await websocket.send_text('{"type": "subscribed", "message_types": []}')
                if parsed.get("type") != "subscribe":
                    continue
                for payload in scenario.ws_payloads:
                    if isinstance(payload, str):
                        await websocket.send_text(payload)
                    else:
                        await websocket.send_text(orjson.dumps(payload).decode())
                if scenario.close_ws_after_payloads:
                    await websocket.close()
                    return
        except Exception:  # noqa: BLE001 - browser disconnect is normal in tests
            return

    @app.get("/metrics", response_class=PlainTextResponse)
    async def _metrics() -> str:
        return "# stub\n"

    return app


__all__ = [
    "DashboardHarness",
    "DashboardScenario",
    "RestOverride",
    "RestOverrideBody",
    "dashboard_harness_for_browser",
    "playwright_ready",
]
