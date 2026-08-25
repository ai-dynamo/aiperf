# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the AIPerf API dashboard (``dashboard.html``).

Tests are layered from cheapest to heaviest:

* ``test_inline_js_parses`` - ``node --check`` on the extracted inline script.
  No DOM, no browser; catches syntax regressions quickly.
* ``test_renderConfig_populates_config_bar`` - Playwright drives a real
  Chromium page against a live uvicorn serving the real ``/dashboard`` route.
  Asserts that ``renderConfig`` wrote the expected tokens into
  ``#config-bar`` for both multi-phase and single-phase configs, that the
  WebSocket handshake completes, and that the ``api_key`` is not leaked.

Both layers skip gracefully when their runtime is missing:

* ``node`` not on PATH -> ``test_inline_js_parses`` skips.
* ``playwright`` not installed in the venv, or Chromium not downloaded
  (``uv run playwright install chromium``) -> browser tests skip.
"""

from __future__ import annotations

import contextlib
import json
import re
import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient
from pytest import param
from starlette.websockets import WebSocket as StarletteWebSocket

from aiperf.api.routers.core import core_router
from aiperf.api.routers.static import static_router
from aiperf.config import AIPerfConfig, BenchmarkRun

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Page

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DASHBOARD_HTML = _REPO_ROOT / "src" / "aiperf" / "api" / "static" / "dashboard.html"


# -----------------------------------------------------------------------------
# Runtime availability
# -----------------------------------------------------------------------------


def _node_binary() -> str | None:
    return shutil.which("node")


def _playwright_ready() -> tuple[bool, str]:
    """Return (available, reason). Available means Chromium can be launched."""
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except ImportError:
        return (
            False,
            "playwright not installed (`uv pip install playwright pytest-playwright`)",
        )
    # Check Chromium binary: the launch call is the authoritative test, but
    # failing fast here avoids a cryptic stacktrace if the browser is missing.
    #
    # NOTE: ``sync_playwright()`` spins up (and tears down) an event loop on
    # the calling thread. This probe runs at module import, so the loop churn
    # lands before any test body -- keep it out of collection-time paths that
    # pytest-asyncio shares with unrelated async tests on the same worker.
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
    except Exception as exc:  # noqa: BLE001 - one-shot probe, message surfaces via skip reason
        return (
            False,
            f"Chromium not launchable: {exc!s}. Run `uv run playwright install chromium`.",
        )
    return (True, "")


_NODE_REASON = "node binary not on PATH"
_PLAYWRIGHT_AVAILABLE, _PLAYWRIGHT_REASON = _playwright_ready()


# -----------------------------------------------------------------------------
# Inline-script helpers
# -----------------------------------------------------------------------------


def _extract_inline_js(html: str) -> str:
    """Return the content of the single inline ``<script>...</script>`` block."""
    match = re.search(
        r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>",
        html,
        re.DOTALL | re.IGNORECASE,
    )
    assert match is not None, "dashboard.html must contain exactly one inline <script>"
    return match.group(1)


# -----------------------------------------------------------------------------
# Minimal live FastAPI app for the dashboard to fetch against
# -----------------------------------------------------------------------------


class _StubAPIService:
    """Just enough surface for ``core.get_config`` (``svc.run.cfg``)."""

    def __init__(self, run: BenchmarkRun) -> None:
        self.run = run
        self.app: FastAPI  # filled in by ``_build_app``

    def is_healthy(self) -> bool:
        return True

    def is_ready(self) -> bool:
        return True


def _build_app(
    cfg: AIPerfConfig,
    broadcast_phases: bool = False,
    extra_ws_payloads: list[dict[str, Any]] | None = None,
) -> FastAPI:
    app = FastAPI(title="aiperf-dashboard-test")
    run = BenchmarkRun(
        benchmark_id="dashboard-test",
        cfg=cfg.benchmark,
        artifact_dir=Path("/tmp/aiperf-dashboard-test"),
    )
    svc = _StubAPIService(run)
    svc.app = app
    app.state.service = svc

    app.include_router(static_router)  # GET /, GET /dashboard, GET /dashboard-v2
    app.include_router(core_router)  # GET /api/config + health

    @app.get("/api/progress")
    async def _progress() -> dict[str, Any]:
        return {"phases": {}}

    @app.get("/api/server-metrics")
    async def _server_metrics() -> dict[str, Any]:
        return {"endpoint_summaries": []}

    # Pre-computed phase announcements (for the v2 tests that need to see the
    # PhaseCards component render an entry per configured phase name).
    phase_names = [p.name for p in cfg.benchmark.phases]
    ws_payloads = list(extra_ws_payloads or [])

    # NOTE: FastAPI's ``@app.websocket`` rejects the upgrade (HTTP 403) unless
    # the ``websocket`` parameter is annotated - the type hint is what drives
    # dependency resolution for the WS route. Without it the handler never
    # runs and uvicorn sends a synthetic close -> 403.
    @app.websocket("/ws")
    async def _ws(websocket: StarletteWebSocket) -> None:
        import json

        await websocket.accept()
        try:
            while True:
                raw = await websocket.receive_text()
                await websocket.send_text('{"type": "subscribed", "message_types": []}')
                try:
                    parsed = json.loads(raw)
                except Exception:  # noqa: BLE001
                    parsed = {}
                if parsed.get("type") != "subscribe":
                    continue

                if broadcast_phases:
                    for name in phase_names:
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "credit_phase_start",
                                    "phase": name,
                                    "stats": {
                                        "start_ns": 1,
                                        "total_expected_requests": 100,
                                    },
                                }
                            )
                        )

                for payload in ws_payloads:
                    await websocket.send_text(json.dumps(payload))
        except Exception:  # noqa: BLE001 - test stub; client disconnect is normal
            return

    @app.get("/metrics", response_class=PlainTextResponse)
    async def _metrics() -> str:
        return "# stub\n"

    return app


def _build_multi_phase_cfg() -> AIPerfConfig:
    return AIPerfConfig(
        benchmark={
            "models": ["llama3-8b", "llama3-70b"],
            "endpoint": {
                "urls": ["http://srv:8000/v1/chat/completions"],
                "type": "chat",
                "streaming": True,
                "api_key": "SHOULD_NOT_LEAK",
            },
            "datasets": [
                {
                    "name": "default",
                    "type": "synthetic",
                    "entries": 100,
                    "prompts": {"isl": 128, "osl": 64},
                }
            ],
            "phases": [
                {
                    "name": "warmup",
                    "type": "concurrency",
                    "requests": 50,
                    "concurrency": 4,
                },
                {
                    "name": "profiling",
                    "type": "poisson",
                    "rate": 20,
                    "duration": 300,
                    "concurrency": 32,
                },
            ],
            "runtime": {"api_port": 8080},
        }
    )


def _build_single_phase_cfg() -> AIPerfConfig:
    return AIPerfConfig(
        benchmark={
            "models": ["gpt-4o-mini"],
            "endpoint": {
                "urls": ["http://srv:8000/v1/chat/completions"],
                "type": "chat",
            },
            "datasets": [
                {
                    "name": "default",
                    "type": "synthetic",
                    "entries": 50,
                    "prompts": {"isl": 128, "osl": 32},
                }
            ],
            "phases": [
                {
                    "name": "default",
                    "type": "concurrency",
                    "kind": "profiling",
                    "requests": 100,
                    "concurrency": 8,
                }
            ],
            "runtime": {"api_port": 8080},
        }
    )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _run_server(
    cfg: AIPerfConfig,
    broadcast_phases: bool = False,
    extra_ws_payloads: list[dict[str, Any]] | None = None,
) -> Iterator[str]:
    """Boot uvicorn on a free port in a background thread; yield the base URL."""
    app = _build_app(
        cfg,
        broadcast_phases=broadcast_phases,
        extra_ws_payloads=extra_ws_payloads,
    )
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # Wait for readiness: uvicorn sets ``started`` once serve() is past startup.
    deadline = time.monotonic() + 10.0
    while not getattr(server, "started", False):
        if time.monotonic() > deadline:
            server.should_exit = True
            raise RuntimeError("uvicorn did not start within 10 s")
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


# -----------------------------------------------------------------------------
# Playwright fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _browser() -> Iterator[Browser]:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            # Teardown fires at session end, by which time pytest-asyncio's
            # event loop is gone. Playwright's sync `browser.close()` routes
            # through the loop-backed connection and raises
            # `RuntimeError: Browser.close: no running event loop`. The
            # `sync_playwright` context manager on exit still reaps the
            # Chromium subprocess cleanly, so swallow just that specific
            # teardown RuntimeError rather than reporting a spurious pytest
            # error under whichever unrelated test ran last.
            try:
                browser.close()
            except RuntimeError as exc:
                if "no running event loop" not in str(exc):
                    raise


@pytest.fixture
def _page(_browser: Browser) -> Iterator[Page]:
    context = _browser.new_context()
    page = context.new_page()
    try:
        yield page
    finally:
        context.close()


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


class TestDashboardInlineJS:
    """Inline-JS checks that don't need a DOM."""

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_inline_js_parses(self, tmp_path: Path) -> None:
        """``node --check`` on the extracted inline script."""
        html = _DASHBOARD_HTML.read_text()
        js = _extract_inline_js(html)
        js_path = tmp_path / "dashboard_inline.js"
        js_path.write_text(js)
        proc = subprocess.run(
            [_node_binary(), "--check", str(js_path)],
            capture_output=True,
            timeout=15,
        )
        assert proc.returncode == 0, (
            f"inline JS failed `node --check`:\n{proc.stderr.decode(errors='replace')}"
        )


class TestDashboardRenderConfig:
    """Drive a real Chromium browser against a live uvicorn serving dashboard.html.

    Skips when Playwright (or the Chromium download) is not available.
    """

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    @pytest.mark.parametrize(
        ("cfg_builder", "must_contain", "must_not_contain"),
        [
            param(
                _build_multi_phase_cfg,
                [
                    "Model",
                    "llama3-8b",
                    "llama3-70b",
                    "Endpoint",
                    "chat (streaming)",
                    "URL",
                    "http://srv:8000/v1/chat/completions",
                    "warmup Type",
                    "concurrency",
                    "warmup Concurrency",
                    "warmup Requests",
                    "50",
                    "profiling Type",
                    "poisson",
                    "profiling Rate",
                    "20 QPS",
                    "profiling Duration",
                    "5m 0s",
                    "profiling Concurrency",
                    "32",
                ],
                ["SHOULD_NOT_LEAK"],
                id="multi-phase",
            ),
            param(
                _build_single_phase_cfg,
                [
                    "Model",
                    "gpt-4o-mini",
                    "Endpoint",
                    "chat",
                    "URL",
                    "http://srv:8000/v1/chat/completions",
                    "Type",
                    "concurrency",
                    "Concurrency",
                    "8",
                    "Requests",
                    "100",
                ],
                # Single-phase: the phase-name prefix must not appear.
                ["default Type", "default Concurrency", "default Requests"],
                id="single-phase",
            ),
        ],
    )  # fmt: skip
    def test_renderConfig_populates_config_bar(
        self,
        _page: Page,
        cfg_builder: Any,
        must_contain: list[str],
        must_not_contain: list[str],
    ) -> None:
        """renderConfig must emit the right label text end-to-end through a real browser."""
        console_errors: list[str] = []
        _page.on(
            "console",
            lambda msg: console_errors.append(msg.text)
            if msg.type in ("error", "warning")
            else None,
        )

        with _run_server(cfg_builder()) as base_url:
            _page.goto(f"{base_url}/dashboard", wait_until="networkidle")

            # ``.visible`` is added the moment renderConfig finishes a successful dump.
            _page.wait_for_selector("#config-bar.visible", timeout=10_000)

            # And the WebSocket handshake should flip the status badge.
            _page.wait_for_function(
                """() => {
                    const s = document.getElementById('status');
                    return s && s.classList.contains('connected');
                }""",
                timeout=10_000,
            )

            # ``inner_text`` returns the CSS-rendered text (labels are
            # uppercased via ``text-transform``); use ``text_content`` to see
            # the source strings the script wrote into the DOM.
            text = _page.locator("#config-bar").text_content() or ""
            status_text = _page.locator("#status").text_content() or ""
            log_text = _page.locator("#log").text_content() or ""

        assert console_errors == [], (
            "unexpected browser console errors:\n  " + "\n  ".join(console_errors)
        )

        missing = [t for t in must_contain if t not in text]
        assert not missing, (
            f"renderConfig output missing tokens: {missing}\nactual text: {text!r}"
        )

        for forbidden in must_not_contain:
            assert forbidden not in text, (
                f"renderConfig output unexpectedly contains {forbidden!r}\ntext: {text!r}"
            )

        assert "Connected" in status_text, (
            f"#status did not show Connected; text={status_text!r}"
        )
        assert "Connected" in log_text, (
            f"log did not record Connected; text={log_text!r}"
        )

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_api_config_does_not_leak_api_key(self, _page: Page) -> None:
        """End-to-end: the browser should never see the api_key field on /api/config."""
        captured: list[dict[str, Any]] = []

        def on_response(response: Any) -> None:
            if response.url.endswith("/api/config"):
                try:
                    captured.append(response.json())
                except Exception:  # noqa: BLE001
                    captured.append({"__raw__": response.text()})

        _page.on("response", on_response)

        with _run_server(_build_multi_phase_cfg()) as base_url:
            _page.goto(f"{base_url}/dashboard", wait_until="networkidle")
            _page.wait_for_selector("#config-bar.visible", timeout=10_000)

        assert captured, "/api/config response was not captured"
        body = captured[-1]
        assert "endpoint" in body
        assert "api_key" not in body["endpoint"], (
            f"api_key must be excluded from /api/config; got endpoint={body['endpoint']!r}"
        )


# -----------------------------------------------------------------------------
# v2 dashboard (src/aiperf/api/static-v2/) - Preact/htm/signals stack
# -----------------------------------------------------------------------------

_STATIC_V2_DIR = _REPO_ROOT / "src" / "aiperf" / "api" / "static-v2"


def _v2_js_files() -> list[Path]:
    """All ES modules shipped by the v2 dashboard."""
    return sorted(_STATIC_V2_DIR.rglob("*.js"))


def _run_v2_node_script(tmp_path: Path, script: str) -> dict[str, Any]:
    """Run a dashboard-v2 ES-module script with tiny browser-dependency stubs."""
    node = _node_binary()
    assert node is not None, _NODE_REASON

    sandbox = tmp_path / "dashboard-v2-node"
    shutil.copytree(_STATIC_V2_DIR, sandbox)
    (sandbox / "package.json").write_text('{"type":"module"}\n')

    signals_dir = sandbox / "node_modules" / "@preact" / "signals"
    signals_dir.mkdir(parents=True)
    (signals_dir / "package.json").write_text('{"type":"module","main":"./index.js"}\n')
    (signals_dir / "index.js").write_text(
        "export function signal(value) { return { value }; }\n"
    )

    htm_dir = sandbox / "node_modules" / "htm"
    htm_dir.mkdir(parents=True)
    (htm_dir / "package.json").write_text(
        '{"type":"module","exports":{"./preact":"./preact.js"}}\n'
    )
    (htm_dir / "preact.js").write_text(
        "export function html(strings, ...values) { return { strings, values }; }\n"
    )

    preact_dir = sandbox / "node_modules" / "preact"
    preact_dir.mkdir(parents=True)
    (preact_dir / "package.json").write_text(
        '{"type":"module","exports":{"./hooks":"./hooks.js"}}\n'
    )
    (preact_dir / "hooks.js").write_text(
        "export function useState(value) { return [value, () => {}]; }\n"
    )

    script_path = sandbox / "adversarial-test.mjs"
    script_path.write_text(script)
    proc = subprocess.run(
        [node, str(script_path)],
        capture_output=True,
        timeout=15,
        text=True,
        cwd=sandbox,
    )
    assert proc.returncode == 0, (
        f"node dashboard-v2 module test failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return json.loads(proc.stdout)


@pytest.fixture
def _static_client() -> TestClient:
    app = FastAPI(title="aiperf-static-test")
    app.include_router(static_router)
    return TestClient(app)


class TestDashboardV2StaticServing:
    """Route-level coverage for the API server's static-v2 asset handler."""

    def test_dashboard_v2_without_trailing_slash_redirects(
        self, _static_client: TestClient
    ) -> None:
        response = _static_client.get("/dashboard-v2", follow_redirects=False)

        assert response.status_code == 307
        assert response.headers["location"] == "/dashboard-v2/"

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            param("/dashboard-v2/", "text/html", id="index"),
            param("/dashboard-v2/app.js", "application/javascript", id="js"),
            param("/dashboard-v2/style.css", "text/css", id="css"),
        ],
    )  # fmt: skip
    def test_dashboard_v2_serves_expected_content_types(
        self, _static_client: TestClient, path: str, expected: str
    ) -> None:
        response = _static_client.get(path)

        assert response.status_code == 200
        assert response.headers["content-type"].startswith(expected)

    @pytest.mark.parametrize(
        "path",
        [
            param("/dashboard-v2/%2E%2E/static/dashboard.html", id="encoded-dot-dot"),
            param("/dashboard-v2/%2e%2e/static/dashboard.html", id="encoded-dot-dot-lowercase"),
        ],
    )  # fmt: skip
    def test_dashboard_v2_rejects_path_traversal(
        self, _static_client: TestClient, path: str
    ) -> None:
        response = _static_client.get(path)

        assert response.status_code == 400
        assert response.json()["detail"] == "Invalid asset path"

    def test_dashboard_v2_missing_asset_returns_404(
        self, _static_client: TestClient
    ) -> None:
        response = _static_client.get("/dashboard-v2/no-such-asset.js")

        assert response.status_code == 404
        assert response.json()["detail"] == "no-such-asset.js not found"

    def test_dashboard_v2_index_serves_app_shell(
        self, _static_client: TestClient
    ) -> None:
        response = _static_client.get("/dashboard-v2/")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "<title>AIPerf Dashboard</title>" in response.text
        assert 'src="./app.js"' in response.text
        assert 'href="./style.css"' in response.text

    def test_dashboard_v2_does_not_serve_operator_ui_assets(
        self, _static_client: TestClient
    ) -> None:
        response = _static_client.get("/dashboard-v2/components/artifacts-card.js")

        assert response.status_code == 404
        assert response.json()["detail"] == "components/artifacts-card.js not found"


class TestDashboardV2InlineJS:
    """Cheap syntax gates that don't need a browser."""

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_v2_js_modules_parse(self) -> None:
        """``node --check`` each .js file under ``static-v2/``.

        Catches syntax regressions across the lib/ and components/ split
        without needing jsdom or a browser.
        """
        files = _v2_js_files()
        assert files, "no v2 JS modules found; static-v2/ is missing files"
        failures: list[str] = []
        for path in files:
            proc = subprocess.run(
                [_node_binary(), "--check", str(path)],
                capture_output=True,
                timeout=15,
            )
            if proc.returncode != 0:
                failures.append(
                    f"{path.relative_to(_STATIC_V2_DIR)}:\n  "
                    + proc.stderr.decode(errors="replace").replace("\n", "\n  ")
                )
        assert not failures, "v2 JS files failed node --check:\n" + "\n\n".join(
            failures
        )


class TestDashboardV2ModuleAdversarial:
    """Node-backed adversarial coverage for dashboard-v2 modules."""

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_formatters_use_fallbacks_for_non_finite_values(
        self, tmp_path: Path
    ) -> None:
        result = _run_v2_node_script(
            tmp_path,
            """
            import { fmtInt, fmtNumber, fmtPercent } from './lib/format.js';
            console.log(JSON.stringify({
              numberNaN: fmtNumber(Number.NaN),
              numberInfinity: fmtNumber(Number.POSITIVE_INFINITY),
              numberString: fmtNumber('42'),
              intNaN: fmtInt(Number.NaN),
              percentNaN: fmtPercent(Number.NaN),
              valid: fmtNumber(1234.567, 2),
            }));
            """,
        )

        assert result == {
            "numberNaN": "---",
            "numberInfinity": "---",
            "numberString": "---",
            "intNaN": "---",
            "percentNaN": "---",
            "valid": "1,234.57",
        }

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_timeseries_and_sparkline_ignore_non_finite_inputs(
        self, tmp_path: Path
    ) -> None:
        result = _run_v2_node_script(
            tmp_path,
            """
            import { pushSample, pluck } from './lib/timeseries.js';
            import { Sparkline } from './components/sparkline.js';
            const series = pushSample([
              { t: 3000, values: { avg: 3 } },
              { t: Number.NaN, values: { avg: 99 } },
              { t: 1000, values: { avg: 1 } },
            ], { t: 2000, values: { avg: Number.POSITIVE_INFINITY, current: 2 } });
            const points = pluck(series, 'avg');
            const spark = Sparkline({ points: [
              { t: 2, v: 2 },
              { t: Number.NaN, v: 5 },
              { t: 1, v: 1 },
              { t: 3, v: Number.POSITIVE_INFINITY },
            ] });
            console.log(JSON.stringify({
              series: series.map(s => s.t),
              points,
              hasNaN: JSON.stringify(spark).includes('NaN'),
              hasInfinity: JSON.stringify(spark).includes('Infinity'),
            }));
            """,
        )

        assert result["series"] == [1000, 2000, 3000]
        assert result["points"] == [{"t": 1000, "v": 1}, {"t": 3000, "v": 3}]
        assert result["hasNaN"] is False
        assert result["hasInfinity"] is False

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_phase_dispatch_preserves_terminal_and_failed_states(
        self, tmp_path: Path
    ) -> None:
        result = _run_v2_node_script(
            tmp_path,
            """
            import { handleWsMessage } from './lib/ws-dispatch.js';
            import { phases, logs } from './lib/state.js';
            phases.value = {};
            logs.value = [];
            handleWsMessage({
              type: 'credit_phase_complete',
              phase: 'profiling',
              stats: { start_ns: 1, requests_end_ns: 10, total_expected_requests: 100, final_requests_completed: 100 },
            });
            handleWsMessage({
              type: 'credit_phase_progress',
              phase: 'profiling',
              stats: { start_ns: 1, total_expected_requests: 100, requests_completed: 60 },
            });
            handleWsMessage({
              type: 'credit_phase_failed',
              phase: 'cleanup',
              stats: { start_ns: 1, requests_end_ns: 2, request_errors: 28 },
            });
            console.log(JSON.stringify({ phases: phases.value, logs: logs.value }));
            """,
        )

        assert result["phases"]["profiling"]["complete"] is True
        assert result["phases"]["profiling"]["active"] is False
        assert result["phases"]["profiling"]["final_requests_completed"] == 100
        assert result["phases"]["cleanup"]["failed"] is True
        assert result["phases"]["cleanup"]["complete"] is False
        assert any(
            "Phase failed: cleanup" in entry["message"] for entry in result["logs"]
        )

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_server_metrics_render_non_finite_values_as_fallback(
        self, tmp_path: Path
    ) -> None:
        result = _run_v2_node_script(
            tmp_path,
            """
            import { ServerMetrics } from './components/server-metrics.js';
            import { serverMetrics } from './lib/state.js';
            serverMetrics.value = [{ endpoint: 'srv-a', metrics: [
              { name: 'kv_cache_utilization', value: Number.POSITIVE_INFINITY, unit: '%' },
              { name: 'queue_depth', value: Number.NaN, unit: 'requests' },
              { name: 'goodput', value: 1234.56, unit: 'req/s' },
            ] }];
            const rendered = JSON.stringify(ServerMetrics());
            console.log(JSON.stringify({
              rendered,
              hasInfinity: rendered.includes('Infinity'),
              hasNaN: rendered.includes('NaN'),
              fallbackCount: (rendered.match(/---/g) || []).length,
            }));
            """,
        )

        assert result["hasInfinity"] is False
        assert result["hasNaN"] is False
        assert result["fallbackCount"] >= 2
        assert "1,235 req/s" in result["rendered"]

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_server_metrics_normalization_preserves_full_stats(
        self, tmp_path: Path
    ) -> None:
        result = _run_v2_node_script(
            tmp_path,
            """
            import { normalizeEndpointSummaries } from './lib/ws-dispatch.js';
            const summaries = normalizeEndpointSummaries({
              'http://srv:8000': {
                metrics: {
                  kv_cache_utilization: {
                    unit: 'ratio',
                    series: [{ stats: { avg: 0.92, min: 0.70, max: 0.99, p99: 0.98, p90: 0.95, p50: 0.90 } }],
                  },
                  tokens_total: {
                    unit: 'tokens',
                    series: [{ stats: { value: 125000 } }],
                  },
                },
              },
            });
            console.log(JSON.stringify(summaries));
            """,
        )

        metrics = {m["name"]: m for m in result[0]["metrics"]}
        assert metrics["kv_cache_utilization"] == {
            "name": "kv_cache_utilization",
            "value": 0.92,
            "unit": "ratio",
            "avg": 0.92,
            "min": 0.70,
            "max": 0.99,
            "p99": 0.98,
            "p90": 0.95,
            "p50": 0.90,
        }
        assert metrics["tokens_total"] == {
            "name": "tokens_total",
            "value": 125000,
            "unit": "tokens",
        }

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_full_metrics_table_formats_stats_and_fallbacks(
        self, tmp_path: Path
    ) -> None:
        result = _run_v2_node_script(
            tmp_path,
            """
            import { FullMetricsTable } from './components/full-metrics-table.js';
            function flatten(node, acc = { text: [], templates: [] }) {
              if (node == null || typeof node === 'boolean' || typeof node === 'function') return acc;
              if (Array.isArray(node)) { for (const item of node) flatten(item, acc); return acc; }
              if (typeof node === 'string' || typeof node === 'number') { acc.text.push(String(node)); return acc; }
              if (node.strings) {
                acc.templates.push(Array.from(node.strings).join(''));
                for (const value of node.values ?? []) flatten(value, acc);
              }
              return acc;
            }
            const rendered = FullMetricsTable({
              title: 'Full Benchmark Metrics',
              rows: [
                { key: 'latency', metric: 'Request Latency', unit: 'ms', avg: 510.25, min: 120, max: 9000, p99: 812, p90: Number.NaN, p50: null },
              ],
            });
            const flat = flatten(rendered);
            const text = flat.text.join('|');
            console.log(JSON.stringify({
              text,
              templates: flat.templates.join('|'),
              fallbackCount: (text.match(/---/g) || []).length,
            }));
            """,
        )

        assert "Full Benchmark Metrics" in result["text"]
        assert "Request Latency" in result["text"]
        assert "ms" in result["text"]
        assert "510.25" in result["text"]
        assert "9,000" in result["text"]
        assert result["fallbackCount"] >= 2
        assert "full-metrics-table" in result["templates"]

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_full_metrics_adapters_normalize_three_sources(
        self, tmp_path: Path
    ) -> None:
        result = _run_v2_node_script(
            tmp_path,
            """
            import {
              rowsFromMetrics,
              rowsFromServerMetrics,
            } from './components/full-metrics-table.js';
            const benchmarkRows = rowsFromMetrics([
              { tag: 'request_latency', header: 'Request Latency', unit: 'ms', avg: 1, min: 2, max: 3, p99: 4, p90: 5, p50: 6 },
              { tag: 'bad_metric', header: null, unit: 'x', avg: Number.POSITIVE_INFINITY },
              null,
            ]);
            const gpuRows = rowsFromMetrics([
              {
                tag: 'gpu_utilization_dcgm_gpu_0',
                header: 'GPU Utilization | http://srv:9400 | GPU 0 | NVIDIA H100',
                unit: '%',
                current: 96.0,
              },
            ]);
            const serverRows = rowsFromServerMetrics([
              { endpoint: 'http://srv:8000', metrics: [
                { name: 'kv_cache_utilization', unit: 'ratio', avg: 0.5, min: 0.1, max: 0.9, p99: 0.8, p90: 0.7, p50: 0.4 },
              ]},
            ]);
            console.log(JSON.stringify({ benchmarkRows, gpuRows, serverRows }));
            """,
        )

        assert result["benchmarkRows"][0] == {
            "key": "request_latency",
            "metric": "Request Latency",
            "unit": "ms",
            "avg": 1,
            "min": 2,
            "max": 3,
            "p99": 4,
            "p90": 5,
            "p50": 6,
        }
        assert result["benchmarkRows"][1]["metric"] == "bad_metric"
        assert result["benchmarkRows"][1]["avg"] is None
        assert result["gpuRows"][0]["key"] == "gpu_utilization_dcgm_gpu_0"
        assert (
            result["gpuRows"][0]["metric"]
            == "GPU Utilization | http://srv:9400 | GPU 0 | NVIDIA H100"
        )
        assert result["gpuRows"][0]["unit"] == "%"
        assert result["gpuRows"][0]["avg"] == 96.0
        assert result["serverRows"][0]["key"] == "http://srv:8000::kv_cache_utilization"
        assert (
            result["serverRows"][0]["metric"]
            == "http://srv:8000 · kv_cache_utilization"
        )

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_records_dispatch_reads_real_wire_keys(self, tmp_path: Path) -> None:
        """RecordsProcessingStatsMessage / AllRecordsReceivedMessage nest their
        counters under ``processing_stats`` / ``final_processing_stats`` on the
        wire (never ``stats``); the dispatcher must read those keys."""
        result = _run_v2_node_script(
            tmp_path,
            """
            import { handleWsMessage } from './lib/ws-dispatch.js';
            import { records } from './lib/state.js';
            handleWsMessage({
              type: 'processing_stats',
              processing_stats: {
                success_records: 97,
                error_records: 1,
                final_requests_completed: 98,
                start_ns: 1000,
              },
            });
            const mid = { ...records.value };
            handleWsMessage({
              type: 'all_records_received',
              final_processing_stats: {
                success_records: 99,
                error_records: 1,
                final_requests_completed: 100,
                records_end_ns: 2000,
              },
            });
            console.log(JSON.stringify({ mid, final: records.value }));
            """,
        )

        assert result["mid"]["successRecords"] == 97
        assert result["mid"]["errorRecords"] == 1
        assert result["mid"]["finalRequestsCompleted"] == 98
        assert result["mid"]["active"] is True
        assert result["final"]["successRecords"] == 99
        assert result["final"]["errorRecords"] == 1
        assert result["final"]["finalRequestsCompleted"] == 100
        assert result["final"]["endNs"] == 2000
        assert result["final"]["complete"] is True

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_worker_group_in_flight_derived_from_wire_counters(
        self, tmp_path: Path
    ) -> None:
        """WorkerTaskStats.in_progress is a non-serialized @property; the
        dispatcher derives in-flight as total - completed - failed for both
        the group row and each per-worker child."""
        result = _run_v2_node_script(
            tmp_path,
            """
            import { handleWsMessage } from './lib/ws-dispatch.js';
            import { workerGroups } from './lib/state.js';
            handleWsMessage({
              type: 'worker_group_stats',
              group_id: 'wg-primary',
              status: 'healthy',
              task_stats: { total: 101, completed: 97, failed: 1 },
              worker_statuses: { 'w-a': 'healthy', 'w-b': 'healthy' },
              worker_task_stats: {
                'w-a': { total: 52, completed: 51, failed: 0 },
                'w-b': { total: 49, completed: 46, failed: 1 },
              },
            });
            const g = workerGroups.value['wg-primary'];
            console.log(JSON.stringify({
              group: g.inFlight,
              a: g.workers['w-a'].inFlight,
              b: g.workers['w-b'].inFlight,
            }));
            """,
        )

        assert result == {"group": 3, "a": 1, "b": 2}

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_unknown_websocket_messages_log_once(self, tmp_path: Path) -> None:
        result = _run_v2_node_script(
            tmp_path,
            """
            import { handleWsMessage } from './lib/ws-dispatch.js';
            import { logs } from './lib/state.js';
            logs.value = [];
            handleWsMessage(null);
            handleWsMessage('bad');
            handleWsMessage({ type: 'future_payload' });
            handleWsMessage({ type: 'future_payload' });
            console.log(JSON.stringify(logs.value));
            """,
        )

        matching = [entry for entry in result if "future_payload" in entry["message"]]
        assert len(matching) == 1
        assert matching[0]["category"] == "ws"

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_gpu_telemetry_accepts_headers_without_model_suffix(
        self, tmp_path: Path
    ) -> None:
        result = _run_v2_node_script(
            tmp_path,
            """
            import { GpuTelemetryCard } from './components/gpu-telemetry.js';
            import { telemetryMetrics } from './lib/state.js';
            telemetryMetrics.value = [
              {
                tag: 'gpu_utilization_dcgm_http___node1_9401_metrics_gpu0_uuid',
                header: 'GPU Utilization | node1:9401 | GPU 0',
                unit: '%',
                current: 88.5,
                avg: 87.0,
              },
            ];
            const rendered = JSON.stringify(GpuTelemetryCard());
            console.log(JSON.stringify({
              rendered,
              hasGpuCard: rendered.includes('node1:9401 | GPU 0'),
              hasUtilization: rendered.includes('Utilization'),
              hasValue: rendered.includes('88.5'),
            }));
            """,
        )

        assert result["hasGpuCard"] is True, result["rendered"]
        assert result["hasUtilization"] is True, result["rendered"]
        assert result["hasValue"] is True, result["rendered"]

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_worker_table_tolerates_malformed_missing_worker_state(
        self, tmp_path: Path
    ) -> None:
        result = _run_v2_node_script(
            tmp_path,
            """
            import { workerGroups } from './lib/state.js';
            import { WorkerTable } from './components/worker-table.js';
            function flatten(node, acc = { text: [], templates: [] }) {
              if (node == null || typeof node === 'boolean' || typeof node === 'function') return acc;
              if (Array.isArray(node)) { for (const item of node) flatten(item, acc); return acc; }
              if (typeof node === 'string' || typeof node === 'number') { acc.text.push(String(node)); return acc; }
              if (node.strings) {
                acc.templates.push(Array.from(node.strings).join(''));
                for (const value of node.values ?? []) flatten(value, acc);
              }
              return acc;
            }
            workerGroups.value = {
              'worker-group-b': null,
              'worker-group-a': {
                status: null,
                workers: {
                  'worker-child-b': null,
                  'worker-child-a': { completed: 7 },
                },
              },
              'worker-group-c': 'bad-state',
            };
            const flat = flatten(WorkerTable());
            console.log(JSON.stringify({
              text: flat.text.join('|'),
              rowCount: flat.templates.filter(t => t.includes('<tr')).length,
              hasTitle: flat.templates.join('|').includes('Worker Groups'),
              hasUndefined: flat.text.includes('undefined'),
              hasNull: flat.text.includes('null'),
            }));
            """,
        )

        assert result["rowCount"] >= 3
        assert result["hasTitle"] is True
        assert result["hasUndefined"] is False
        assert result["hasNull"] is False

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_worker_table_sorts_many_workers_and_escapes_ids(
        self, tmp_path: Path
    ) -> None:
        result = _run_v2_node_script(
            tmp_path,
            """
            import { workerGroups } from './lib/state.js';
            import { WorkerTable } from './components/worker-table.js';
            function flatten(node, acc = { text: [], templates: [] }) {
              if (node == null || typeof node === 'boolean' || typeof node === 'function') return acc;
              if (Array.isArray(node)) { for (const item of node) flatten(item, acc); return acc; }
              if (typeof node === 'string' || typeof node === 'number') { acc.text.push(String(node)); return acc; }
              if (node.strings) {
                acc.templates.push(Array.from(node.strings).join(''));
                for (const value of node.values ?? []) flatten(value, acc);
              }
              return acc;
            }
            const dangerousId = 'worker-zz-<img src=x onerror=alert(1)>-&';
            const workers = {};
            for (let i = 20; i >= 0; i -= 1) {
              const suffix = String(i).padStart(3, '0');
              workers[`worker-child-${suffix}`] = { status: i % 2 ? 'healthy' : 'high_load', completed: i };
            }
            workers[dangerousId] = { status: 'healthy' };
            workerGroups.value = {
              'worker-group-main': { status: 'healthy', declaredWorkers: 22, readyWorkers: 22, workers },
            };
            const flat = flatten(WorkerTable());
            const text = flat.text.join('|');
            const templates = flat.templates.join('|');
            console.log(JSON.stringify({
              rowCount: flat.templates.filter(t => t.includes('<tr')).length,
              firstIndex: text.indexOf('child-000'),
              middleIndex: text.indexOf('child-010'),
              lastIndex: text.indexOf('child-020'),
              dangerousInText: text.includes('<img src=x onerror=alert(1)>'),
              dangerousInTemplates: templates.includes('<img src=x onerror=alert(1)>'),
            }));
            """,
        )

        assert result["rowCount"] == 24
        assert 0 <= result["firstIndex"] < result["middleIndex"] < result["lastIndex"]
        assert result["dangerousInText"] is True
        assert result["dangerousInTemplates"] is False

    @pytest.mark.skipif(_node_binary() is None, reason=_NODE_REASON)
    def test_websocket_error_and_teardown_do_not_lose_status_or_reconnect(
        self, tmp_path: Path
    ) -> None:
        result = _run_v2_node_script(
            tmp_path,
            """
            import { connection, logs } from './lib/state.js';
            import { connectWebSocket, teardownWebSocket } from './lib/ws.js';
            const timers = [];
            globalThis.window = { location: { protocol: 'http:', host: 'example.test' } };
            globalThis.setTimeout = (fn, ms) => { timers.push({ fn, ms }); return timers.length - 1; };
            globalThis.clearTimeout = () => {};
            class FakeWebSocket {
              static instances = [];
              constructor(url) { this.url = url; FakeWebSocket.instances.push(this); }
              send() {}
              close() { this.closed = true; this.onclose?.({ code: 1000 }); }
            }
            globalThis.WebSocket = FakeWebSocket;

            connectWebSocket();
            const first = FakeWebSocket.instances[0];
            first.onerror(new Error('boom'));
            first.onclose({ code: 1006 });
            const statusAfterErrorClose = connection.value;
            const reconnectsAfterErrorClose = timers.length;
            const errorLogged = logs.value.some(entry => entry.severity === 'error' && entry.message === 'WebSocket error');
            timers[0].fn();
            teardownWebSocket();
            console.log(JSON.stringify({
              statusAfterErrorClose,
              reconnectsAfterErrorClose,
              errorLogged,
              secondClosed: FakeWebSocket.instances[1].closed,
              reconnectsAfterTeardown: timers.length,
            }));
            """,
        )

        assert result == {
            "statusAfterErrorClose": "error",
            "reconnectsAfterErrorClose": 1,
            "errorLogged": True,
            "secondClosed": True,
            "reconnectsAfterTeardown": 1,
        }


class TestDashboardV2Render:
    """Drive a real Chromium browser against the v2 dashboard.

    The v2 app is a Preact/signals SPA served from ``/dashboard-v2`` with
    ES modules under ``/dashboard-v2/lib/*`` and ``/dashboard-v2/components/*``.
    """

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_v2_boots_and_shows_config_bar(self, _page: Page) -> None:
        """v2 dashboard must boot, render the config bar, and flip status to Connected."""
        console_errors: list[str] = []
        _page.on(
            "console",
            lambda msg: console_errors.append(f"{msg.type}: {msg.text}")
            if msg.type in ("error",)
            else None,
        )

        with _run_server(_build_multi_phase_cfg()) as base_url:
            _page.goto(f"{base_url}/dashboard-v2", wait_until="networkidle")
            _page.wait_for_selector("#config-bar.visible", timeout=10_000)
            _page.wait_for_function(
                """() => {
                    const dot = document.querySelector('.status-dot.connected');
                    return dot !== null;
                }""",
                timeout=10_000,
            )

            config_text = _page.locator("#config-bar").text_content() or ""

        assert console_errors == [], (
            "unexpected browser console errors:\n  " + "\n  ".join(console_errors)
        )

        # Same label set as v1's renderConfig — v2 reimplements the same
        # source-of-truth mapping against the current BenchmarkConfig shape.
        required = [
            "Model",
            "llama3-8b",
            "llama3-70b",
            "Endpoint",
            "chat (streaming)",
            "URL",
            "http://srv:8000/v1/chat/completions",
            "warmup Type",
            "concurrency",
            "warmup Concurrency",
            "4",
            "profiling Type",
            "poisson",
            "profiling Rate",
            "20 QPS",
            "profiling Duration",
            "5m 0s",
        ]
        missing = [t for t in required if t not in config_text]
        assert not missing, (
            f"v2 config bar missing tokens: {missing}\nactual text: {config_text!r}"
        )
        assert "SHOULD_NOT_LEAK" not in config_text

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_v2_phases_keyed_by_name_not_collapsed(self, _page: Page) -> None:
        """v2's PhaseCards keys on the backend phase name (fixes the v1 collapse bug).

        We push ``credit_phase_start`` for each configured phase via the
        stub WebSocket and assert that one phase card appears per name.
        """
        # Config has 3 phases with non-warmup-or-profiling names to prove
        # the v1 bucketing behavior is gone.
        cfg = AIPerfConfig(
            benchmark={
                "models": ["llama3-8b"],
                "endpoint": {
                    "urls": ["http://srv:8000/v1/chat/completions"],
                    "type": "chat",
                },
                "datasets": [
                    {
                        "name": "default",
                        "type": "synthetic",
                        "entries": 10,
                        "prompts": {"isl": 128, "osl": 32},
                    }
                ],
                "phases": [
                    {
                        "name": "phase_alpha",
                        "type": "concurrency",
                        "kind": "profiling",
                        "requests": 10,
                        "concurrency": 1,
                    },
                    {
                        "name": "phase_beta",
                        "type": "concurrency",
                        "kind": "profiling",
                        "requests": 20,
                        "concurrency": 2,
                    },
                    {
                        "name": "phase_gamma",
                        "type": "concurrency",
                        "kind": "profiling",
                        "requests": 30,
                        "concurrency": 3,
                    },
                ],
                "runtime": {"api_port": 8080},
            }
        )

        with _run_server(cfg, broadcast_phases=True) as base_url:
            _page.goto(f"{base_url}/dashboard-v2", wait_until="networkidle")
            _page.wait_for_selector("#config-bar.visible", timeout=10_000)

            # Give the WS a moment to emit the three credit_phase_start messages
            # and let Preact flush the resulting signal updates.
            _page.wait_for_function(
                """() => document.querySelectorAll('.phase-card').length >= 3""",
                timeout=10_000,
            )

            phase_names = _page.evaluate(
                """() => Array.from(document.querySelectorAll('.phase-name'))
                    .map(n => n.textContent.trim())"""
            )

        assert set(phase_names) == {"phase_alpha", "phase_beta", "phase_gamma"}, (
            f"v2 should render one phase card per backend phase name; got {phase_names!r}"
        )

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_v2_serves_all_module_assets(self, _page: Page) -> None:
        """Every /dashboard-v2/lib/* and /dashboard-v2/components/* request must 200.

        Catches regressions in the FastAPI static asset handler (path
        traversal rejects, wrong content-type, missing dir, etc.).
        """
        bad_responses: list[tuple[str, int]] = []

        def on_response(response: Any) -> None:
            url = response.url
            if "/dashboard-v2/" in url and response.status >= 400:
                bad_responses.append((url, response.status))

        _page.on("response", on_response)

        with _run_server(_build_multi_phase_cfg()) as base_url:
            _page.goto(f"{base_url}/dashboard-v2", wait_until="networkidle")
            _page.wait_for_selector("#config-bar.visible", timeout=10_000)

        assert not bad_responses, (
            "some /dashboard-v2/ assets returned >= 400:\n  "
            + "\n  ".join(f"{s} {u}" for u, s in bad_responses)
        )


# -----------------------------------------------------------------------------
# v2: realtime metrics + GPU telemetry cards
# -----------------------------------------------------------------------------


def _metric_result(
    tag: str,
    header: str,
    unit: str,
    *,
    current: float | None = None,
    avg: float | None = None,
    p99: float | None = None,
    max: float | None = None,
    p50: float | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable ``MetricResult`` shaped like msgspec emits."""
    return {
        "tag": tag,
        "header": header,
        "unit": unit,
        "count": 60,
        "current": current,
        "sum": None,
        "avg": avg,
        "p1": None,
        "p5": None,
        "p10": None,
        "p25": None,
        "p50": p50 if p50 is not None else avg,
        "p75": None,
        "p90": p99,
        "p95": None,
        "p99": p99,
        "min": None,
        "max": max,
        "std": None,
    }


class TestDashboardV2RealtimeMetrics:
    """``realtime_metrics`` WS messages must populate the KPI tile grid.

    Metric selection + stat picking in ``components/realtime-metrics.js`` is
    grounded in published LLM-inference benchmarking guidance:

    * NVIDIA NIM Benchmarking docs (TTFT, ITL, E2E latency, TPS, RPS)
    * AIPerf's customer docs (Pareto analysis, Goodput for SLO compliance)
    * BentoML LLM Inference Handbook (Goodput = "direct measure of meeting
      performance and user-experience goals")
    * vLLM production guide (p99 for tail SLOs, ITL headline as streaming
      smoothness)

    SLO policy: the dashboard only renders pass/fail chips against
    thresholds the user declared via ``cfg.benchmark.slos`` (the same dict AIPerf's
    goodput feature consumes). No fabricated "industry defaults" - silence
    is the honest option when the user hasn't said what good looks like.
    """

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_realtime_metrics_tiles_render_expected_values(self, _page: Page) -> None:
        """Each hero tile must show its canonical primary stat + secondary stat.

        No ``cfg.benchmark.slos`` is configured in the multi-phase fixture, so no chip
        should render on latency tiles - we're testing the stat-picker, not
        the threshold policy.
        """
        payload = {
            "type": "realtime_metrics",
            "metrics": [
                _metric_result(
                    "request_throughput",
                    "Request Throughput",
                    "req/s",
                    current=19.8,
                    avg=20.1,
                    p99=21.0,
                ),
                _metric_result(
                    "output_token_throughput",
                    "Output Token Throughput",
                    "tok/s",
                    current=1823.4,
                    avg=1798.1,
                    p99=1920.0,
                ),
                _metric_result(
                    "request_latency",
                    "Request Latency",
                    "ms",
                    current=482.3,
                    avg=465.5,
                    p99=812.0,
                ),
                _metric_result(
                    "time_to_first_token",
                    "Time To First Token",
                    "ms",
                    current=73.2,
                    avg=68.7,
                    p99=118.0,
                ),
                _metric_result(
                    "inter_token_latency",
                    "Inter Token Latency",
                    "ms",
                    current=12.1,
                    avg=11.8,
                    p99=21.4,
                ),
            ],
        }

        with _run_server(
            _build_multi_phase_cfg(), extra_ws_payloads=[payload]
        ) as base_url:
            _page.goto(f"{base_url}/dashboard-v2", wait_until="networkidle")
            _page.wait_for_selector("#config-bar.visible", timeout=10_000)
            _page.wait_for_function(
                "() => document.querySelectorAll('.kpi-tile').length >= 5",
                timeout=10_000,
            )

            tiles = _page.evaluate(
                """() => Array.from(document.querySelectorAll('.kpi-tile')).map(t => ({
                    label:        t.querySelector('.kpi-tile-label > span:first-child')?.textContent?.trim(),
                    primary_stat: t.querySelector('.kpi-tile-primary-stat')?.textContent?.trim(),
                    val:          t.querySelector('.kpi-big-val')?.textContent?.trim(),
                    unit:         t.querySelector('.kpi-big-unit')?.textContent?.trim() ?? '',
                    sub:          t.querySelector('.kpi-tile-sub')?.textContent?.trim().replace(/\\s+/g, ' '),
                    slo_kind:     Array.from(t.classList).find(c => c.startsWith('kpi-tile--slo-'))?.replace('kpi-tile--slo-', '') ?? null,
                    chip_kind:    Array.from(t.querySelector('.kpi-chip')?.classList ?? []).find(c => c.startsWith('kpi-chip--'))?.replace('kpi-chip--', '') ?? null,
                }))"""
            )

        by_label = {t["label"]: t for t in tiles}

        # --- Capacity tier: rates use `current`, no SLO chip. ---
        rt = by_label["Requests/s"]
        assert rt["primary_stat"] == "current"
        assert rt["val"] == "19.80" and rt["unit"] == "req/s"
        assert "20.10" in rt["sub"] and "avg" in rt["sub"].lower()
        assert rt["slo_kind"] is None, (
            "throughput is not an SLO metric by NIM convention"
        )

        out = by_label["Output Tokens/s"]
        assert out["primary_stat"] == "current"
        assert out["val"] == "1,823" and out["unit"] == "tok/s"
        assert "1,798" in out["sub"]

        # --- UX tier: TTFT headline = p99. ---
        ttft = by_label["TTFT"]
        assert ttft["primary_stat"] == "p99"
        assert ttft["val"] == "118.00" and ttft["unit"] == "ms"
        assert "68.70" in ttft["sub"] and "avg" in ttft["sub"].lower()
        # No user SLO declared → no chip, no border color.
        assert ttft["slo_kind"] is None and ttft["chip_kind"] is None, ttft

        # --- SLO tier: Request Latency = p99, tail guarantee. ---
        rl = by_label["Request Latency"]
        assert rl["primary_stat"] == "p99"
        assert rl["val"] == "812.00" and rl["unit"] == "ms"
        assert rl["slo_kind"] is None, rl  # same rule

        # --- UX tier: ITL headline = avg (streaming smoothness), p99 in sub. ---
        itl = by_label["ITL"]
        assert itl["primary_stat"] == "avg"
        assert itl["val"] == "11.80" and itl["unit"] == "ms"
        assert "21.40" in itl["sub"] and "p99" in itl["sub"].lower()
        assert itl["slo_kind"] is None, itl

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_realtime_metrics_chip_honors_user_slo_and_renders_threshold(
        self, _page: Page
    ) -> None:
        """When the user declares ``cfg.benchmark.slos``, the chip is binary pass/fail
        against that value and the chip label echoes the user's threshold.

        Customer: "I want TTFT p99 ≤ 100 ms, ITL avg ≤ 10 ms, Request
        Latency p99 ≤ 1500 ms." One run meets them all, one misses one,
        and we check both outcomes against the same SLO declaration.
        """
        cfg_with_slo = AIPerfConfig(
            benchmark={
                "models": ["llama3-8b"],
                "endpoint": {
                    "urls": ["http://srv:8000/v1/chat/completions"],
                    "type": "chat",
                },
                "datasets": [
                    {
                        "name": "default",
                        "type": "synthetic",
                        "entries": 10,
                        "prompts": {"isl": 128, "osl": 32},
                    }
                ],
                "phases": [
                    {
                        "name": "default",
                        "type": "concurrency",
                        "kind": "profiling",
                        "requests": 10,
                        "concurrency": 1,
                    }
                ],
                "slos": {
                    "time_to_first_token": 100.0,
                    "inter_token_latency": 10.0,
                    "request_latency": 1500.0,
                },
                "runtime": {"api_port": 8080},
            }
        )

        # Scenario A: all three pass → green chips with the user's thresholds.
        passing = {
            "type": "realtime_metrics",
            "metrics": [
                _metric_result(
                    "time_to_first_token",
                    "Time To First Token",
                    "ms",
                    current=80.0,
                    avg=75.0,
                    p99=92.0,
                ),  # ≤ 100 ✓
                _metric_result(
                    "inter_token_latency",
                    "Inter Token Latency",
                    "ms",
                    current=7.5,
                    avg=7.8,
                    p99=12.0,
                ),  # avg 7.8 ≤ 10 ✓
                _metric_result(
                    "request_latency",
                    "Request Latency",
                    "ms",
                    current=1100.0,
                    avg=1050.0,
                    p99=1420.0,
                ),  # ≤ 1500 ✓
            ],
        }

        def collect_slos(base_url: str) -> dict[str, dict[str, Any]]:
            _page.goto(f"{base_url}/dashboard-v2", wait_until="networkidle")
            _page.wait_for_function(
                "() => Array.from(document.querySelectorAll('.kpi-tile-label > span:first-child'))"
                "       .some(s => s.textContent.trim() === 'TTFT')",
                timeout=10_000,
            )
            # Wait until at least one SLO chip has rendered so we don't race.
            _page.wait_for_function(
                "() => document.querySelector('.kpi-tile[class*=\"slo-\"] .kpi-chip')",
                timeout=10_000,
            )
            return _page.evaluate(
                """() => Object.fromEntries(
                    Array.from(document.querySelectorAll('.kpi-tile')).map(t => ([
                        t.querySelector('.kpi-tile-label > span:first-child')?.textContent?.trim(),
                        {
                            slo_kind:  Array.from(t.classList)
                                .find(c => c.startsWith('kpi-tile--slo-'))
                                ?.replace('kpi-tile--slo-', '') ?? null,
                            chip_text: t.querySelector('.kpi-chip')?.textContent?.trim()
                                ?.replace(/\\s+/g, ' ') ?? null,
                        }
                    ]))
                )"""
            )

        with _run_server(cfg_with_slo, extra_ws_payloads=[passing]) as base_url:
            pass_state = collect_slos(base_url)

        assert pass_state["TTFT"]["slo_kind"] == "good", pass_state["TTFT"]
        assert "100" in pass_state["TTFT"]["chip_text"], pass_state["TTFT"]
        assert pass_state["ITL"]["slo_kind"] == "good", pass_state["ITL"]
        assert "10" in pass_state["ITL"]["chip_text"], pass_state["ITL"]
        assert pass_state["Request Latency"]["slo_kind"] == "good", pass_state[
            "Request Latency"
        ]
        assert "1500" in pass_state["Request Latency"]["chip_text"], pass_state[
            "Request Latency"
        ]

        # Scenario B: TTFT violates (p99=140 > 100). Others still pass.
        failing = {
            "type": "realtime_metrics",
            "metrics": [
                _metric_result(
                    "time_to_first_token",
                    "Time To First Token",
                    "ms",
                    current=130.0,
                    avg=110.0,
                    p99=140.0,
                ),  # > 100 ✗
                _metric_result(
                    "inter_token_latency",
                    "Inter Token Latency",
                    "ms",
                    current=7.5,
                    avg=7.8,
                    p99=12.0,
                ),
                _metric_result(
                    "request_latency",
                    "Request Latency",
                    "ms",
                    current=1100.0,
                    avg=1050.0,
                    p99=1420.0,
                ),
            ],
        }

        with _run_server(cfg_with_slo, extra_ws_payloads=[failing]) as base_url:
            fail_state = collect_slos(base_url)

        assert fail_state["TTFT"]["slo_kind"] == "bad", fail_state["TTFT"]
        assert "100" in fail_state["TTFT"]["chip_text"], fail_state["TTFT"]
        # Others unchanged.
        assert fail_state["ITL"]["slo_kind"] == "good"
        assert fail_state["Request Latency"]["slo_kind"] == "good"

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_realtime_metrics_no_chip_without_user_slo(self, _page: Page) -> None:
        """Absolute regression guard against fabricated defaults: no chip may
        appear on any latency tile when ``cfg.benchmark.slos`` does not cover it.

        Tile renders values + secondary stat, but no pass/fail judgment —
        the dashboard does not claim to know whether 500 ms TTFT is "good"
        for the customer's use case.
        """
        payload = {
            "type": "realtime_metrics",
            "metrics": [
                _metric_result(
                    "time_to_first_token",
                    "Time To First Token",
                    "ms",
                    current=480.0,
                    avg=450.0,
                    p99=720.0,
                ),
                _metric_result(
                    "request_latency",
                    "Request Latency",
                    "ms",
                    current=9000.0,
                    avg=8500.0,
                    p99=11000.0,
                ),
                _metric_result(
                    "inter_token_latency",
                    "Inter Token Latency",
                    "ms",
                    current=95.0,
                    avg=90.0,
                    p99=180.0,
                ),
            ],
        }

        with _run_server(
            _build_multi_phase_cfg(), extra_ws_payloads=[payload]
        ) as base_url:
            _page.goto(f"{base_url}/dashboard-v2", wait_until="networkidle")
            _page.wait_for_function(
                "() => Array.from(document.querySelectorAll('.kpi-tile-label > span:first-child'))"
                "       .some(s => s.textContent.trim() === 'TTFT')",
                timeout=10_000,
            )
            # Small settle so any erroneously-rendered chip has time to show up.
            _page.wait_for_timeout(400)

            chips = _page.evaluate(
                """() => Array.from(document.querySelectorAll('.kpi-tile')).map(t => ({
                    label: t.querySelector('.kpi-tile-label > span:first-child')?.textContent?.trim(),
                    has_chip: !!t.querySelector('.kpi-chip'),
                    slo_class: Array.from(t.classList).find(c => c.startsWith('kpi-tile--slo-')) ?? null,
                }))"""
            )

        for tile in chips:
            assert tile["has_chip"] is False, (
                f"tile {tile['label']!r}: no chip should render without a user SLO; got {tile}"
            )
            assert tile["slo_class"] is None, tile

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_realtime_metrics_goodput_tile_green_when_100_percent(
        self, _page: Page
    ) -> None:
        """Goodput tile is green iff every request met every user SLO.

        Binary policy ties the reliability chip to the user's own
        declarations rather than a fabricated pass-rate band.
        """
        cfg = AIPerfConfig(
            benchmark={
                "models": ["llama3-8b"],
                "endpoint": {
                    "urls": ["http://srv:8000/v1/chat/completions"],
                    "type": "chat",
                },
                "datasets": [
                    {
                        "name": "default",
                        "type": "synthetic",
                        "entries": 10,
                        "prompts": {"isl": 128, "osl": 32},
                    }
                ],
                "phases": [
                    {
                        "name": "default",
                        "type": "concurrency",
                        "kind": "profiling",
                        "requests": 10,
                        "concurrency": 1,
                    }
                ],
                "slos": {"time_to_first_token": 500.0, "inter_token_latency": 30.0},
                "runtime": {"api_port": 8080},
            }
        )

        # 100% passes → green.
        perfect = {
            "type": "realtime_metrics",
            "metrics": [
                _metric_result("goodput", "Goodput", "req/s", current=19.2),
                _metric_result(
                    "request_count", "Request Count", "requests", current=1000.0
                ),
                _metric_result(
                    "good_request_count",
                    "Good Request Count",
                    "requests",
                    current=1000.0,
                ),
            ],
        }
        with _run_server(cfg, extra_ws_payloads=[perfect]) as base_url:
            _page.goto(f"{base_url}/dashboard-v2", wait_until="networkidle")
            _page.wait_for_function(
                "() => Array.from(document.querySelectorAll('.kpi-tile-label > span:first-child'))"
                "       .some(s => s.textContent.trim() === 'Goodput')",
                timeout=10_000,
            )
            perfect_state = _page.evaluate(
                """() => {
                    const tile = Array.from(document.querySelectorAll('.kpi-tile'))
                      .find(t => t.querySelector('.kpi-tile-label > span:first-child')?.textContent.trim() === 'Goodput');
                    return {
                      kind: Array.from(tile?.classList ?? [])
                              .find(c => c.startsWith('kpi-tile--slo-'))
                              ?.replace('kpi-tile--slo-', '') ?? null,
                      chip: tile?.querySelector('.kpi-chip')?.textContent?.trim() ?? null,
                    };
                }"""
            )
        assert perfect_state["kind"] == "good", perfect_state
        # Chip headlines the failure *count* ("0 failed"), not a pass rate —
        # so a glance sees the size of the problem, not a lulling 100% number.
        assert "0 failed" in (perfect_state["chip"] or ""), perfect_state

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_realtime_metrics_goodput_tile_warn_when_any_failure(
        self, _page: Page
    ) -> None:
        """Any user-SLO failure → warn. No fake band; binary at the user's bar."""
        cfg = AIPerfConfig(
            benchmark={
                "models": ["llama3-8b"],
                "endpoint": {
                    "urls": ["http://srv:8000/v1/chat/completions"],
                    "type": "chat",
                },
                "datasets": [
                    {
                        "name": "default",
                        "type": "synthetic",
                        "entries": 10,
                        "prompts": {"isl": 128, "osl": 32},
                    }
                ],
                "phases": [
                    {
                        "name": "default",
                        "type": "concurrency",
                        "kind": "profiling",
                        "requests": 10,
                        "concurrency": 1,
                    }
                ],
                "slos": {"time_to_first_token": 500.0},
                "runtime": {"api_port": 8080},
            }
        )
        near_miss = {
            "type": "realtime_metrics",
            "metrics": [
                _metric_result("goodput", "Goodput", "req/s", current=19.2),
                _metric_result(
                    "request_count", "Request Count", "requests", current=1000.0
                ),
                _metric_result(
                    "good_request_count",
                    "Good Request Count",
                    "requests",
                    current=999.0,
                ),
            ],
        }
        with _run_server(cfg, extra_ws_payloads=[near_miss]) as base_url:
            _page.goto(f"{base_url}/dashboard-v2", wait_until="networkidle")
            _page.wait_for_function(
                "() => Array.from(document.querySelectorAll('.kpi-tile-label > span:first-child'))"
                "       .some(s => s.textContent.trim() === 'Goodput')",
                timeout=10_000,
            )
            state = _page.evaluate(
                """() => {
                    const tile = Array.from(document.querySelectorAll('.kpi-tile'))
                      .find(t => t.querySelector('.kpi-tile-label > span:first-child')?.textContent.trim() === 'Goodput');
                    return {
                      kind: Array.from(tile?.classList ?? [])
                              .find(c => c.startsWith('kpi-tile--slo-'))
                              ?.replace('kpi-tile--slo-', '') ?? null,
                      chip: tile?.querySelector('.kpi-chip')?.textContent?.trim() ?? null,
                    };
                }"""
            )
        # 999/1000 = 99.9% pass, but the goodput bar is "every single request".
        # Chip headlines the failed-request count so the size of the problem
        # is the first thing you see; the 99.9% lives in the sub-line.
        assert state["kind"] == "warn", state
        assert "1 failed" in (state["chip"] or ""), state

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_realtime_metrics_goodput_tile_reads_avg_when_no_current(
        self, _page: Page
    ) -> None:
        """Scalar counters/derived metrics (``good_request_count``,
        ``request_count``, ``goodput``) come off the real server with a value
        in ``avg`` and ``current=None`` — they're single-value scalars, not
        sliding-window stats. The tile must fall back to ``avg`` so the
        failed-count chip and pass-rate stay live during a run.
        """
        cfg = AIPerfConfig(
            benchmark={
                "models": ["llama3-8b"],
                "endpoint": {
                    "urls": ["http://srv:8000/v1/chat/completions"],
                    "type": "chat",
                },
                "datasets": [
                    {
                        "name": "default",
                        "type": "synthetic",
                        "entries": 10,
                        "prompts": {"isl": 128, "osl": 32},
                    }
                ],
                "phases": [
                    {
                        "name": "default",
                        "type": "concurrency",
                        "kind": "profiling",
                        "requests": 10,
                        "concurrency": 1,
                    }
                ],
                "slos": {"time_to_first_token": 500.0, "inter_token_latency": 30.0},
                "runtime": {"api_port": 8080},
            }
        )
        # Real-server shape: goodput/good_request_count/request_count populate
        # ``avg`` only. 997 / 1000 = 3 failed, pass rate 99.7%.
        payload = {
            "type": "realtime_metrics",
            "metrics": [
                _metric_result("goodput", "Goodput", "req/s", avg=19.1),
                _metric_result(
                    "request_count", "Request Count", "requests", avg=1000.0
                ),
                _metric_result(
                    "good_request_count", "Good Request Count", "requests", avg=997.0
                ),
            ],
        }
        with _run_server(cfg, extra_ws_payloads=[payload]) as base_url:
            _page.goto(f"{base_url}/dashboard-v2", wait_until="networkidle")
            _page.wait_for_function(
                "() => Array.from(document.querySelectorAll('.kpi-tile-label > span:first-child'))"
                "       .some(s => s.textContent.trim() === 'Goodput')",
                timeout=10_000,
            )
            state = _page.evaluate(
                """() => {
                    const tile = Array.from(document.querySelectorAll('.kpi-tile'))
                      .find(t => t.querySelector('.kpi-tile-label > span:first-child')?.textContent.trim() === 'Goodput');
                    return {
                      kind: Array.from(tile?.classList ?? [])
                              .find(c => c.startsWith('kpi-tile--slo-'))
                              ?.replace('kpi-tile--slo-', '') ?? null,
                      chip: tile?.querySelector('.kpi-chip')?.textContent?.trim() ?? null,
                      sub:  tile?.querySelector('.kpi-tile-sub')?.textContent?.trim().replace(/\\s+/g, ' ') ?? null,
                    };
                }"""
            )
        assert state["kind"] == "warn", state
        assert "3 failed" in (state["chip"] or ""), state
        assert "of 1,000" in (state["sub"] or ""), state

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_realtime_metrics_success_rate_tile_when_no_slos(self, _page: Page) -> None:
        """Without configured SLOs, the reliability tile falls back to
        Success Rate. The chip is green iff zero errors, warn otherwise —
        no fabricated pass-rate threshold.
        """
        payload = {
            "type": "realtime_metrics",
            "metrics": [
                _metric_result(
                    "request_count", "Request Count", "requests", current=1000.0
                ),
                _metric_result(
                    "error_request_count",
                    "Error Request Count",
                    "requests",
                    current=3.0,
                ),
                _metric_result(
                    "time_to_first_token",
                    "Time To First Token",
                    "ms",
                    current=140.0,
                    avg=120.0,
                    p99=220.0,
                ),
            ],
        }

        with _run_server(
            _build_multi_phase_cfg(), extra_ws_payloads=[payload]
        ) as base_url:
            _page.goto(f"{base_url}/dashboard-v2", wait_until="networkidle")
            _page.wait_for_function(
                "() => Array.from(document.querySelectorAll('.kpi-tile-label > span:first-child'))"
                "       .some(s => s.textContent.trim() === 'Success Rate')",
                timeout=10_000,
            )
            info = _page.evaluate(
                """() => {
                    const tile = Array.from(document.querySelectorAll('.kpi-tile'))
                      .find(t => t.querySelector('.kpi-tile-label > span:first-child')?.textContent.trim() === 'Success Rate');
                    return {
                      val:  tile?.querySelector('.kpi-big-val')?.textContent?.trim() ?? null,
                      kind: Array.from(tile?.classList ?? [])
                              .find(c => c.startsWith('kpi-tile--slo-'))
                              ?.replace('kpi-tile--slo-', '') ?? null,
                      chip: tile?.querySelector('.kpi-chip')?.textContent?.trim() ?? null,
                      sub:  tile?.querySelector('.kpi-tile-sub')?.textContent?.trim().replace(/\\s+/g, ' ') ?? null,
                    };
                }"""
            )

        # 3 / 1000 = 0.3% errors → 99.70% success → warn (any errors = warn).
        assert info["val"] == "99.70%", info
        assert info["kind"] == "warn", info
        # Chip text should say '3 errors', not a fake "≥ 99%" threshold.
        assert (
            "3" in (info["chip"] or "") and "error" in (info["chip"] or "").lower()
        ), info
        assert "3" in (info["sub"] or ""), info

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_realtime_metrics_success_rate_green_when_zero_errors(
        self, _page: Page
    ) -> None:
        """With zero errors and no SLOs, Success Rate is green with a
        '0 errors' chip — an objective fact, not a claim."""
        payload = {
            "type": "realtime_metrics",
            "metrics": [
                _metric_result(
                    "request_count", "Request Count", "requests", current=1000.0
                ),
                _metric_result(
                    "error_request_count",
                    "Error Request Count",
                    "requests",
                    current=0.0,
                ),
            ],
        }
        with _run_server(
            _build_multi_phase_cfg(), extra_ws_payloads=[payload]
        ) as base_url:
            _page.goto(f"{base_url}/dashboard-v2", wait_until="networkidle")
            _page.wait_for_function(
                "() => Array.from(document.querySelectorAll('.kpi-tile-label > span:first-child'))"
                "       .some(s => s.textContent.trim() === 'Success Rate')",
                timeout=10_000,
            )
            info = _page.evaluate(
                """() => {
                    const tile = Array.from(document.querySelectorAll('.kpi-tile'))
                      .find(t => t.querySelector('.kpi-tile-label > span:first-child')?.textContent.trim() === 'Success Rate');
                    return {
                      kind: Array.from(tile?.classList ?? [])
                              .find(c => c.startsWith('kpi-tile--slo-'))
                              ?.replace('kpi-tile--slo-', '') ?? null,
                      chip: tile?.querySelector('.kpi-chip')?.textContent?.trim() ?? null,
                    };
                }"""
            )
        assert info["kind"] == "good", info
        assert (
            "0" in (info["chip"] or "") and "error" in (info["chip"] or "").lower()
        ), info

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_realtime_metrics_card_hidden_without_data(self, _page: Page) -> None:
        """The KPI card must stay out of the DOM until at least one known metric
        lands, so the dashboard doesn't render a wall of ``---`` tiles at boot."""
        with _run_server(_build_multi_phase_cfg()) as base_url:
            _page.goto(f"{base_url}/dashboard-v2", wait_until="networkidle")
            _page.wait_for_selector("#config-bar.visible", timeout=10_000)
            _page.wait_for_timeout(400)
            tiles = _page.locator(".kpi-tile").count()

        assert tiles == 0, f"expected zero KPI tiles before data arrives, got {tiles}"


class TestDashboardV2GpuTelemetry:
    """``realtime_telemetry_metrics`` payloads must yield one card per
    ``(endpoint, gpu_index)`` parsed from the metric header."""

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_gpu_telemetry_groups_by_endpoint_and_index(self, _page: Page) -> None:
        """Two endpoints × two GPUs = four cards, each scoped to the right GPU."""

        def gpu(
            tag_base: str,
            header_name: str,
            endpoint: str,
            gpu_idx: int,
            uuid: str,
            unit: str,
            *,
            current: float,
            avg: float,
        ) -> dict[str, Any]:
            enc_ep = endpoint.replace(":", "_").replace(".", "_")
            return _metric_result(
                tag=f"{tag_base}_dcgm_http___{enc_ep}_metrics_gpu{gpu_idx}_{uuid}",
                header=f"{header_name} | {endpoint} | GPU {gpu_idx} | NVIDIA H100 80GB HBM3",
                unit=unit,
                current=current,
                avg=avg,
                p99=current,
            )

        metrics = []
        for endpoint, uuid_base in [
            ("node1:9401", "uuid-n1"),
            ("node2:9401", "uuid-n2"),
        ]:
            for gi in (0, 1):
                u = f"{uuid_base}-{gi}"
                load = 0.85 if (endpoint == "node1:9401" and gi == 0) else 0.60
                metrics += [
                    gpu(
                        "gpu_power_usage",
                        "GPU Power Usage",
                        endpoint,
                        gi,
                        u,
                        "W",
                        current=round(400 * load, 0),
                        avg=round(380 * load, 0),
                    ),
                    gpu(
                        "gpu_utilization",
                        "GPU Utilization",
                        endpoint,
                        gi,
                        u,
                        "%",
                        current=round(100 * load, 1),
                        avg=round(95 * load, 1),
                    ),
                    gpu(
                        "gpu_temperature",
                        "GPU Temperature",
                        endpoint,
                        gi,
                        u,
                        "C",
                        current=60 + round(18 * load),
                        avg=58 + round(17 * load),
                    ),
                    gpu(
                        "gpu_memory_used",
                        "GPU Memory Used",
                        endpoint,
                        gi,
                        u,
                        "GB",
                        current=round(48 * load, 1),
                        avg=round(47 * load, 1),
                    ),
                    # Extra metric to verify the "other" table populates too.
                    gpu(
                        "gpu_sm_clock",
                        "SM Clock",
                        endpoint,
                        gi,
                        u,
                        "MHz",
                        current=1620 if load > 0.8 else 1410,
                        avg=1580,
                    ),
                ]

        payload = {"type": "realtime_telemetry_metrics", "metrics": metrics}

        with _run_server(
            _build_multi_phase_cfg(), extra_ws_payloads=[payload]
        ) as base_url:
            _page.goto(f"{base_url}/dashboard-v2", wait_until="networkidle")
            _page.wait_for_selector("#config-bar.visible", timeout=10_000)
            _page.wait_for_function(
                "() => document.querySelectorAll('.gpu-card').length >= 4",
                timeout=10_000,
            )

            gpus = _page.evaluate(
                """() => Array.from(document.querySelectorAll('.gpu-card')).map(c => ({
                    header: c.querySelector('.gpu-header')?.textContent?.trim(),
                    tiles: Array.from(c.querySelectorAll('.gpu-tile')).map(t => ({
                        label: t.querySelector('.gpu-tile-label')?.textContent?.trim(),
                        val:   t.querySelector('.gpu-tile-val')?.textContent?.trim(),
                    })),
                    extra: Array.from(c.querySelectorAll('.gpu-extra tr')).map(
                        r => r.textContent.trim().replace(/\\s+/g, ' ')
                    ),
                }))"""
            )

        assert len(gpus) == 4, f"expected 4 GPU cards; got {len(gpus)}"

        # One card per (endpoint, gpu_index) pair; headers should contain both.
        headers = [g["header"] for g in gpus]
        expected_pairs = {
            ("node1:9401", 0),
            ("node1:9401", 1),
            ("node2:9401", 0),
            ("node2:9401", 1),
        }
        found_pairs = set()
        for h in headers:
            for ep, idx in expected_pairs:
                if ep in h and f"GPU {idx}" in h:
                    found_pairs.add((ep, idx))
        assert found_pairs == expected_pairs, (
            f"expected one card per GPU; got headers={headers!r}"
        )

        # Locate the hot GPU (node1 / GPU 0, load=0.85) and verify its primary
        # tiles carry the expected labels + display units.
        hot = next(
            g for g in gpus if "node1:9401" in g["header"] and "GPU 0" in g["header"]
        )
        labels = {t["label"]: t["val"] for t in hot["tiles"]}
        assert "Power" in labels and labels["Power"].endswith("W"), labels
        # Power at load=0.85 is round(400*0.85, 0) = 340 W.
        assert labels["Power"].startswith("340"), labels["Power"]
        assert "Utilization" in labels and labels["Utilization"].endswith("%"), labels
        assert "Temp" in labels and labels["Temp"].endswith("C"), labels
        assert "Memory" in labels and labels["Memory"].endswith("GB"), labels

        # SM Clock goes into the .gpu-extra table (not a primary tile).
        assert any("SM Clock" in row for row in hot["extra"]), hot["extra"]
        assert not any(t["label"] == "SM Clock" for t in hot["tiles"]), hot["tiles"]

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_gpu_telemetry_card_hidden_without_data(self, _page: Page) -> None:
        """No telemetry → the GPU section must stay out of the DOM entirely."""
        with _run_server(_build_multi_phase_cfg()) as base_url:
            _page.goto(f"{base_url}/dashboard-v2", wait_until="networkidle")
            _page.wait_for_selector("#config-bar.visible", timeout=10_000)
            _page.wait_for_timeout(400)
            cards = _page.locator(".gpu-card").count()

        assert cards == 0, (
            f"expected zero GPU cards before telemetry arrives, got {cards}"
        )


# -----------------------------------------------------------------------------
# v2: hero strip, sparklines, log severity, throughput-vs-latency chart
# -----------------------------------------------------------------------------


class TestDashboardV2HeroStrip:
    """The hero strip is the focal point of the live view.

    It answers three questions — "is my run healthy", "how much longer",
    "what's it doing" — from state that already exists, no new WS types
    required. These tests pin the three answers to realistic backend
    payloads.
    """

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_hero_health_green_when_all_slos_met(self, _page: Page) -> None:
        """Health = OK when every user SLO's p99 is at or under the user's
        threshold and no requests are failing."""
        cfg = AIPerfConfig(
            benchmark={
                "models": ["llama3-8b"],
                "endpoint": {
                    "urls": ["http://srv:8000/v1/chat/completions"],
                    "type": "chat",
                },
                "datasets": [
                    {
                        "name": "default",
                        "type": "synthetic",
                        "entries": 10,
                        "prompts": {"isl": 128, "osl": 32},
                    }
                ],
                "phases": [
                    {
                        "name": "default",
                        "type": "concurrency",
                        "kind": "profiling",
                        "requests": 1000,
                        "concurrency": 4,
                    }
                ],
                "slos": {"time_to_first_token": 500.0, "inter_token_latency": 30.0},
                "runtime": {"api_port": 8080},
            }
        )
        payload = [
            {
                "type": "credit_phase_start",
                "phase": "default",
                "stats": {
                    "start_ns": int(time.time_ns()) - int(10e9),
                    "total_expected_requests": 1000,
                },
            },
            {
                "type": "realtime_metrics",
                "metrics": [
                    _metric_result(
                        "time_to_first_token",
                        "Time To First Token",
                        "ms",
                        current=120.0,
                        avg=115.0,
                        p99=180.0,
                    ),
                    _metric_result(
                        "inter_token_latency",
                        "Inter Token Latency",
                        "ms",
                        current=12.0,
                        avg=11.5,
                        p99=22.0,
                    ),
                    _metric_result(
                        "request_count", "Request Count", "requests", current=100.0
                    ),
                    _metric_result(
                        "good_request_count",
                        "Good Request Count",
                        "requests",
                        current=100.0,
                    ),
                ],
            },
        ]

        with _run_server(cfg, extra_ws_payloads=payload) as base_url:
            _page.goto(f"{base_url}/dashboard-v2", wait_until="networkidle")
            _page.wait_for_selector(".hero--ok", timeout=10_000)
            label = _page.locator(".hero-health-label").text_content()
            reasons = _page.locator(".hero-health-reasons").text_content() or ""

        assert label and "target" in label.lower(), label
        assert "all declared SLOs passing" in reasons, reasons

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_hero_health_error_when_slo_violated(self, _page: Page) -> None:
        """SLO violation → hero turns red and spells out the violation."""
        cfg = AIPerfConfig(
            benchmark={
                "models": ["llama3-8b"],
                "endpoint": {
                    "urls": ["http://srv:8000/v1/chat/completions"],
                    "type": "chat",
                },
                "datasets": [
                    {
                        "name": "default",
                        "type": "synthetic",
                        "entries": 10,
                        "prompts": {"isl": 128, "osl": 32},
                    }
                ],
                "phases": [
                    {
                        "name": "default",
                        "type": "concurrency",
                        "kind": "profiling",
                        "requests": 100,
                        "concurrency": 4,
                    }
                ],
                "slos": {"time_to_first_token": 200.0},
                "runtime": {"api_port": 8080},
            }
        )
        payload = [
            {
                "type": "credit_phase_start",
                "phase": "default",
                "stats": {
                    "start_ns": int(time.time_ns()) - int(5e9),
                    "total_expected_requests": 100,
                },
            },
            {
                "type": "realtime_metrics",
                "metrics": [
                    _metric_result(
                        "time_to_first_token",
                        "Time To First Token",
                        "ms",
                        current=320.0,
                        avg=280.0,
                        p99=400.0,
                    ),  # 400 > 200
                ],
            },
        ]

        with _run_server(cfg, extra_ws_payloads=payload) as base_url:
            _page.goto(f"{base_url}/dashboard-v2", wait_until="networkidle")
            _page.wait_for_selector(".hero--error", timeout=10_000)
            reasons = _page.locator(".hero-health-reasons").text_content() or ""
            label = _page.locator(".hero-health-label").text_content() or ""

        assert "violated" in label.lower(), label
        # The violation reason should name the metric and include the
        # user's threshold so the customer doesn't have to cross-reference.
        assert "time_to_first_token" in reasons, reasons
        assert "200" in reasons, reasons

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_hero_shows_elapsed_eta_and_active_phase(self, _page: Page) -> None:
        """Elapsed + ETA compute from start_ns; active-phase progress bar
        shows the phase by name and completion pct."""
        five_s_ago_ns = int(time.time_ns()) - int(5e9)
        payload = [
            {
                "type": "credit_phase_start",
                "phase": "profiling",
                "stats": {
                    "start_ns": five_s_ago_ns,
                    "total_expected_requests": 1000,
                    "requests_completed": 250,
                },
            },
            {
                "type": "realtime_metrics",
                "metrics": [
                    _metric_result(
                        "request_throughput",
                        "Request Throughput",
                        "req/s",
                        current=50.0,
                        avg=49.0,
                        p99=52.0,
                    ),
                ],
            },
        ]

        with _run_server(
            _build_multi_phase_cfg(), extra_ws_payloads=payload
        ) as base_url:
            _page.goto(f"{base_url}/dashboard-v2", wait_until="networkidle")
            _page.wait_for_selector(".hero-phase-name", timeout=10_000)
            phase_name = _page.locator(".hero-phase-name").first.text_content()
            pct_text = _page.locator(".hero-phase-pct").first.text_content() or ""

            # Elapsed should be populated with a seconds value.
            elapsed_text = _page.evaluate(
                """() => document.querySelectorAll('.hero-clock-val')[0]?.textContent.trim()"""
            )
            eta_text = _page.evaluate(
                """() => document.querySelectorAll('.hero-clock-val')[1]?.textContent.trim()"""
            )

        assert phase_name == "profiling", phase_name
        # 250/1000 = 25%.
        assert pct_text.strip().startswith("25"), pct_text
        # Elapsed must contain a digit (seconds-scale number) and not be '--'.
        assert elapsed_text and elapsed_text != "--", elapsed_text
        assert any(ch.isdigit() for ch in elapsed_text), elapsed_text
        # ETA should also be populated (derived from rate), not the dim '—'.
        assert eta_text and eta_text != "—", eta_text


class TestDashboardV2Sparklines:
    """Each KPI tile has an inline sparkline driven by the rolling
    timeseries in ``lib/timeseries.js`` fed from successive
    ``realtime_metrics`` messages.
    """

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_sparklines_render_after_repeated_samples(self, _page: Page) -> None:
        """Two distinct realtime_metrics batches → each tile's sparkline
        must contain a polyline with at least two points."""

        def sample(ttft_p99):
            return {
                "type": "realtime_metrics",
                "metrics": [
                    _metric_result(
                        "request_throughput",
                        "Request Throughput",
                        "req/s",
                        current=20.0,
                        avg=20.0,
                        p99=21.0,
                    ),
                    _metric_result(
                        "output_token_throughput",
                        "Output Token Throughput",
                        "tok/s",
                        current=1800.0,
                        avg=1790.0,
                        p99=1900.0,
                    ),
                    _metric_result(
                        "time_to_first_token",
                        "Time To First Token",
                        "ms",
                        current=ttft_p99 * 0.7,
                        avg=ttft_p99 * 0.6,
                        p99=ttft_p99,
                    ),
                    _metric_result(
                        "inter_token_latency",
                        "Inter Token Latency",
                        "ms",
                        current=12.0,
                        avg=11.5,
                        p99=18.0,
                    ),
                    _metric_result(
                        "request_latency",
                        "Request Latency",
                        "ms",
                        current=800.0,
                        avg=760.0,
                        p99=900.0,
                    ),
                ],
            }

        with _run_server(
            _build_multi_phase_cfg(),
            extra_ws_payloads=[sample(150.0), sample(200.0), sample(180.0)],
        ) as base_url:
            _page.goto(f"{base_url}/dashboard-v2", wait_until="networkidle")
            _page.wait_for_function(
                "() => document.querySelectorAll('.kpi-tile').length >= 5",
                timeout=10_000,
            )
            # Wait until at least one sparkline path has two segments.
            _page.wait_for_function(
                """() => {
                    const paths = Array.from(document.querySelectorAll('.sparkline path'));
                    // Each path's `d` attribute for a >=2-point line contains an 'L' command.
                    return paths.some(p => (p.getAttribute('d') || '').includes('L'));
                }""",
                timeout=10_000,
            )

            info = _page.evaluate(
                """() => Array.from(document.querySelectorAll('.kpi-tile')).map(t => ({
                    label: t.querySelector('.kpi-tile-label > span:first-child')?.textContent?.trim(),
                    has_spark: !!t.querySelector('.sparkline path'),
                    d_len: (t.querySelector('.sparkline path[fill="none"]')?.getAttribute('d') || '').length,
                }))"""
            )

        for tile in info:
            if tile["label"] in (None, "Goodput", "Success Rate"):
                continue
            assert tile["has_spark"], f"no sparkline on tile {tile['label']!r}"
            # A 3-sample line has 2 L commands; path string length is non-trivial.
            assert tile["d_len"] > 10, tile


class TestDashboardV2LogPane:
    """The log pane now carries severity coloring + phase/worker/records
    categories, and lets the user narrow to warn/error only.
    """

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_log_records_phase_and_worker_events_with_severity(
        self, _page: Page
    ) -> None:
        """Phase-start and worker-error events must land in the log with
        distinct categories and severity classes."""
        payload = [
            # First push: group is healthy.
            {
                "type": "worker_group_stats",
                "service_id": "wgm-0",
                "group_id": "wgm-0",
                "status": "healthy",
                "task_stats": {"total": 0, "failed": 0, "completed": 0},
                "worker_statuses": {"w-alpha": "healthy"},
                "worker_startup_states": {},
                "worker_task_stats": {},
                "worker_health": {},
            },
            # Phase starts — info/phase.
            {
                "type": "credit_phase_start",
                "phase": "profiling",
                "stats": {
                    "start_ns": int(time.time_ns()) - int(1e9),
                    "total_expected_requests": 100,
                },
            },
            # Group flips to error — error/worker.
            {
                "type": "worker_group_stats",
                "service_id": "wgm-0",
                "group_id": "wgm-0",
                "status": "error",
                "task_stats": {"total": 0, "failed": 5, "completed": 0},
                "worker_statuses": {"w-alpha": "error"},
                "worker_startup_states": {},
                "worker_task_stats": {},
                "worker_health": {},
            },
        ]

        with _run_server(
            _build_multi_phase_cfg(), extra_ws_payloads=payload
        ) as base_url:
            _page.goto(f"{base_url}/dashboard-v2", wait_until="networkidle")
            _page.wait_for_function(
                "() => document.querySelectorAll('.log-entry--error').length >= 1",
                timeout=10_000,
            )

            entries = _page.evaluate(
                """() => Array.from(document.querySelectorAll('.log-entry')).map(e => ({
                    severity: Array.from(e.classList).find(c => c.startsWith('log-entry--'))?.replace('log-entry--',''),
                    cat: e.querySelector('.log-cat')?.textContent?.trim() ?? null,
                    msg: e.querySelector('.log-msg')?.textContent?.trim() ?? null,
                }))"""
            )

        has_phase_info = any(
            e["severity"] == "info"
            and e["cat"] == "phase"
            and "profiling" in (e["msg"] or "")
            for e in entries
        )
        has_worker_err = any(
            e["severity"] == "error" and e["cat"] == "worker" for e in entries
        )
        assert has_phase_info, f"missing phase-start info entry; got {entries!r}"
        assert has_worker_err, f"missing worker error entry; got {entries!r}"

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_log_filter_narrows_to_warn_plus(self, _page: Page) -> None:
        """Clicking the 'warn+' filter must hide info-only entries."""
        payload = [
            {
                "type": "worker_group_stats",
                "service_id": "wgm-0",
                "group_id": "wgm-0",
                "status": "healthy",
                "task_stats": {"total": 0, "failed": 0, "completed": 0},
                "worker_statuses": {"w-a": "healthy"},
                "worker_startup_states": {},
                "worker_task_stats": {},
                "worker_health": {},
            },
            {
                "type": "credit_phase_start",
                "phase": "default",
                "stats": {
                    "start_ns": int(time.time_ns()),
                    "total_expected_requests": 50,
                },
            },
            {
                "type": "worker_group_stats",
                "service_id": "wgm-0",
                "group_id": "wgm-0",
                "status": "high_load",
                "task_stats": {"total": 45, "failed": 0, "completed": 40},
                "worker_statuses": {"w-a": "high_load"},
                "worker_startup_states": {},
                "worker_task_stats": {},
                "worker_health": {},
            },
        ]

        with _run_server(
            _build_multi_phase_cfg(), extra_ws_payloads=payload
        ) as base_url:
            _page.goto(f"{base_url}/dashboard-v2", wait_until="networkidle")
            _page.wait_for_function(
                "() => document.querySelectorAll('.log-entry').length >= 2",
                timeout=10_000,
            )
            before = _page.locator(".log-entry").count()

            # Click the 'warn+' filter.
            _page.locator("button.log-filter", has_text="warn+").click()
            # Now only the high_load warning entry should be visible.
            _page.wait_for_function(
                """() => {
                    const visible = document.querySelectorAll('.log-entry').length;
                    const infos = document.querySelectorAll('.log-entry--info').length;
                    return visible >= 1 && infos === 0;
                }""",
                timeout=5_000,
            )
            after = _page.locator(".log-entry").count()

        assert before > after, (
            f"warn+ filter should reduce entries: before={before} after={after}"
        )


class TestDashboardV2ThroughputLatencyChart:
    """The live throughput-vs-latency chart must render a canvas with
    plotted datasets after multiple realtime samples arrive."""

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_chart_renders_after_multiple_samples(self, _page: Page) -> None:
        def sample(i):
            return {
                "type": "realtime_metrics",
                "metrics": [
                    _metric_result(
                        "request_throughput",
                        "Request Throughput",
                        "req/s",
                        current=18.0 + i,
                        avg=18.0 + i,
                        p99=20.0 + i,
                    ),
                    _metric_result(
                        "request_latency",
                        "Request Latency",
                        "ms",
                        current=400.0 + 20 * i,
                        avg=380.0,
                        p99=800.0 + 50 * i,
                    ),
                    _metric_result(
                        "time_to_first_token",
                        "Time To First Token",
                        "ms",
                        current=100.0 + 10 * i,
                        avg=95.0,
                        p99=150.0 + 20 * i,
                    ),
                ],
            }

        with _run_server(
            _build_multi_phase_cfg(),
            extra_ws_payloads=[sample(0), sample(1), sample(2), sample(3)],
        ) as base_url:
            _page.goto(f"{base_url}/dashboard-v2", wait_until="networkidle")
            _page.wait_for_selector(".chart-box canvas", timeout=10_000)
            # Wait for Chart.js to populate at least one dataset with points.
            _page.wait_for_function(
                """() => {
                    const canvases = Array.from(document.querySelectorAll('.chart-box canvas'));
                    // Chart.js registers the chart on window.Chart.instances (v4 uses Chart.getChart).
                    for (const c of canvases) {
                        const chart = window.Chart && window.Chart.getChart && window.Chart.getChart(c);
                        if (chart && chart.data.datasets.some(d => d.data.length >= 2)) return true;
                    }
                    return false;
                }""",
                timeout=10_000,
            )

            info = _page.evaluate(
                """() => {
                    const canvas = document.querySelector('.chart-box canvas');
                    const chart = window.Chart?.getChart?.(canvas);
                    if (!chart) return { labels: [], sizes: [] };
                    return {
                        labels: chart.data.datasets.map(d => d.label),
                        sizes: chart.data.datasets.map(d => d.data.length),
                    };
                }"""
            )

        # We fed three metrics; expect at least three labeled datasets.
        assert len(info["labels"]) >= 3, info
        assert all(s >= 2 for s in info["sizes"]), info
        label_joined = " | ".join(info["labels"]).lower()
        assert "req/s" in label_joined, info
        assert "ttft" in label_joined, info
        assert "latency" in label_joined, info
