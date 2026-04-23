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
        return (False, "playwright not installed (`uv pip install playwright pytest-playwright`)")
    # Check Chromium binary: the launch call is the authoritative test, but
    # failing fast here avoids a cryptic stacktrace if the browser is missing.
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
    except Exception as exc:  # noqa: BLE001 - one-shot probe, message surfaces via skip reason
        return (False, f"Chromium not launchable: {exc!s}. Run `uv run playwright install chromium`.")
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
        cfg=cfg,
        artifact_dir=Path("/tmp/aiperf-dashboard-test"),
    )
    svc = _StubAPIService(run)
    svc.app = app
    app.state.service = svc

    app.include_router(static_router)  # GET /, GET /dashboard, GET /dashboard-v2
    app.include_router(core_router)    # GET /api/config + health

    @app.get("/api/progress")
    async def _progress() -> dict[str, Any]:
        return {"phases": {}}

    @app.get("/api/server-metrics")
    async def _server_metrics() -> dict[str, Any]:
        return {"endpoint_summaries": []}

    # Pre-computed phase announcements (for the v2 tests that need to see the
    # PhaseCards component render an entry per configured phase name).
    phase_names = list(cfg.phases.keys())
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
                await websocket.send_text(
                    '{"type": "subscribed", "message_types": []}'
                )
                try:
                    parsed = json.loads(raw)
                except Exception:  # noqa: BLE001
                    parsed = {}
                if parsed.get("type") != "subscribe":
                    continue

                if broadcast_phases:
                    for name in phase_names:
                        await websocket.send_text(json.dumps({
                            "type": "credit_phase_start",
                            "phase": name,
                            "stats": {"start_ns": 1, "total_expected_requests": 100},
                        }))

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
        models=["llama3-8b", "llama3-70b"],
        endpoint={
            "urls": ["http://srv:8000/v1/chat/completions"],
            "type": "chat",
            "streaming": True,
            "api_key": "SHOULD_NOT_LEAK",
        },
        datasets={
            "default": {
                "type": "synthetic",
                "entries": 100,
                "prompts": {"isl": 128, "osl": 64},
            }
        },
        phases={
            "warmup": {"type": "concurrency", "requests": 50, "concurrency": 4},
            "profiling": {
                "type": "poisson",
                "rate": 20,
                "duration": 300,
                "concurrency": 32,
            },
        },
        runtime={"api_port": 8080},
    )


def _build_single_phase_cfg() -> AIPerfConfig:
    return AIPerfConfig(
        models=["gpt-4o-mini"],
        endpoint={
            "urls": ["http://srv:8000/v1/chat/completions"],
            "type": "chat",
        },
        datasets={
            "default": {
                "type": "synthetic",
                "entries": 50,
                "prompts": {"isl": 128, "osl": 32},
            }
        },
        phases={
            "default": {"type": "concurrency", "requests": 100, "concurrency": 8},
        },
        runtime={"api_port": 8080},
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
def _browser() -> "Iterator[Browser]":
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            browser.close()


@pytest.fixture
def _page(_browser: "Browser") -> "Iterator[Page]":
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
        _page: "Page",
        cfg_builder: "Any",
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
            f"unexpected browser console errors:\n  " + "\n  ".join(console_errors)
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
    def test_api_config_does_not_leak_api_key(self, _page: "Page") -> None:
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
        assert not failures, "v2 JS files failed node --check:\n" + "\n\n".join(failures)


class TestDashboardV2Render:
    """Drive a real Chromium browser against the v2 dashboard.

    The v2 app is a Preact/signals SPA served from ``/dashboard-v2`` with
    ES modules under ``/dashboard-v2/lib/*`` and ``/dashboard-v2/components/*``.
    """

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_v2_boots_and_shows_config_bar(self, _page: "Page") -> None:
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
            "Model", "llama3-8b", "llama3-70b",
            "Endpoint", "chat (streaming)",
            "URL", "http://srv:8000/v1/chat/completions",
            "warmup Type", "concurrency", "warmup Concurrency", "4",
            "profiling Type", "poisson", "profiling Rate", "20 QPS",
            "profiling Duration", "5m 0s",
        ]
        missing = [t for t in required if t not in config_text]
        assert not missing, (
            f"v2 config bar missing tokens: {missing}\nactual text: {config_text!r}"
        )
        assert "SHOULD_NOT_LEAK" not in config_text

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_v2_phases_keyed_by_name_not_collapsed(self, _page: "Page") -> None:
        """v2's PhaseCards keys on the backend phase name (fixes the v1 collapse bug).

        We push ``credit_phase_start`` for each configured phase via the
        stub WebSocket and assert that one phase card appears per name.
        """
        # Config has 3 phases with non-warmup-or-profiling names to prove
        # the v1 bucketing behavior is gone.
        cfg = AIPerfConfig(
            models=["llama3-8b"],
            endpoint={"urls": ["http://srv:8000/v1/chat/completions"], "type": "chat"},
            datasets={"default": {
                "type": "synthetic", "entries": 10,
                "prompts": {"isl": 128, "osl": 32},
            }},
            phases={
                "phase_alpha": {"type": "concurrency", "requests": 10, "concurrency": 1},
                "phase_beta":  {"type": "concurrency", "requests": 20, "concurrency": 2},
                "phase_gamma": {"type": "concurrency", "requests": 30, "concurrency": 3},
            },
            runtime={"api_port": 8080},
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
    def test_v2_serves_all_module_assets(self, _page: "Page") -> None:
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
        "p1": None, "p5": None, "p10": None, "p25": None,
        "p50": p50 if p50 is not None else avg,
        "p75": None, "p90": p99, "p95": None, "p99": p99,
        "min": None, "max": max, "std": None,
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
    thresholds the user declared via ``cfg.slos`` (the same dict AIPerf's
    goodput feature consumes). No fabricated "industry defaults" - silence
    is the honest option when the user hasn't said what good looks like.
    """

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_realtime_metrics_tiles_render_expected_values(self, _page: "Page") -> None:
        """Each hero tile must show its canonical primary stat + secondary stat.

        No ``cfg.slos`` is configured in the multi-phase fixture, so no chip
        should render on latency tiles - we're testing the stat-picker, not
        the threshold policy.
        """
        payload = {
            "type": "realtime_metrics",
            "metrics": [
                _metric_result("request_throughput",      "Request Throughput",      "req/s",
                               current=19.8, avg=20.1, p99=21.0),
                _metric_result("output_token_throughput", "Output Token Throughput", "tok/s",
                               current=1823.4, avg=1798.1, p99=1920.0),
                _metric_result("request_latency",         "Request Latency",         "ms",
                               current=482.3, avg=465.5, p99=812.0),
                _metric_result("time_to_first_token",     "Time To First Token",     "ms",
                               current=73.2, avg=68.7, p99=118.0),
                _metric_result("inter_token_latency",     "Inter Token Latency",     "ms",
                               current=12.1, avg=11.8, p99=21.4),
            ],
        }

        with _run_server(_build_multi_phase_cfg(), extra_ws_payloads=[payload]) as base_url:
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
        assert rt["slo_kind"] is None, "throughput is not an SLO metric by NIM convention"

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
        self, _page: "Page"
    ) -> None:
        """When the user declares ``cfg.slos``, the chip is binary pass/fail
        against that value and the chip label echoes the user's threshold.

        Customer: "I want TTFT p99 ≤ 100 ms, ITL avg ≤ 10 ms, Request
        Latency p99 ≤ 1500 ms." One run meets them all, one misses one,
        and we check both outcomes against the same SLO declaration.
        """
        cfg_with_slo = AIPerfConfig(
            models=["llama3-8b"],
            endpoint={"urls": ["http://srv:8000/v1/chat/completions"], "type": "chat"},
            datasets={"default": {
                "type": "synthetic", "entries": 10,
                "prompts": {"isl": 128, "osl": 32},
            }},
            phases={"default": {"type": "concurrency", "requests": 10, "concurrency": 1}},
            slos={
                "time_to_first_token": 100.0,
                "inter_token_latency": 10.0,
                "request_latency": 1500.0,
            },
            runtime={"api_port": 8080},
        )

        # Scenario A: all three pass → green chips with the user's thresholds.
        passing = {
            "type": "realtime_metrics",
            "metrics": [
                _metric_result("time_to_first_token", "Time To First Token", "ms",
                               current=80.0, avg=75.0, p99=92.0),    # ≤ 100 ✓
                _metric_result("inter_token_latency", "Inter Token Latency", "ms",
                               current=7.5, avg=7.8, p99=12.0),       # avg 7.8 ≤ 10 ✓
                _metric_result("request_latency",     "Request Latency",     "ms",
                               current=1100.0, avg=1050.0, p99=1420.0), # ≤ 1500 ✓
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
        assert pass_state["Request Latency"]["slo_kind"] == "good", pass_state["Request Latency"]
        assert "1500" in pass_state["Request Latency"]["chip_text"], pass_state["Request Latency"]

        # Scenario B: TTFT violates (p99=140 > 100). Others still pass.
        failing = {
            "type": "realtime_metrics",
            "metrics": [
                _metric_result("time_to_first_token", "Time To First Token", "ms",
                               current=130.0, avg=110.0, p99=140.0),  # > 100 ✗
                _metric_result("inter_token_latency", "Inter Token Latency", "ms",
                               current=7.5, avg=7.8, p99=12.0),
                _metric_result("request_latency",     "Request Latency",     "ms",
                               current=1100.0, avg=1050.0, p99=1420.0),
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
    def test_realtime_metrics_no_chip_without_user_slo(self, _page: "Page") -> None:
        """Absolute regression guard against fabricated defaults: no chip may
        appear on any latency tile when ``cfg.slos`` does not cover it.

        Tile renders values + secondary stat, but no pass/fail judgment —
        the dashboard does not claim to know whether 500 ms TTFT is "good"
        for the customer's use case.
        """
        payload = {
            "type": "realtime_metrics",
            "metrics": [
                _metric_result("time_to_first_token", "Time To First Token", "ms",
                               current=480.0, avg=450.0, p99=720.0),
                _metric_result("request_latency",     "Request Latency",     "ms",
                               current=9000.0, avg=8500.0, p99=11000.0),
                _metric_result("inter_token_latency", "Inter Token Latency", "ms",
                               current=95.0, avg=90.0, p99=180.0),
            ],
        }

        with _run_server(_build_multi_phase_cfg(), extra_ws_payloads=[payload]) as base_url:
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
        self, _page: "Page"
    ) -> None:
        """Goodput tile is green iff every request met every user SLO.

        Binary policy ties the reliability chip to the user's own
        declarations rather than a fabricated pass-rate band.
        """
        cfg = AIPerfConfig(
            models=["llama3-8b"],
            endpoint={"urls": ["http://srv:8000/v1/chat/completions"], "type": "chat"},
            datasets={"default": {
                "type": "synthetic", "entries": 10,
                "prompts": {"isl": 128, "osl": 32},
            }},
            phases={"default": {"type": "concurrency", "requests": 10, "concurrency": 1}},
            slos={"time_to_first_token": 500.0, "inter_token_latency": 30.0},
            runtime={"api_port": 8080},
        )

        # 100% passes → green.
        perfect = {
            "type": "realtime_metrics",
            "metrics": [
                _metric_result("goodput", "Goodput", "req/s", current=19.2),
                _metric_result("request_count", "Request Count", "requests", current=1000.0),
                _metric_result("good_request_count", "Good Request Count", "requests", current=1000.0),
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
        self, _page: "Page"
    ) -> None:
        """Any user-SLO failure → warn. No fake band; binary at the user's bar."""
        cfg = AIPerfConfig(
            models=["llama3-8b"],
            endpoint={"urls": ["http://srv:8000/v1/chat/completions"], "type": "chat"},
            datasets={"default": {
                "type": "synthetic", "entries": 10,
                "prompts": {"isl": 128, "osl": 32},
            }},
            phases={"default": {"type": "concurrency", "requests": 10, "concurrency": 1}},
            slos={"time_to_first_token": 500.0},
            runtime={"api_port": 8080},
        )
        near_miss = {
            "type": "realtime_metrics",
            "metrics": [
                _metric_result("goodput", "Goodput", "req/s", current=19.2),
                _metric_result("request_count", "Request Count", "requests", current=1000.0),
                _metric_result("good_request_count", "Good Request Count", "requests", current=999.0),
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
        self, _page: "Page"
    ) -> None:
        """Scalar counters/derived metrics (``good_request_count``,
        ``request_count``, ``goodput``) come off the real server with a value
        in ``avg`` and ``current=None`` — they're single-value scalars, not
        sliding-window stats. The tile must fall back to ``avg`` so the
        failed-count chip and pass-rate stay live during a run.
        """
        cfg = AIPerfConfig(
            models=["llama3-8b"],
            endpoint={"urls": ["http://srv:8000/v1/chat/completions"], "type": "chat"},
            datasets={"default": {
                "type": "synthetic", "entries": 10,
                "prompts": {"isl": 128, "osl": 32},
            }},
            phases={"default": {"type": "concurrency", "requests": 10, "concurrency": 1}},
            slos={"time_to_first_token": 500.0, "inter_token_latency": 30.0},
            runtime={"api_port": 8080},
        )
        # Real-server shape: goodput/good_request_count/request_count populate
        # ``avg`` only. 997 / 1000 = 3 failed, pass rate 99.7%.
        payload = {
            "type": "realtime_metrics",
            "metrics": [
                _metric_result("goodput", "Goodput", "req/s", avg=19.1),
                _metric_result("request_count", "Request Count", "requests", avg=1000.0),
                _metric_result("good_request_count", "Good Request Count", "requests", avg=997.0),
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
    def test_realtime_metrics_success_rate_tile_when_no_slos(self, _page: "Page") -> None:
        """Without configured SLOs, the reliability tile falls back to
        Success Rate. The chip is green iff zero errors, warn otherwise —
        no fabricated pass-rate threshold.
        """
        payload = {
            "type": "realtime_metrics",
            "metrics": [
                _metric_result("request_count", "Request Count", "requests",
                               current=1000.0),
                _metric_result("error_request_count", "Error Request Count", "requests",
                               current=3.0),
                _metric_result("time_to_first_token", "Time To First Token", "ms",
                               current=140.0, avg=120.0, p99=220.0),
            ],
        }

        with _run_server(_build_multi_phase_cfg(), extra_ws_payloads=[payload]) as base_url:
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
        assert "3" in (info["chip"] or "") and "error" in (info["chip"] or "").lower(), info
        assert "3" in (info["sub"] or ""), info

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_realtime_metrics_success_rate_green_when_zero_errors(
        self, _page: "Page"
    ) -> None:
        """With zero errors and no SLOs, Success Rate is green with a
        '0 errors' chip — an objective fact, not a claim."""
        payload = {
            "type": "realtime_metrics",
            "metrics": [
                _metric_result("request_count", "Request Count", "requests",
                               current=1000.0),
                _metric_result("error_request_count", "Error Request Count", "requests",
                               current=0.0),
            ],
        }
        with _run_server(_build_multi_phase_cfg(), extra_ws_payloads=[payload]) as base_url:
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
        assert "0" in (info["chip"] or "") and "error" in (info["chip"] or "").lower(), info

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_realtime_metrics_card_hidden_without_data(self, _page: "Page") -> None:
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
    def test_gpu_telemetry_groups_by_endpoint_and_index(self, _page: "Page") -> None:
        """Two endpoints × two GPUs = four cards, each scoped to the right GPU."""
        def gpu(tag_base: str, header_name: str, endpoint: str, gpu_idx: int,
                uuid: str, unit: str, *, current: float, avg: float) -> dict[str, Any]:
            enc_ep = endpoint.replace(":", "_").replace(".", "_")
            return _metric_result(
                tag=f"{tag_base}_dcgm_http___{enc_ep}_metrics_gpu{gpu_idx}_{uuid}",
                header=f"{header_name} | {endpoint} | GPU {gpu_idx} | NVIDIA H100 80GB HBM3",
                unit=unit,
                current=current, avg=avg, p99=current,
            )

        metrics = []
        for endpoint, uuid_base in [("node1:9401", "uuid-n1"), ("node2:9401", "uuid-n2")]:
            for gi in (0, 1):
                u = f"{uuid_base}-{gi}"
                load = 0.85 if (endpoint == "node1:9401" and gi == 0) else 0.60
                metrics += [
                    gpu("gpu_power_usage",  "GPU Power Usage",  endpoint, gi, u, "W",
                        current=round(400 * load, 0), avg=round(380 * load, 0)),
                    gpu("gpu_utilization",  "GPU Utilization",  endpoint, gi, u, "%",
                        current=round(100 * load, 1), avg=round(95 * load, 1)),
                    gpu("gpu_temperature",  "GPU Temperature",  endpoint, gi, u, "C",
                        current=60 + round(18 * load), avg=58 + round(17 * load)),
                    gpu("gpu_memory_used",  "GPU Memory Used",  endpoint, gi, u, "GB",
                        current=round(48 * load, 1), avg=round(47 * load, 1)),
                    # Extra metric to verify the "other" table populates too.
                    gpu("gpu_sm_clock",     "SM Clock",         endpoint, gi, u, "MHz",
                        current=1620 if load > 0.8 else 1410, avg=1580),
                ]

        payload = {"type": "realtime_telemetry_metrics", "metrics": metrics}

        with _run_server(_build_multi_phase_cfg(), extra_ws_payloads=[payload]) as base_url:
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
            ("node1:9401", 0), ("node1:9401", 1),
            ("node2:9401", 0), ("node2:9401", 1),
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
        hot = next(g for g in gpus
                   if "node1:9401" in g["header"] and "GPU 0" in g["header"])
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
    def test_gpu_telemetry_card_hidden_without_data(self, _page: "Page") -> None:
        """No telemetry → the GPU section must stay out of the DOM entirely."""
        with _run_server(_build_multi_phase_cfg()) as base_url:
            _page.goto(f"{base_url}/dashboard-v2", wait_until="networkidle")
            _page.wait_for_selector("#config-bar.visible", timeout=10_000)
            _page.wait_for_timeout(400)
            cards = _page.locator(".gpu-card").count()

        assert cards == 0, f"expected zero GPU cards before telemetry arrives, got {cards}"


# -----------------------------------------------------------------------------
# v2: hero strip, sparklines, log severity, server-metrics saturation,
# throughput-vs-latency chart
# -----------------------------------------------------------------------------


class TestDashboardV2HeroStrip:
    """The hero strip is the focal point of the live view.

    It answers three questions — "is my run healthy", "how much longer",
    "what's it doing" — from state that already exists, no new WS types
    required. These tests pin the three answers to realistic backend
    payloads.
    """

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_hero_health_green_when_all_slos_met(self, _page: "Page") -> None:
        """Health = OK when every user SLO's p99 is at or under the user's
        threshold and no requests are failing."""
        cfg = AIPerfConfig(
            models=["llama3-8b"],
            endpoint={"urls": ["http://srv:8000/v1/chat/completions"], "type": "chat"},
            datasets={"default": {"type": "synthetic", "entries": 10,
                                  "prompts": {"isl": 128, "osl": 32}}},
            phases={"default": {"type": "concurrency", "requests": 1000,
                                "concurrency": 4}},
            slos={"time_to_first_token": 500.0, "inter_token_latency": 30.0},
            runtime={"api_port": 8080},
        )
        payload = [
            {
                "type": "credit_phase_start",
                "phase": "default",
                "stats": {"start_ns": int(time.time_ns()) - int(10e9),
                          "total_expected_requests": 1000},
            },
            {
                "type": "realtime_metrics",
                "metrics": [
                    _metric_result("time_to_first_token", "Time To First Token", "ms",
                                   current=120.0, avg=115.0, p99=180.0),
                    _metric_result("inter_token_latency", "Inter Token Latency", "ms",
                                   current=12.0, avg=11.5, p99=22.0),
                    _metric_result("request_count", "Request Count", "requests", current=100.0),
                    _metric_result("good_request_count", "Good Request Count",
                                   "requests", current=100.0),
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
    def test_hero_health_error_when_slo_violated(self, _page: "Page") -> None:
        """SLO violation → hero turns red and spells out the violation."""
        cfg = AIPerfConfig(
            models=["llama3-8b"],
            endpoint={"urls": ["http://srv:8000/v1/chat/completions"], "type": "chat"},
            datasets={"default": {"type": "synthetic", "entries": 10,
                                  "prompts": {"isl": 128, "osl": 32}}},
            phases={"default": {"type": "concurrency", "requests": 100, "concurrency": 4}},
            slos={"time_to_first_token": 200.0},
            runtime={"api_port": 8080},
        )
        payload = [
            {
                "type": "credit_phase_start", "phase": "default",
                "stats": {"start_ns": int(time.time_ns()) - int(5e9),
                          "total_expected_requests": 100},
            },
            {
                "type": "realtime_metrics",
                "metrics": [
                    _metric_result("time_to_first_token", "Time To First Token", "ms",
                                   current=320.0, avg=280.0, p99=400.0),  # 400 > 200
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
    def test_hero_shows_elapsed_eta_and_active_phase(self, _page: "Page") -> None:
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
                    _metric_result("request_throughput", "Request Throughput", "req/s",
                                   current=50.0, avg=49.0, p99=52.0),
                ],
            },
        ]

        with _run_server(_build_multi_phase_cfg(), extra_ws_payloads=payload) as base_url:
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
    def test_sparklines_render_after_repeated_samples(self, _page: "Page") -> None:
        """Two distinct realtime_metrics batches → each tile's sparkline
        must contain a polyline with at least two points."""
        sample = lambda ttft_p99: {
            "type": "realtime_metrics",
            "metrics": [
                _metric_result("request_throughput", "Request Throughput", "req/s",
                               current=20.0, avg=20.0, p99=21.0),
                _metric_result("output_token_throughput", "Output Token Throughput", "tok/s",
                               current=1800.0, avg=1790.0, p99=1900.0),
                _metric_result("time_to_first_token", "Time To First Token", "ms",
                               current=ttft_p99 * 0.7, avg=ttft_p99 * 0.6, p99=ttft_p99),
                _metric_result("inter_token_latency", "Inter Token Latency", "ms",
                               current=12.0, avg=11.5, p99=18.0),
                _metric_result("request_latency", "Request Latency", "ms",
                               current=800.0, avg=760.0, p99=900.0),
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
        self, _page: "Page"
    ) -> None:
        """Phase-start and worker-error events must land in the log with
        distinct categories and severity classes."""
        payload = [
            # First push establishes a worker in healthy state.
            {
                "type": "worker_health", "worker_id": "w-alpha",
                "status": "healthy", "in_flight": 0, "completed": 0, "failed": 0,
            },
            # Phase starts — info/phase.
            {
                "type": "credit_phase_start", "phase": "profiling",
                "stats": {"start_ns": int(time.time_ns()) - int(1e9),
                          "total_expected_requests": 100},
            },
            # Worker flips to error — error/worker.
            {
                "type": "worker_health", "worker_id": "w-alpha",
                "status": "error", "in_flight": 0, "completed": 0, "failed": 5,
            },
        ]

        with _run_server(_build_multi_phase_cfg(), extra_ws_payloads=payload) as base_url:
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
            e["severity"] == "info" and e["cat"] == "phase"
            and "profiling" in (e["msg"] or "")
            for e in entries
        )
        has_worker_err = any(
            e["severity"] == "error" and e["cat"] == "worker" for e in entries
        )
        assert has_phase_info, f"missing phase-start info entry; got {entries!r}"
        assert has_worker_err, f"missing worker error entry; got {entries!r}"

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_log_filter_narrows_to_warn_plus(self, _page: "Page") -> None:
        """Clicking the 'warn+' filter must hide info-only entries."""
        payload = [
            {"type": "worker_health", "worker_id": "w-a",
             "status": "healthy", "in_flight": 0, "completed": 0, "failed": 0},
            {"type": "credit_phase_start", "phase": "default",
             "stats": {"start_ns": int(time.time_ns()), "total_expected_requests": 50}},
            {"type": "worker_health", "worker_id": "w-a",
             "status": "high_load", "in_flight": 5, "completed": 40, "failed": 0},
        ]

        with _run_server(_build_multi_phase_cfg(), extra_ws_payloads=payload) as base_url:
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

        assert before > after, f"warn+ filter should reduce entries: before={before} after={after}"


class TestDashboardV2ServerMetricsContext:
    """KV-cache utilization and queue depth rows gain saturation bands +
    tooltips so raw numbers aren't left uninterpreted."""

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_server_metrics_saturation_classes(self, _page: "Page") -> None:
        """kv_cache_utilization 0.45 → good, 0.80 → warn, 0.95 → bad.
           queue_depth 3 → good, 20 → warn, 100 → bad."""
        payload = {
            "type": "realtime_server_metrics",
            "endpoint_summaries": [{"endpoint": "http://srv:8000", "metrics": [
                {"name": "kv_cache_utilization", "value": 0.95, "unit": "ratio"},
                {"name": "queue_depth", "value": 3, "unit": "requests"},
                {"name": "batch_size_avg", "value": 22.5, "unit": "requests"},
            ]}],
        }

        with _run_server(_build_multi_phase_cfg(), extra_ws_payloads=[payload]) as base_url:
            _page.goto(f"{base_url}/dashboard-v2", wait_until="networkidle")
            _page.wait_for_selector(".server-metrics", timeout=10_000)
            _page.wait_for_function(
                """() => document.querySelectorAll('.server-metrics-row--bad').length >= 1""",
                timeout=5_000,
            )

            rows = _page.evaluate(
                """() => Array.from(document.querySelectorAll('.server-metrics tbody tr')).map(r => ({
                    name: r.cells[0]?.textContent?.trim(),
                    kind: Array.from(r.classList).find(c => c.startsWith('server-metrics-row--'))?.replace('server-metrics-row--','') ?? null,
                    chip: r.querySelector('.server-chip')?.textContent?.trim() ?? null,
                    tooltip: r.getAttribute('title') ?? '',
                }))"""
            )

        by_name = {r["name"]: r for r in rows}
        # KV cache at 0.95 → saturated (bad).
        kv = by_name["kv_cache_utilization"]
        assert kv["kind"] == "bad", kv
        assert kv["chip"] == "saturated", kv
        assert "saturat" in (kv["tooltip"] or "").lower(), kv

        # queue_depth at 3 → good / headroom.
        qd = by_name["queue_depth"]
        assert qd["kind"] == "good", qd
        assert "headroom" in (qd["chip"] or "").lower(), qd

        # Metrics with no guardrail must have no class or chip.
        bs = by_name["batch_size_avg"]
        assert bs["kind"] is None, bs
        assert bs["chip"] is None or bs["chip"] == "", bs


class TestDashboardV2ThroughputLatencyChart:
    """The live throughput-vs-latency chart must render a canvas with
    plotted datasets after multiple realtime samples arrive."""

    @pytest.mark.skipif(not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON)
    def test_chart_renders_after_multiple_samples(self, _page: "Page") -> None:
        sample = lambda i: {
            "type": "realtime_metrics",
            "metrics": [
                _metric_result("request_throughput", "Request Throughput", "req/s",
                               current=18.0 + i, avg=18.0 + i, p99=20.0 + i),
                _metric_result("request_latency", "Request Latency", "ms",
                               current=400.0 + 20 * i, avg=380.0, p99=800.0 + 50 * i),
                _metric_result("time_to_first_token", "Time To First Token", "ms",
                               current=100.0 + 10 * i, avg=95.0, p99=150.0 + 20 * i),
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
