#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end live-dashboard smoke test.

What this script does, in order:

1. Picks two random free ports — one for the aiperf mock server, one for
   the ``aiperf profile`` API server.
2. Spawns ``aiperf_mock_server`` as a subprocess on the mock port.
3. Waits for the mock to answer ``/health``.
4. Spawns ``aiperf profile`` against the mock, with
   ``AIPERF_API_SERVER_PORT=<api_port>`` in the environment and a
   user-declared goodput SLO set so the dashboard's SLO chips light up.
5. Polls ``http://127.0.0.1:<api_port>/healthz`` until the API is live.
6. Drives a headless Chromium via Playwright to ``/dashboard-v2/``,
   waits for the real live data to flow (sparklines, phase progress,
   chart), inspects the DOM, and saves screenshots.
7. Prints an inspection report (SLO chip state, phase pct, goodput %,
   hero health) alongside the paths of the captured screenshots.
8. Cleans up both subprocesses.

Run with ``uv run --no-sync python tools/live_dashboard_e2e.py`` from
the repo root.

Requirements: playwright + Chromium installed in the active venv
(``uv pip install playwright && uv run playwright install chromium``);
``aiperf_mock_server`` installed (comes from ``tests/aiperf_mock_server``).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCREENSHOT_DIR = Path("/tmp/aiperf-live-e2e")


# ───────────────────────── helpers ─────────────────────────


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_http(url: str, *, deadline: float, label: str,
                  ok_status: int | None = 200,
                  accept_any_response: bool = False) -> bool:
    """Poll ``url`` until it returns ``ok_status`` or the deadline passes.

    If ``accept_any_response`` is True, *any* HTTP response code (including
    4xx/5xx) counts as success — useful for "is the server listening at
    all" checks where the specific endpoint may reject our verb but is
    nonetheless proof the port is bound.
    """
    import urllib.error
    import urllib.request

    last_err: str | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as resp:
                if accept_any_response or (ok_status is not None and resp.status == ok_status):
                    return True
                last_err = f"HTTP {resp.status}"
        except urllib.error.HTTPError as e:
            if accept_any_response:
                return True
            last_err = f"HTTP {e.code}"
        except Exception as e:  # noqa: BLE001 - polling loop
            last_err = type(e).__name__
        time.sleep(0.4)
    print(f"[e2e] {label}: gave up waiting for {url} ({last_err})", flush=True)
    return False


def tail_file(path: Path, label: str, lines: int = 20) -> None:
    """Print the last few lines of a subprocess log file."""
    if not path.is_file():
        return
    try:
        content = path.read_text(errors="replace")
    except Exception:  # noqa: BLE001
        return
    if content:
        tail = "\n".join(content.splitlines()[-lines:])
        print(f"[e2e] --- {label} tail ({lines} lines) ---\n{tail}\n[e2e] ---", flush=True)


def terminate(proc: subprocess.Popen, label: str, timeout: float = 5.0) -> None:
    """SIGINT → SIGTERM → SIGKILL escalation."""
    if proc.poll() is not None:
        return
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            print(f"[e2e] {label}: {sig.name} did not stop process, escalating", flush=True)


# ───────────────────────── mock server ─────────────────────────


def start_mock_server(port: int, log_path: Path) -> subprocess.Popen:
    """Launch the aiperf mock server on ``port``.

    Its stdout/stderr go to ``log_path`` — never a PIPE — so the subprocess
    can't block on a full OS pipe buffer.
    """
    cmd = [
        sys.executable, "-m", "aiperf_mock_server",
        "--host", "127.0.0.1",
        "--port", str(port),
    ]
    print(f"[e2e] starting mock server: {' '.join(cmd)}", flush=True)
    print(f"[e2e]   log → {log_path}", flush=True)
    log_fh = log_path.open("wb")
    return subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )


# ───────────────────────── aiperf benchmark ─────────────────────────


def start_aiperf(
    *,
    mock_port: int,
    api_port: int,
    model: str,
    duration_sec: int,
    request_rate: float,
    concurrency: int,
    warmup: int,
    isl: int,
    osl: int,
    goodput: str,
    artifact_dir: Path,
    log_path: Path,
) -> subprocess.Popen:
    """Launch ``aiperf profile`` with the API server exposed on ``api_port``.

    Uses a fixed-rate (poisson) phase for ``duration_sec`` seconds so the
    benchmark takes a predictable amount of wall time — long enough for
    the headless browser to navigate, watch sparklines fill in, and
    capture a screenshot while the run is still live.
    """
    aiperf = shutil.which("aiperf") or "aiperf"
    cmd = [
        aiperf, "profile",
        "--model", model,
        "--url", f"http://127.0.0.1:{mock_port}",
        "--endpoint-type", "chat",
        "--streaming",
        "--request-rate", str(request_rate),
        # Uses the shorthand string form — "40s" — to exercise the
        # duration parser (helped shake out a pydantic float-only bug).
        "--benchmark-duration", f"{duration_sec}s",
        "--concurrency", str(concurrency),
        "--warmup-request-count", str(warmup),
        "--isl", str(isl),
        "--osl", str(osl),
        "--tokenizer", "builtin",
        "--random-seed", "42",
        "--ui", "simple",
        "--goodput", goodput,
        "--artifact-dir", str(artifact_dir),
    ]
    env = os.environ.copy()
    env["AIPERF_API_SERVER_PORT"] = str(api_port)
    env.setdefault("AIPERF_API_SERVER_HOST", "127.0.0.1")
    # Force realtime_metrics broadcasting even with --ui simple. The
    # records_manager's internal guard only publishes when ui=dashboard,
    # runtime.api_port is set, or this env var is true. AIPERF_API_SERVER_PORT
    # alone doesn't set runtime.api_port, so flip this bit explicitly.
    env.setdefault("AIPERF_UI_REALTIME_METRICS_ENABLED", "true")
    env.setdefault("AIPERF_LOG_LEVEL", "warning")
    print(f"[e2e] starting aiperf: AIPERF_API_SERVER_PORT={api_port}", flush=True)
    print(f"[e2e]   {' '.join(cmd)}", flush=True)
    print(f"[e2e]   log → {log_path}", flush=True)
    log_fh = log_path.open("wb")
    return subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
    )


# ───────────────────────── playwright inspection ─────────────────────────


def inspect_with_playwright(
    *,
    api_port: int,
    shots_dir: Path,
    wait_for_samples: int = 4,
    wait_for_chart_pts: int = 3,
    nav_timeout_ms: int = 45_000,
    settle_after_ms: int = 1200,
) -> dict[str, Any]:
    """Navigate the real /dashboard-v2/ and return a DOM-state report."""
    from playwright.sync_api import sync_playwright

    shots_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{api_port}"
    result: dict[str, Any] = {
        "base_url": base_url,
        "screenshots": [],
        "console": [],
        "dom": {},
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1400, "height": 2000})
        page = ctx.new_page()
        page.on("console", lambda m: result["console"].append(f"[{m.type}] {m.text[:200]}"))
        page.on("pageerror", lambda e: result["console"].append(f"[pageerror] {str(e)[:200]}"))

        page.goto(f"{base_url}/dashboard-v2/")
        # Take an initial screenshot of the page in whatever state we find it.
        early = shots_dir / "01-initial.png"
        page.screenshot(path=str(early), full_page=True)
        result["screenshots"].append(str(early))

        # Wait until the live data has flowed in: hero, at least one phase card,
        # KPI tiles populated, multi-sample sparklines, chart with enough
        # datapoints to show a line.
        try:
            page.wait_for_selector(".hero", timeout=nav_timeout_ms)
            page.wait_for_function(
                "() => document.querySelectorAll('.phase-card').length >= 1",
                timeout=nav_timeout_ms,
            )
            page.wait_for_function(
                "() => document.querySelectorAll('.kpi-tile').length >= 3",
                timeout=nav_timeout_ms,
            )
            page.wait_for_function(
                f"() => document.querySelectorAll('.sparkline path').length >= 3 "
                f"&& window.Chart && document.querySelector('.chart-box canvas') "
                f"&& window.Chart.getChart(document.querySelector('.chart-box canvas'))"
                f"?.data?.datasets?.some(d => d.data.length >= {wait_for_chart_pts})",
                timeout=nav_timeout_ms * 2,
            )
            page.wait_for_timeout(settle_after_ms)
        except Exception as e:  # noqa: BLE001 - test failure is recorded, not raised
            result["wait_error"] = f"{type(e).__name__}: {e}"

        # Full capture of what the user actually sees during a live run.
        live = shots_dir / "02-live.png"
        page.screenshot(path=str(live), full_page=True)
        result["screenshots"].append(str(live))

        # Inspect DOM state — the script's actual job is to prove the
        # dashboard rendered meaningful live data, not just that some
        # HTML was served.
        result["dom"] = page.evaluate(
            """() => {
              const kpi = Array.from(document.querySelectorAll('.kpi-tile')).map(t => ({
                label: t.querySelector('.kpi-tile-label > span:first-child')?.textContent?.trim(),
                primary_stat: t.querySelector('.kpi-tile-primary-stat')?.textContent?.trim(),
                val: t.querySelector('.kpi-big-val')?.textContent?.trim(),
                unit: t.querySelector('.kpi-big-unit')?.textContent?.trim() ?? '',
                sub: t.querySelector('.kpi-tile-sub')?.textContent?.trim().replace(/\\s+/g, ' '),
                slo_kind: Array.from(t.classList)
                  .find(c => c.startsWith('kpi-tile--slo-'))
                  ?.replace('kpi-tile--slo-', '') ?? null,
                chip_text: t.querySelector('.kpi-chip')?.textContent?.trim().replace(/\\s+/g, ' ') ?? null,
                sparkline_points: t.querySelectorAll('.sparkline path').length,
              }));
              const phases = Array.from(document.querySelectorAll('.phase-card')).map(c => ({
                name: c.querySelector('.phase-name')?.textContent?.trim(),
                badge: c.querySelector('.phase-badge')?.textContent?.trim(),
                progress: c.querySelector('.phase-stat-val')?.textContent?.trim(),
              }));
              const chart = (() => {
                const cv = document.querySelector('.chart-box canvas');
                const c = cv && window.Chart?.getChart?.(cv);
                if (!c) return null;
                return {
                  labels: c.data.datasets.map(d => d.label),
                  sample_counts: c.data.datasets.map(d => d.data.length),
                };
              })();
              const hero = document.querySelector('.hero');
              const heroKind = hero
                ? Array.from(hero.classList).find(cl => cl.startsWith('hero--'))?.replace('hero--','') ?? null
                : null;
              return {
                status_connected: !!document.querySelector('.status-dot.connected'),
                hero: {
                  kind: heroKind,
                  label: document.querySelector('.hero-health-label')?.textContent?.trim() ?? null,
                  reasons: document.querySelector('.hero-health-reasons')?.textContent?.trim() ?? null,
                  elapsed: document.querySelectorAll('.hero-clock-val')[0]?.textContent?.trim() ?? null,
                  eta: document.querySelectorAll('.hero-clock-val')[1]?.textContent?.trim() ?? null,
                  phase_name: document.querySelector('.hero-phase-name')?.textContent?.trim() ?? null,
                  phase_pct: document.querySelector('.hero-phase-pct')?.textContent?.trim() ?? null,
                },
                kpi, phases, chart,
                server_metrics: Array.from(document.querySelectorAll('.server-metrics tbody tr')).map(r => ({
                  name: r.cells[0]?.textContent?.trim(),
                  value: r.cells[1]?.textContent?.trim(),
                  saturation: Array.from(r.classList)
                    .find(c => c.startsWith('server-metrics-row--'))
                    ?.replace('server-metrics-row--','') ?? null,
                })),
                log_entries: Array.from(document.querySelectorAll('.log-entry')).length,
              };
            }"""
        )

        browser.close()

    return result


# ───────────────────────── report ─────────────────────────


def print_report(report: dict[str, Any]) -> None:
    print("\n[e2e] ===== LIVE DASHBOARD INSPECTION =====", flush=True)
    if "wait_error" in report:
        print(f"[e2e] WAIT ERROR: {report['wait_error']}", flush=True)
    dom = report.get("dom", {})
    hero = dom.get("hero", {})
    print(f"[e2e] status dot connected:   {dom.get('status_connected')}", flush=True)
    print(f"[e2e] hero kind / label:      {hero.get('kind')} / {hero.get('label')}", flush=True)
    print(f"[e2e] hero reasons:           {hero.get('reasons')}", flush=True)
    print(f"[e2e] elapsed / eta:          {hero.get('elapsed')} / {hero.get('eta')}", flush=True)
    print(f"[e2e] active phase / pct:     {hero.get('phase_name')} / {hero.get('phase_pct')}", flush=True)
    phases = dom.get("phases", [])
    print(f"[e2e] phase cards ({len(phases)}):", flush=True)
    for ph in phases:
        print(f"[e2e]    {ph.get('name'):<12} {ph.get('badge'):<10} {ph.get('progress')}", flush=True)
    chart = dom.get("chart")
    if chart:
        pts = list(zip(chart["labels"], chart["sample_counts"], strict=False))
        print(f"[e2e] chart datasets:         {pts}", flush=True)
    kpi = dom.get("kpi", [])
    print(f"[e2e] KPI tiles ({len(kpi)}):", flush=True)
    for tile in kpi:
        slo = tile.get("slo_kind") or "—"
        chip = tile.get("chip_text") or ""
        print(
            f"[e2e]   {tile.get('label'):<20} [{tile.get('primary_stat','?'):<7}] "
            f"{tile.get('val','?'):<10}{tile.get('unit',''):<6}  "
            f"{slo:<5}  {chip}  (sparkline paths={tile.get('sparkline_points')})",
            flush=True,
        )
    srv = dom.get("server_metrics", [])
    if srv:
        print(f"[e2e] server metrics ({len(srv)}):", flush=True)
        for row in srv:
            print(f"[e2e]   {row.get('name'):<32} {row.get('value'):<18} {row.get('saturation') or ''}", flush=True)
    print(f"[e2e] log entries:            {dom.get('log_entries')}", flush=True)
    console_errs = [c for c in report.get("console", []) if "pageerror" in c or c.startswith("[error]")]
    if console_errs:
        print(f"[e2e] browser console errors ({len(console_errs)}):", flush=True)
        for c in console_errs[:10]:
            print(f"[e2e]   {c}", flush=True)
    else:
        print("[e2e] browser console: clean (no errors)", flush=True)
    print("[e2e] screenshots:", flush=True)
    for s in report.get("screenshots", []):
        print(f"[e2e]   {s}", flush=True)
    print("[e2e] ====================================\n", flush=True)


# ───────────────────────── main ─────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run a real aiperf benchmark against a real mock server, "
                    "then inspect the live API dashboard in a headless browser.",
    )
    ap.add_argument("--model", default="mock-llama")
    ap.add_argument("--duration-sec", type=int, default=45,
                    help="Profiling-phase duration in seconds. Needs to be long "
                         "enough for the browser to observe live data (default: 45 s).")
    ap.add_argument("--request-rate", type=float, default=10.0,
                    help="Target request rate in req/s (poisson arrival).")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=5,
                    help="Warmup request count. Keep small so warmup doesn't "
                         "eat the Playwright observation window.")
    ap.add_argument("--isl", type=int, default=200)
    ap.add_argument("--osl", type=int, default=100)
    ap.add_argument(
        "--goodput",
        default="time_to_first_token:500 inter_token_latency:30",
        help="AIPerf goodput SLOs (same syntax as `aiperf profile --goodput`).",
    )
    ap.add_argument(
        "--screenshots-dir", type=Path, default=DEFAULT_SCREENSHOT_DIR,
        help="Directory to save captured screenshots into.",
    )
    ap.add_argument(
        "--save-to-docs", action="store_true",
        help="Also copy the live screenshot to docs/media/images/api-dashboard-v2.png "
             "so the repo's canonical dashboard image reflects this e2e run.",
    )
    args = ap.parse_args()

    shots_dir = args.screenshots_dir
    artifact_dir = shots_dir / "artifacts"
    shots_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    mock_log = shots_dir / "mock-server.log"
    aiperf_log = shots_dir / "aiperf.log"

    mock_port = free_port()
    api_port = free_port()
    # Guard against an unlikely collision (free_port gives one port then
    # immediately releases it; a second call can pick the same port).
    while api_port == mock_port:
        api_port = free_port()
    print(f"[e2e] mock_port={mock_port}  api_port={api_port}", flush=True)

    mock_proc: subprocess.Popen | None = None
    aiperf_proc: subprocess.Popen | None = None
    exit_code = 1

    try:
        # 1. Launch mock server.
        mock_proc = start_mock_server(mock_port, log_path=mock_log)

        # 2. Wait until it's serving any HTTP response on /health (the
        # mock returns 200 on GET; we accept anything because "server is
        # responding" is all we need to know).
        ok = wait_for_http(
            f"http://127.0.0.1:{mock_port}/health",
            deadline=time.monotonic() + 20,
            label="mock server",
            accept_any_response=True,
        )
        if not ok:
            tail_file(mock_log, "mock server")
            return 2
        print("[e2e] mock server: up", flush=True)

        # 3. Launch aiperf profile with the API server exposed.
        aiperf_proc = start_aiperf(
            mock_port=mock_port, api_port=api_port,
            model=args.model,
            duration_sec=args.duration_sec,
            request_rate=args.request_rate,
            concurrency=args.concurrency, warmup=args.warmup,
            isl=args.isl, osl=args.osl, goodput=args.goodput,
            artifact_dir=artifact_dir,
            log_path=aiperf_log,
        )

        # 4. Poll /api/config until the API answers. Accept any HTTP
        # response so we detect bring-up ASAP (uvicorn returns 405 for
        # a blink while routes are being attached). The dashboard code
        # itself only hits it via GET, which is always valid once the
        # router is up.
        ok = wait_for_http(
            f"http://127.0.0.1:{api_port}/api/config",
            deadline=time.monotonic() + 60,
            label="api server",
            accept_any_response=True,
        )
        if not ok:
            tail_file(aiperf_log, "aiperf")
            return 3
        print("[e2e] aiperf API server: up", flush=True)

        # 5. Wait long enough for aiperf to finish warmup and emit at
        # least a few realtime_metrics / credit_phase_progress batches
        # from the profiling phase. Playwright's wait_for_function
        # expiries also absorb slack here, but giving the benchmark a
        # head start prevents the script from racing the first sample.
        time.sleep(12.0)

        # 6. Drive Playwright against the real API.
        report = inspect_with_playwright(api_port=api_port, shots_dir=shots_dir)
        print_report(report)

        # Optionally mirror to the repo's canonical location.
        if args.save_to_docs and report["screenshots"]:
            canonical = REPO_ROOT / "docs" / "media" / "images" / "api-dashboard-v2.png"
            canonical.parent.mkdir(parents=True, exist_ok=True)
            src_path = Path(report["screenshots"][-1])
            shutil.copyfile(src_path, canonical)
            print(f"[e2e] canonical image updated: {canonical}", flush=True)

        # Heuristic pass/fail: hero visible, KPI tiles populated, chart
        # had real datapoints, no console errors.
        dom = report.get("dom", {})
        kpi = dom.get("kpi") or []
        chart = dom.get("chart") or {}
        chart_has_points = any(n >= 3 for n in chart.get("sample_counts", []))
        has_values = any(t.get("val") and t["val"] != "---" for t in kpi)
        hero_kind = (dom.get("hero") or {}).get("kind")
        console_errors = [c for c in report.get("console", [])
                          if c.startswith("[error]") or c.startswith("[pageerror]")]

        print("[e2e] pass/fail gate:", flush=True)
        print(f"[e2e]   status connected:   {dom.get('status_connected')}", flush=True)
        print(f"[e2e]   hero kind present:  {hero_kind}", flush=True)
        print(f"[e2e]   kpi values shown:   {has_values}", flush=True)
        print(f"[e2e]   chart has points:   {chart_has_points}", flush=True)
        print(f"[e2e]   console clean:      {len(console_errors) == 0}", flush=True)

        if dom.get("status_connected") and hero_kind and has_values \
                and chart_has_points and not console_errors:
            print("[e2e] LIVE DASHBOARD CHECK: PASS", flush=True)
            exit_code = 0
        else:
            print("[e2e] LIVE DASHBOARD CHECK: FAIL", flush=True)
            exit_code = 4

    finally:
        print("[e2e] tearing down subprocesses", flush=True)
        if aiperf_proc is not None:
            terminate(aiperf_proc, "aiperf")
        if mock_proc is not None:
            terminate(mock_proc, "mock server")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
