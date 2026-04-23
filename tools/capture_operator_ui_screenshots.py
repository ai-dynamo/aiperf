#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture full-page screenshots of every operator SPA page via headless Chromium.

Reuses the e2e suite's fixture stack to drive a real in-process uvicorn against
the real ``aiperf.operator.results_server.create_app()``, with the committed
golden results tree, a monkeypatched k8s client, and the ``tests/_js_cache/``
CDN replay. Writes one PNG per SPA page into ``--out-dir``.

No cluster, no network, no real benchmark run required — pure static capture
driven by committed fixtures, suitable for refreshing the docs screenshots
when the UI changes.

Usage::

    uv run python tools/capture_operator_ui_screenshots.py
    uv run python tools/capture_operator_ui_screenshots.py --out-dir /tmp/preview
    uv run python tools/capture_operator_ui_screenshots.py --width 1600 --height 2000

Default output: ``docs/media/images/`` (one ``operator-ui-<page>.png`` per route).

Companion: ``tools/capture_dashboard_screenshot.py`` is the single-run v2
dashboard screenshot tool. This tool covers the multi-run operator UI.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import shutil
import sys
from pathlib import Path

import orjson
import uvicorn
from playwright.async_api import async_playwright

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from aiperf.operator.results_server import create_app  # noqa: E402
from tests.e2e.operator_ui.conftest import (  # noqa: E402
    CACHEABLE_HOSTS,
    GOLDEN_K8S,
    GOLDEN_RESULTS,
    STUB_EMPTY_MAP,
    _content_type_for,
    _find_jobs_router_holder,
    _free_port,
    _load_cdn_cached,
    _pod_raw_to_v1pod,
)


# Route → output-filename mapping. Order matters only for console readability.
ROUTES = [
    ("operator-ui-01-dashboard",            "/"),
    ("operator-ui-02-jobs",                 "/#/jobs"),
    ("operator-ui-03-job-detail-completed", "/#/jobs/aiperf-bench/aiperf-llama3-c128"),
    ("operator-ui-04-job-detail-running",   "/#/jobs/aiperf-bench/live-run"),
    ("operator-ui-05-leaderboard",          "/#/leaderboard"),
    ("operator-ui-06-compare",              "/#/compare"),
    ("operator-ui-07-history",              "/#/history"),
    ("operator-ui-08-job-detail-archived",  "/#/jobs/ml-lab/ghost-run"),
]

# The UI reads status.summary as flat keys (throughput_rps, ttft_avg_ms, …),
# while profile_export_aiperf.json uses nested metrics (request_throughput.avg).
# This script seeds CR status with the flat form so KPIs render populated.
_FLAT_METRIC_MAP = {
    "throughput_rps":              ("request_throughput",     "avg"),
    "latency_p99_ms":              ("request_latency",        "p99"),
    "latency_avg_ms":              ("request_latency",        "avg"),
    "ttft_avg_ms":                 ("time_to_first_token",    "avg"),
    "ttft_p99_ms":                 ("time_to_first_token",    "p99"),
    "itl_avg_ms":                  ("inter_token_latency",    "avg"),
    "itl_p99_ms":                  ("inter_token_latency",    "p99"),
    "output_token_throughput_tps": ("output_token_throughput", "avg"),
}


def _build_flat_summary(pxjson: dict) -> dict:
    """Convert nested profile_export metrics into the flat CR-status shape."""
    out: dict = {}
    for flat_key, (metric, stat) in _FLAT_METRIC_MAP.items():
        val = pxjson.get(metric)
        if isinstance(val, dict) and val.get(stat) is not None:
            out[flat_key] = val[stat]
    out["total_requests"] = 100
    out["error_rate"] = 0.0
    return out


def _seed_results_dir(out: Path) -> Path:
    """Copy the committed golden tree, plus one archived-only job, into `out`."""
    results_dir = out / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)
    for ns in GOLDEN_RESULTS.iterdir():
        if ns.is_dir():
            shutil.copytree(ns, results_dir / ns.name)

    # One PVC-only job (no matching CR) so the archived row + banner render.
    ghost_dir = results_dir / "ml-lab" / "ghost-run"
    ghost_dir.mkdir(parents=True, exist_ok=True)
    (ghost_dir / "profile_export_aiperf.json").write_bytes(orjson.dumps({
        "status": "Succeeded",
        "start_time": "2026-04-20T10:00:00Z",
        "end_time":   "2026-04-20T10:45:00Z",
        "request_throughput":     {"avg": 55.5, "unit": "requests/sec"},
        "request_latency":        {"p99": 421.0, "unit": "ms"},
        "time_to_first_token":    {"avg": 195.0, "unit": "ms"},
        "inter_token_latency":    {"avg": 31.0,  "unit": "ms"},
        "output_token_throughput": {"avg": 7104.0, "unit": "tokens/sec"},
        "input_config": {
            "models":   {"items": [{"name": "mistral-7b"}]},
            "endpoint": {"urls": ["http://mistral.svc:8000/v1"], "type": "chat", "streaming": True},
        },
    }))
    (ghost_dir / ".aiperf_results_ready.json").write_bytes(orjson.dumps({"ready": True}))
    return results_dir


def _enrich_cr_status(jobs_raw: list[dict], results_dir: Path) -> None:
    """Populate CR ``status`` with workers, phases, timestamps, and flat summary.

    The committed golden CRs only carry ``phase`` + minimal metadata. The UI
    expects per-phase request counts, worker readiness, timestamps, and flat
    summary keys — all of which the operator writes in production. Mirror
    that here so screenshots look like a real run.
    """
    base_time = _dt.datetime(2026, 4, 22, 10, 0, tzinfo=_dt.timezone.utc)
    for idx, j in enumerate(jobs_raw):
        name = j["metadata"]["name"]
        ns = j["metadata"]["namespace"]
        created = base_time - _dt.timedelta(hours=idx * 4)
        j["metadata"]["creationTimestamp"] = created.isoformat().replace("+00:00", "Z")

        status = j.setdefault("status", {})
        phase = status.get("phase", "Succeeded")
        status["jobId"] = name
        status["jobSetName"] = f"{name}-jobset"
        status["startTime"] = created.isoformat().replace("+00:00", "Z")
        if phase != "Running":
            status["completionTime"] = (
                created + _dt.timedelta(minutes=45)
            ).isoformat().replace("+00:00", "Z")

        # Workers snapshot
        if phase == "Failed":
            status["workers"] = {"ready": 2, "total": 4}
        elif phase == "Running":
            status["workers"] = {"ready": 8, "total": 8}
        else:
            status["workers"] = {"ready": 4, "total": 4}

        # Phase progress — UI reads both camelCase (PhaseBar) and percent (jobs table)
        def _phase_block(completed: int, total: int) -> dict:
            pct = 100 if total == 0 else round((completed / total) * 100)
            return {
                "requestsCompleted": completed,
                "requestsTotal":     total,
                "requestsProgressPercent": pct,
            }

        if phase == "Running":
            status["currentPhase"] = "benchmark"
            status["phases"] = {
                "warmup":    _phase_block(500, 500),
                "benchmark": _phase_block(3348, 5400),
            }
        elif phase == "Failed":
            status["currentPhase"] = "benchmark"
            status["phases"] = {
                "warmup":    _phase_block(500, 500),
                "benchmark": _phase_block(180, 1000),
            }
        else:
            status["currentPhase"] = "completed"
            status["phases"] = {
                "warmup":    _phase_block(500, 500),
                "benchmark": _phase_block(100, 100),
            }

        # Flat summary from profile_export (or a plausible live snapshot)
        profile = results_dir / ns / name / "profile_export_aiperf.json"
        if profile.exists():
            flat = _build_flat_summary(orjson.loads(profile.read_bytes()))
            if phase == "Running":
                status["liveSummary"] = flat
            else:
                status["summary"] = flat
        elif phase == "Running":
            status["liveSummary"] = {
                "throughput_rps":  31.5,
                "latency_p99_ms":  380.0,
                "latency_avg_ms":  290.0,
                "ttft_avg_ms":     170.0,
                "ttft_p99_ms":     240.0,
                "itl_avg_ms":      28.0,
                "itl_p99_ms":      40.0,
                "output_token_throughput_tps": 4032.0,
                "total_requests":  5400,
                "error_rate":      0.0,
            }

        # Conditions — visible badges on the detail page
        status.setdefault("conditions", [])
        if phase == "Succeeded" and not status["conditions"]:
            status["conditions"] = [
                {"type": "Ready",     "status": "True", "reason": "BenchmarkComplete"},
                {"type": "Succeeded", "status": "True"},
            ]
        elif phase == "Failed" and not status["conditions"]:
            status["conditions"] = [
                {"type": "Failed", "status": "True", "reason": "PodError",
                 "message": "controller pod exited non-zero"},
            ]
        elif phase == "Running" and not status["conditions"]:
            status["conditions"] = [{"type": "Ready", "status": "True"}]


def _install_fake_k8s(jobs_raw: list[dict], pods_raw: list[dict], cluster: dict) -> None:
    """Monkeypatch the six k8s helpers across every module that imports them."""
    from aiperf.kubernetes.models import AIPerfJobCR

    async def list_aiperf_jobs(api, *, all_namespaces=True, namespace=None, **_):
        out = []
        for r in jobs_raw:
            if all_namespaces or r["metadata"]["namespace"] == namespace:
                out.append(AIPerfJobCR.model_validate(r).to_info())
        return out

    async def find_aiperf_job(api, name, namespace):
        for r in jobs_raw:
            m = r["metadata"]
            if m["name"] == name and m["namespace"] == namespace:
                return AIPerfJobCR.model_validate(r).to_info()
        return None

    async def get_raw_aiperfjob_status(api, name, namespace):
        for r in jobs_raw:
            m = r["metadata"]
            if m["name"] == name and m["namespace"] == namespace:
                return r.get("status", {})
        return {}

    async def get_raw_aiperfjob(api, namespace, name):
        for r in jobs_raw:
            m = r["metadata"]
            if m["name"] == name and m["namespace"] == namespace:
                return r
        return None

    async def get_pods(api, namespace, label_selector):
        job_id = label_selector.split("=", 1)[1]
        return [
            _pod_raw_to_v1pod(p) for p in pods_raw
            if p["metadata"].get("labels", {}).get("aiperf.nvidia.com/job-id") == job_id
        ]

    async def cluster_version(api):
        return cluster

    async def cancel_aiperf_job(api, name, namespace):
        return None

    import aiperf.kubernetes.client as kc
    import aiperf.kubernetes.client_jobs as kj
    import aiperf.kubernetes.client_pods as kp
    import aiperf.operator.job_union as ju
    import aiperf.operator.routers.jobs as jr
    import aiperf.operator.routers.results_analytics as ra

    patches = [
        ("list_aiperf_jobs",         list_aiperf_jobs),
        ("find_aiperf_job",          find_aiperf_job),
        ("get_raw_aiperfjob_status", get_raw_aiperfjob_status),
        ("get_raw_aiperfjob",        get_raw_aiperfjob),
        ("get_pods",                 get_pods),
        ("cluster_version",          cluster_version),
        ("cancel_aiperf_job",        cancel_aiperf_job),
    ]
    for mod in (kc, kj, kp, ju, jr, ra):
        for nm, fn in patches:
            try:
                setattr(mod, nm, fn)
            except Exception:
                pass


async def _capture_all(out_dir: Path, width: int, height: int, settle_ms: int) -> None:
    results_dir = _seed_results_dir(out_dir.parent)
    app = create_app(results_dir=results_dir)

    jobs_raw = orjson.loads((GOLDEN_K8S / "jobs.json").read_bytes())["items"]
    pods_raw = orjson.loads((GOLDEN_K8S / "pods.json").read_bytes())["items"]
    cluster = orjson.loads((GOLDEN_K8S / "version.json").read_bytes())
    _enrich_cr_status(jobs_raw, results_dir)
    _install_fake_k8s(jobs_raw, pods_raw, cluster)

    holder = _find_jobs_router_holder(app)
    if holder is not None:
        holder[0] = object()

    port = _free_port()
    cfg = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning",
        access_log=False, lifespan="on",
    )
    server = uvicorn.Server(cfg)
    serve_task = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.05)
    if not server.started:
        raise RuntimeError("uvicorn failed to start within 10s")

    base = f"http://127.0.0.1:{port}"
    print(f"serving at {base}")

    async def _route(route):
        url = route.request.url
        if url.startswith(base):
            await route.continue_()
            return
        for needle, content_type in STUB_EMPTY_MAP.items():
            if needle in url:
                await route.fulfill(status=200, content_type=content_type, body=b"")
                return
        for prefix in CACHEABLE_HOSTS:
            if url.startswith(prefix):
                body = await asyncio.to_thread(_load_cdn_cached, url)
                await route.fulfill(status=200, content_type=_content_type_for(url), body=body)
                return
        print(f"  WARN unmapped external: {url}")
        await route.abort()

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            ctx = await browser.new_context(viewport={"width": width, "height": height})
            page = await ctx.new_page()
            await page.route("**/*", _route)

            out_dir.mkdir(parents=True, exist_ok=True)
            for name, hash_route in ROUTES:
                await page.goto(base + hash_route)
                # Hash changes don't reload the SPA, so state leaks across
                # screenshots. Force a full reload to get clean state.
                await page.reload()
                await page.wait_for_load_state("networkidle")
                await page.wait_for_timeout(settle_ms)
                dst = out_dir / f"{name}.png"
                await page.screenshot(path=str(dst), full_page=True)
                print(f"  wrote {dst}")

            await browser.close()
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(serve_task, timeout=5.0)
        except asyncio.TimeoutError:
            serve_task.cancel()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--out-dir", type=Path,
        default=REPO / "docs" / "media" / "images",
        help="Directory to write PNGs into (default: docs/media/images/).",
    )
    parser.add_argument("--width", type=int, default=1440,
                        help="Viewport width in px (default: 1440).")
    parser.add_argument("--height", type=int, default=1600,
                        help="Viewport height in px (default: 1600).")
    parser.add_argument("--settle-ms", type=int, default=900,
                        help="Wait after networkidle before screenshot (default: 900).")
    args = parser.parse_args()

    asyncio.run(_capture_all(args.out_dir, args.width, args.height, args.settle_ms))


if __name__ == "__main__":
    main()
