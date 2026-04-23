# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the operator web UI e2e suite.

Runs a real uvicorn server bound to 127.0.0.1:<random> once per session,
hosting the real ``create_app()`` FastAPI instance with a session-scoped
``results_dir``. Per-test fixtures mutate the contents of that dir and
monkeypatch the k8s helpers — no respawn.
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import socket
import urllib.request
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import orjson
import pytest
import pytest_asyncio
import uvicorn
from fastapi import FastAPI
from playwright.async_api import (
    Browser,
    BrowserContext,
    ConsoleMessage,
    Page,
    Playwright,
    Route,
    async_playwright,
)

from aiperf.operator.results_server import create_app

GOLDEN_RESULTS = (
    Path(__file__).parent.parent.parent / "fixtures" / "operator_ui" / "results"
)
GOLDEN_K8S = Path(__file__).parent.parent.parent / "fixtures" / "operator_ui" / "k8s"


def _free_port() -> int:
    """Bind to port 0 and return the kernel-assigned port.

    There's a TOCTOU race between binding here and re-binding in uvicorn,
    but in practice it's safe on localhost and avoids uvicorn's lack of
    a "port 0 then tell me what you got" API in older versions.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class LiveApp:
    base_url: str
    app: FastAPI
    results_dir: Path


@asynccontextmanager
async def _running_server(app: FastAPI, port: int) -> AsyncIterator[None]:
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
        lifespan="on",
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    # Wait for startup
    for _ in range(200):  # 10s max at 50ms
        if server.started:
            break
        await asyncio.sleep(0.05)
    if not server.started:
        server.should_exit = True
        await task
        raise RuntimeError("uvicorn failed to start within 10s")
    try:
        yield
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except asyncio.TimeoutError:
            task.cancel()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def live_operator_app(tmp_path_factory) -> AsyncIterator[LiveApp]:
    """Real uvicorn + real ``create_app()`` bound to a random port.

    The ``results_dir`` is session-scoped; per-test fixtures rewrite its
    contents. The jobs router's ``ApiClient`` stays ``None`` (tests that
    need it monkeypatch the six ``aiperf.kubernetes.client`` helpers).
    """
    results_dir = tmp_path_factory.mktemp("e2e_results")
    app = create_app(results_dir=results_dir)
    port = _free_port()
    async with _running_server(app, port):
        yield LiveApp(
            base_url=f"http://127.0.0.1:{port}",
            app=app,
            results_dir=results_dir,
        )


@pytest.fixture
def seeded_results_dir(live_operator_app: LiveApp) -> Path:
    """Clear the session results dir and copy the golden tree into it.

    The session-scoped ``results_dir`` persists across tests, so we wipe it
    first to keep each test hermetic. Returns the (now-seeded) ``results_dir``.
    """
    target = live_operator_app.results_dir
    for child in target.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    for ns_dir in GOLDEN_RESULTS.iterdir():
        if ns_dir.is_dir():
            shutil.copytree(ns_dir, target / ns_dir.name)
    return target


# =============================================================================
# Fake Kubernetes client
# =============================================================================


@dataclass
class FakeK8sClient:
    """Canned responses + call recorder for the jobs router.

    Priming helpers let individual tests override defaults; assertion helpers
    like ``cancelled`` let tests verify side effects. The fixture below wires
    the six ``aiperf.kubernetes.client`` helpers called by the jobs router to
    read from this instance.
    """

    jobs_raw: list[dict[str, Any]] = field(default_factory=list)
    """Raw AIPerfJob CR dicts (as returned by the apiserver)."""

    pods_by_job: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    """Raw pod dicts keyed by ``metadata.labels[aiperf.nvidia.com/job-id]``."""

    cluster_version_info: dict[str, Any] = field(default_factory=dict)
    """Return value for :func:`aiperf.kubernetes.client.cluster_version`."""

    cancelled: list[tuple[str, str]] = field(default_factory=list)
    """Recorded ``(namespace, name)`` for each cancel call."""

    def set_jobs(self, jobs_raw: list[dict[str, Any]]) -> None:
        """Replace the canned AIPerfJob CR list."""
        self.jobs_raw = jobs_raw

    def set_pods(self, job_id: str, pods_raw: list[dict[str, Any]]) -> None:
        """Replace the pod list for a given job id."""
        self.pods_by_job[job_id] = pods_raw


def _load_json(path: Path) -> dict[str, Any]:
    return orjson.loads(path.read_bytes())


def _pod_raw_to_v1pod(pod_raw: dict[str, Any]) -> SimpleNamespace:
    """Build a V1Pod-shaped SimpleNamespace the router's ``_pod_summary`` reads.

    ``_pod_summary`` accesses: ``pod.metadata.name``, ``pod.status.phase``,
    ``pod.status.container_statuses``, and per-container ``c.ready`` and
    ``c.restart_count``.
    """
    meta = pod_raw["metadata"]
    status = pod_raw["status"]
    return SimpleNamespace(
        metadata=SimpleNamespace(
            name=meta["name"],
            namespace=meta.get("namespace"),
            labels=meta.get("labels", {}),
        ),
        status=SimpleNamespace(
            phase=status["phase"],
            container_statuses=[
                SimpleNamespace(
                    ready=c.get("ready", False),
                    restart_count=c.get("restartCount", 0),
                )
                for c in status.get("containerStatuses", [])
            ],
        ),
    )


def _find_jobs_router_holder(app: FastAPI) -> list[Any] | None:
    """Walk route closures to find the jobs router's ``api_holder`` list.

    The holder is a closure captured inside ``create_jobs_router``'s
    ``_require_api`` helper. Each route endpoint captures ``_require_api``,
    and ``_require_api`` captures the holder list. Reaching it lets the
    fixture inject a non-None sentinel so the router's 503 guard passes.
    """
    for route in app.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None or endpoint.__closure__ is None:
            continue
        for cell in endpoint.__closure__:
            try:
                candidate = cell.cell_contents
            except ValueError:
                continue
            if not callable(candidate):
                continue
            if getattr(candidate, "__closure__", None) is None:
                continue
            if getattr(candidate, "__name__", "") != "_require_api":
                continue
            for inner in candidate.__closure__:
                try:
                    val = inner.cell_contents
                except ValueError:
                    continue
                if isinstance(val, list) and len(val) >= 1:
                    return val
    return None


@pytest.fixture
def fake_k8s_client(
    live_operator_app: LiveApp, monkeypatch: pytest.MonkeyPatch
) -> FakeK8sClient:
    """Patch the six ``aiperf.kubernetes.client`` helpers the jobs router uses.

    Seeds default responses from ``tests/fixtures/operator_ui/k8s/``; tests
    override via the returned :class:`FakeK8sClient`. Also injects a non-None
    sentinel into the jobs router's ``api_holder`` so the 503 "Kubernetes API
    unavailable" guard passes — the monkeypatched helpers ignore the ``api``
    argument anyway.
    """
    from aiperf.kubernetes.models import AIPerfJobCR

    fake = FakeK8sClient()
    fake.jobs_raw = _load_json(GOLDEN_K8S / "jobs.json")["items"]
    fake.pods_by_job = {
        "live-run": _load_json(GOLDEN_K8S / "pods.json")["items"],
    }
    fake.cluster_version_info = _load_json(GOLDEN_K8S / "version.json")

    async def _list(api, namespace=None, all_namespaces=False, status_filter=None):
        if all_namespaces:
            raws = fake.jobs_raw
        else:
            ns = namespace or "default"
            raws = [j for j in fake.jobs_raw if j["metadata"]["namespace"] == ns]
        infos = [AIPerfJobCR.model_validate(raw).to_info() for raw in raws]
        if status_filter:
            infos = [i for i in infos if i.phase == status_filter]
        return infos

    async def _find(api, name, namespace=None):
        for raw in fake.jobs_raw:
            meta = raw["metadata"]
            if meta["name"] != name:
                continue
            if namespace is not None and meta["namespace"] != namespace:
                continue
            return AIPerfJobCR.model_validate(raw).to_info()
        return None

    async def _raw_status(api, name, namespace):
        for raw in fake.jobs_raw:
            meta = raw["metadata"]
            if meta["name"] == name and meta["namespace"] == namespace:
                return raw.get("status", {}) or {}
        return {}

    async def _get_pods(api, namespace, label_selector):
        # Canonical AIPerf selector is "aiperf.nvidia.com/job-id=<name>".
        _, _, job_id = label_selector.partition("=")
        return [_pod_raw_to_v1pod(p) for p in fake.pods_by_job.get(job_id, [])]

    async def _version(api):
        return dict(fake.cluster_version_info)

    async def _cancel(api, name, namespace):
        fake.cancelled.append((namespace, name))

    # Patch both the source module and the router's local re-imports, because
    # the router does ``from aiperf.kubernetes.client import ...`` which binds
    # the names into its own module namespace.
    import aiperf.kubernetes.client as kc_mod
    import aiperf.kubernetes.client_jobs as kc_jobs_mod
    import aiperf.kubernetes.client_pods as kc_pods_mod
    import aiperf.operator.routers.jobs as jobs_router

    patch_targets = (kc_mod, kc_jobs_mod, kc_pods_mod, jobs_router)
    for target in patch_targets:
        monkeypatch.setattr(target, "list_aiperf_jobs", _list, raising=False)
        monkeypatch.setattr(target, "find_aiperf_job", _find, raising=False)
        monkeypatch.setattr(
            target, "get_raw_aiperfjob_status", _raw_status, raising=False
        )
        monkeypatch.setattr(target, "get_pods", _get_pods, raising=False)
        monkeypatch.setattr(target, "cluster_version", _version, raising=False)
        monkeypatch.setattr(target, "cancel_aiperf_job", _cancel, raising=False)

    # Inject a non-None sentinel into the router's ``api_holder`` so
    # ``_require_api`` passes. The holder is a closure local in
    # ``create_jobs_router``; reach it via ``endpoint.__closure__``.
    holder = _find_jobs_router_holder(live_operator_app.app)
    if holder is None:
        raise RuntimeError(
            "fake_k8s_client could not locate the jobs router api_holder via "
            "route-closure traversal. Did create_jobs_router's wiring change?"
        )
    original = holder[0]
    holder[0] = object()  # sentinel; monkeypatched helpers ignore `api`

    yield fake

    holder[0] = original


# =============================================================================
# Browser page fixture with CDN interception + console-error gate
# =============================================================================

# On-disk cache for CDN responses pulled during e2e runs, committed to git so
# subsequent runs are fully offline. Keyed by ``sha256(url)[:40]`` -> bytes,
# with a sibling ``<digest>.meta`` recording the original URL for auditability.
#
# Why cache live CDN and not use the committed ``ui/vendor/`` stubs? Those
# stubs are version-pinned snippets (e.g. ``preact@10.29.0``) that re-export
# from deeper bundle URLs. Mixing them with the un-pinned specifiers
# ``htm@3/preact`` and ``@preact/signals@1`` pulls in a newer transitive preact
# (e.g. 10.29.1) and the browser treats them as distinct module graphs —
# ``h`` is exported by one copy but missing from the other, breaking htm.
# Letting the full CDN graph cache keeps version coherence.
JS_CACHE = Path(__file__).parent.parent.parent / "_js_cache"

# CDN hosts whose module/bundle responses are cached under ``_js_cache/``.
# First run populates the cache from the live CDN; subsequent runs are offline.
CACHEABLE_HOSTS: tuple[str, ...] = (
    "https://esm.sh/",
    "https://cdn.jsdelivr.net/",
)

# Host-substrings stubbed with empty bodies — fonts are noise for e2e tests.
STUB_EMPTY_MAP: dict[str, str] = {
    "fonts.googleapis.com": "text/css",
    "fonts.gstatic.com": "font/woff2",
}


def _cache_path(url: str) -> Path:
    """Deterministic on-disk cache path for a CDN URL."""
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:40]
    return JS_CACHE / digest


def _load_cdn_cached(url: str) -> bytes:
    """Return the bytes for ``url``, populating ``_js_cache/`` on miss.

    Writes a sibling ``<digest>.meta`` with the source URL for auditability
    (so a reviewer inspecting ``tests/_js_cache/`` can see what each blob is).
    """
    body_path = _cache_path(url)
    if body_path.exists():
        return body_path.read_bytes()
    JS_CACHE.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=15) as resp:  # noqa: S310
        body = resp.read()
    body_path.write_bytes(body)
    body_path.with_suffix(".meta").write_text(url, encoding="utf-8")
    return body


# Override pytest-playwright-asyncio's default ``playwright``/``browser`` fixtures
# so they share the session-scoped event loop used by ``live_operator_app``.
# Without matching ``loop_scope="session"``, pytest-asyncio raises ScopeMismatch
# when a test requests both the session-scoped app and the (otherwise
# function-loop-scoped) ``page``.
@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def playwright() -> AsyncIterator[Playwright]:
    pw = await async_playwright().start()
    try:
        yield pw
    finally:
        await pw.stop()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def browser(playwright: Playwright) -> AsyncIterator[Browser]:
    browser = await playwright.chromium.launch()
    try:
        yield browser
    finally:
        await browser.close()


@pytest_asyncio.fixture(loop_scope="session")
async def context(browser: Browser) -> AsyncIterator[BrowserContext]:
    ctx = await browser.new_context()
    try:
        yield ctx
    finally:
        await ctx.close()


@pytest_asyncio.fixture(loop_scope="session")
async def page(
    live_operator_app: LiveApp, context: BrowserContext
) -> AsyncIterator[Page]:
    """Browser ``page`` with CDN interception and an error gate.

    - Requests to ``live_operator_app.base_url`` pass through unchanged.
    - ``esm.sh`` / ``cdn.jsdelivr.net`` requests are cached under
      ``tests/_js_cache/`` on first hit and replayed offline on subsequent
      runs. Chart.js picks content-type from ``application/javascript``; ES
      modules need ``text/javascript`` — both served via ``text/javascript``
      which is accepted for classic scripts and required for modules in
      modern Chromium.
    - Font CDN requests are stubbed to empty bodies (tests don't need webfonts).
    - Any ``pageerror`` or ``console.error`` fails the test at teardown;
      unmapped external requests fail the test at teardown.
    """
    errors: list[str] = []
    unmapped: list[str] = []

    def _on_pageerror(exc: Exception) -> None:
        errors.append(f"pageerror: {exc}")

    def _on_console(msg: ConsoleMessage) -> None:
        if msg.type == "error":
            errors.append(f"console.error: {msg.text}")

    page = await context.new_page()
    page.on("pageerror", _on_pageerror)
    page.on("console", _on_console)

    async def _handle(route: Route) -> None:
        url = route.request.url
        if url.startswith(live_operator_app.base_url):
            await route.continue_()
            return
        for needle, content_type in STUB_EMPTY_MAP.items():
            if needle in url:
                await route.fulfill(status=200, content_type=content_type, body=b"")
                return
        for prefix in CACHEABLE_HOSTS:
            if url.startswith(prefix):
                body = await asyncio.to_thread(_load_cdn_cached, url)
                await route.fulfill(
                    status=200, content_type="text/javascript", body=body
                )
                return
        unmapped.append(url)
        await route.abort()

    await page.route("**/*", _handle)

    try:
        yield page
    finally:
        await page.close()

    if unmapped:
        pytest.fail(
            "Unmapped external requests (add them to CACHEABLE_HOSTS or fix "
            "the UI):\n" + "\n".join(f"  - {u}" for u in unmapped)
        )
    if errors:
        pytest.fail("Browser errors detected:\n" + "\n".join(errors))
