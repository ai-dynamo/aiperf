# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared Playwright + uvicorn harness for operator-UI end-to-end tests.

Goals:
  * One Chromium browser per test, so sync Playwright's private event loop
    closes before the worker picks up unrelated pytest-asyncio tests.
  * One uvicorn process serving the operator results-server against a
    per-session tmpdir, with kubernetes_asyncio bypassed.
  * Per-test isolation through unique namespaces — tests seed their data at
    ``<results_dir>/<unique_ns>/...`` and never collide.
  * A registry that lets a test install a fake live AIPerfJob CR for a given
    ``(ns, name)`` without monkeypatching the operator module from inside the
    test body. Without an entry, ``find_aiperf_job`` returns ``None``
    (= archived-only / no live cluster state).

Each test receives a single ``harness`` fixture exposing the moving parts:

    def test_X(harness):
        harness.seed_run(ns=harness.ns, name="j", epoch="1714069323",
                         summary=good_summary())
        page = harness.goto_job_detail(harness.ns, "j", epoch="1714069323")
        assert page.locator('[data-testid=...]').is_visible()

Why a fixture object and not a flock of fixtures?

  * Adversarial tests evolve fast and need to pull in seeds, the page, the
    base URL, the registry — collecting these on one object keeps the test
    signature short and the imports trivial for agents.

Skips when Playwright or Chromium is unavailable (same pattern as
``tests/unit/api/test_dashboard_js.py``).
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch
from urllib.parse import quote

import pytest

# The sandbox sets HTTP_PROXY / HTTPS_PROXY pointing at a local proxy that
# returns 405 for any localhost URL it sees, which corrupts every urllib /
# aiohttp call we'd make from inside the harness against our own uvicorn.
# Build a proxyless opener once and route all internal HTTP through it.
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _urlopen_local(url: str, *, timeout: float = 5.0):
    """``urllib.request.urlopen`` that ignores HTTP_PROXY for localhost reads."""
    return _NO_PROXY_OPENER.open(url, timeout=timeout)


# Re-exported so test files only need to import from this conftest module.
from tests.unit.operator.ui_e2e._seeds import (  # noqa: F401, E402  (re-export; intentional late import)
    FakeLiveCR,
    good_summary,
    seed_results_ready,
    seed_run,
    seed_sweep_aggregate,
)

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Page


# ---------------------------------------------------------------------------
# Skip detection: same pattern as tests/unit/api/test_dashboard_js.py.
# ---------------------------------------------------------------------------


def _playwright_ready() -> tuple[bool, str]:
    """Return (ok, reason). ``reason`` is the pytest skip reason on miss."""
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
    except Exception as exc:  # noqa: BLE001
        return (
            False,
            f"Chromium not launchable: {exc!s}. Run `uv run playwright install chromium`.",
        )
    return True, ""


_PLAYWRIGHT_AVAILABLE, _PLAYWRIGHT_REASON = _playwright_ready()


# ---------------------------------------------------------------------------
# Per-process live-CR registries. Tests mutate via ``Harness.register_cr`` /
# ``Harness.register_sweep_cr``. Patched lookups consult these dicts.
#   * jobs    keyed by (namespace, name) -> FakeLiveCR
#   * sweeps  keyed by (namespace, name) -> FakeLiveSweepCR
# ---------------------------------------------------------------------------


_LIVE_CR_REGISTRY: dict[tuple[str, str], FakeLiveCR] = {}
_LIVE_SWEEP_REGISTRY: dict[tuple[str, str], Any] = {}


async def _fake_find_aiperf_job(api: Any, name: str, namespace: str):
    """Replacement for ``aiperf.operator.job_union.find_aiperf_job``."""
    fake = _LIVE_CR_REGISTRY.get((namespace, name))
    return fake.to_info() if fake is not None else None


async def _fake_list_aiperf_jobs(api: Any, all_namespaces: bool = True, **_kw):
    """Replacement for ``aiperf.operator.job_union.list_aiperf_jobs``.

    Reads from the in-process registry so live-only jobs surface in
    ``GET /api/v1/jobs``.
    """
    return [fake.to_info() for fake in _LIVE_CR_REGISTRY.values()]


async def _fake_find_aiperfsweep(api: Any, namespace: str, name: str):
    """Replacement for ``aiperf.operator.sweep_union.find_aiperfsweep``.

    Returns whatever the test stashed (typically a dict or pydantic model
    matching the operator's ``AIPerfSweepCR`` shape). Defaults to ``None``.
    """
    return _LIVE_SWEEP_REGISTRY.get((namespace, name))


async def _fake_list_aiperfsweeps(api: Any, all_namespaces: bool = True, **_kw):
    """Replacement for ``aiperf.operator.sweep_union.list_aiperfsweeps``."""
    return list(_LIVE_SWEEP_REGISTRY.values())


async def _fake_get_raw_aiperfjob(
    api: Any, namespace: str, name: str, *, suppress_api_errors: bool = True
):
    """Replacement for ``aiperf.kubernetes.client.get_raw_aiperfjob``.

    Returns the raw CR shape used by ``/api/v1/config/{ns}/{name}`` and
    ``/api/v1/jobs/{ns}/{name}/events``. Defaults to ``None`` so the routes
    serve a clean "config unavailable" / "no events" response instead of
    surfacing ``MagicMock can't be used in 'await' expression``.
    """
    fake = _LIVE_CR_REGISTRY.get((namespace, name))
    return getattr(fake, "raw_cr", None) if fake is not None else None


# ---------------------------------------------------------------------------
# Pod-roster patch: ``get_pods`` queries the live cluster. With no cluster
# wired up the call would raise; return [] so live CR codepaths still respond.
# ---------------------------------------------------------------------------


async def _fake_get_pods(api: Any, namespace: str, selector: str):
    return []


# ---------------------------------------------------------------------------
# Server fixture: session-scoped, threaded uvicorn against a session tmpdir.
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass(slots=True)
class _ServerHandle:
    base_url: str
    results_dir: Path
    server: Any
    thread: threading.Thread
    stack: contextlib.ExitStack


@pytest.fixture(scope="session")
def _operator_server(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[_ServerHandle]:
    """Run the operator results-server in-process for the whole session."""
    if not _PLAYWRIGHT_AVAILABLE:
        pytest.skip(_PLAYWRIGHT_REASON)

    import uvicorn

    # Apply patches BEFORE create_app() inspects the operator module graph.
    stack = contextlib.ExitStack()

    from aiperf.operator import job_union, sweep_union
    from aiperf.operator.routers import (
        jobs as jobs_router_module,
    )
    from aiperf.operator.routers import (
        results_analytics as results_analytics_module,
    )

    stack.enter_context(
        patch.object(
            job_union, "find_aiperf_job", AsyncMock(side_effect=_fake_find_aiperf_job)
        )
    )
    stack.enter_context(
        patch.object(
            job_union, "list_aiperf_jobs", AsyncMock(side_effect=_fake_list_aiperf_jobs)
        )
    )
    stack.enter_context(
        patch.object(
            sweep_union,
            "find_aiperfsweep",
            AsyncMock(side_effect=_fake_find_aiperfsweep),
        )
    )
    stack.enter_context(
        patch.object(
            sweep_union,
            "list_aiperfsweeps",
            AsyncMock(side_effect=_fake_list_aiperfsweeps),
        )
    )
    # Live job-detail also calls get_raw_aiperfjob_status + get_pods on the CR
    # half. Stub them so live-CR scenarios don't try to hit a real apiserver.
    stack.enter_context(
        patch.object(
            jobs_router_module,
            "get_raw_aiperfjob_status",
            AsyncMock(return_value={}),
        )
    )
    stack.enter_context(
        patch.object(
            jobs_router_module,
            "get_raw_aiperfjob",
            AsyncMock(side_effect=_fake_get_raw_aiperfjob),
        )
    )
    stack.enter_context(
        patch.object(
            results_analytics_module,
            "get_raw_aiperfjob",
            AsyncMock(side_effect=_fake_get_raw_aiperfjob),
        )
    )
    stack.enter_context(
        patch.object(jobs_router_module, "get_pods", AsyncMock(return_value=[]))
    )
    # ``GET /api/v1/cluster/gpu-capacity`` calls list_nodes on the apiserver;
    # without a stub the routes log "Failed to query nodes: object MagicMock
    # can't be used in 'await' expression" and return zero capacity. Harmless
    # but pollutes ``console_errors`` for tests that assert it's empty.
    stack.enter_context(
        patch.object(jobs_router_module, "list_nodes", AsyncMock(return_value=[]))
    )

    # The operator lifespan tries load_incluster_config() then load_kube_config().
    # We need ``api_holder[0]`` to come out non-None so the live-CR routes don't
    # 503 with "Kubernetes API client not yet initialized". Strategy:
    #   * incluster_config raises (no in-pod env)
    #   * kube_config succeeds with a no-op (we don't want real cluster I/O)
    #   * ApiClient() returns a MagicMock so all attribute access is harmless
    #     — the find_aiperf_job / get_pods / get_raw_aiperfjob_status patches
    #     above ensure the mock is never invoked through real codepaths.
    from kubernetes_asyncio import config as _k8s_config
    from kubernetes_asyncio.client import ApiClient as _ApiClient  # noqa: F401

    def _raise_cfg(*_a, **_kw):
        raise _k8s_config.ConfigException("disabled in ui_e2e harness")

    async def _aok(*_a, **_kw):  # load_kube_config
        return None

    stack.enter_context(patch.object(_k8s_config, "load_incluster_config", _raise_cfg))
    stack.enter_context(patch.object(_k8s_config, "load_kube_config", _aok))

    from unittest.mock import MagicMock

    fake_api_client = MagicMock(name="FakeApiClient")

    # close() is awaited during lifespan teardown.
    async def _aclose():
        return None

    fake_api_client.close = _aclose
    import kubernetes_asyncio.client as _k8s_client

    stack.enter_context(
        patch.object(_k8s_client, "ApiClient", MagicMock(return_value=fake_api_client))
    )

    results_dir = tmp_path_factory.mktemp("ui_e2e_results")
    from aiperf.operator.results_server import create_app

    app = create_app(results_dir=results_dir)
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
    # First wait for uvicorn's startup hook to flip `started` — that signals
    # the lifespan completed and the listening socket is up. Then sanity-check
    # with /healthz so a misconfigured app surfaces an obvious error.
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 20.0
    while not getattr(server, "started", False):
        if time.monotonic() > deadline:
            server.should_exit = True
            stack.close()
            raise RuntimeError(
                f"operator uvicorn did not flip ``started`` at {base} within 20 s"
            ) from None
        time.sleep(0.05)
    health_deadline = time.monotonic() + 5.0
    while True:
        try:
            _urlopen_local(f"{base}/healthz", timeout=0.5).read()
            break
        except (urllib.error.URLError, OSError) as exc:
            if time.monotonic() > health_deadline:
                server.should_exit = True
                stack.close()
                raise RuntimeError(
                    f"operator /healthz did not respond at {base} within 5 s "
                    f"after server.started=True"
                ) from exc
            time.sleep(0.05)
    try:
        yield _ServerHandle(
            base_url=base,
            results_dir=results_dir,
            server=server,
            thread=thread,
            stack=stack,
        )
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)
        stack.close()


# ---------------------------------------------------------------------------
# Browser fixture: function-scoped so sync Playwright does not leave a
# running event loop active for unrelated async tests on the same xdist worker.
# ---------------------------------------------------------------------------


@pytest.fixture
def _browser() -> Iterator[Browser]:
    if not _PLAYWRIGHT_AVAILABLE:
        pytest.skip(_PLAYWRIGHT_REASON)
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            try:
                browser.close()
            except RuntimeError as exc:
                if "no running event loop" not in str(exc):
                    raise


# ---------------------------------------------------------------------------
# Per-test harness — bundles seeds, page, base url, registry helpers.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Harness:
    """Per-test handle giving access to seeds, the live SPA page, and helpers.

    Attributes:
        page: Fresh Playwright page (new browser context per test).
        results_dir: Session-shared base dir; tests scope writes under
            ``<results_dir>/<ns>``.
        ns: Unique namespace generated for this test. Use as the ``ns=`` arg
            of seed helpers and routes to avoid collisions across tests.
        base_url: ``http://127.0.0.1:<port>`` of the session uvicorn server.
        console_errors: Browser console errors observed since page setup.
        bad_responses: HTTP 4xx/5xx responses observed since page setup. Each
            entry is ``f"{status} {method} {url}"``.
    """

    page: Page
    results_dir: Path
    ns: str
    base_url: str
    console_errors: list[str]
    bad_responses: list[str]

    # ---- seeds -------------------------------------------------------------

    def seed_run(self, **kwargs) -> Path:
        """Thin wrapper over :func:`seed_run` with a default ``ns``."""
        kwargs.setdefault("ns", self.ns)
        return seed_run(self.results_dir, **kwargs)

    def seed_sweep_aggregate(self, **kwargs) -> Path:
        kwargs.setdefault("ns", self.ns)
        return seed_sweep_aggregate(self.results_dir, **kwargs)

    def seed_results_ready(self, run: Path) -> None:
        seed_results_ready(run)

    # ---- live CR registry --------------------------------------------------

    def register_cr(self, cr: FakeLiveCR) -> None:
        """Install a fake live AIPerfJob CR that ``find_aiperf_job`` will return."""
        _LIVE_CR_REGISTRY[(cr.namespace, cr.name)] = cr

    def register_sweep_cr(self, ns: str, name: str, raw: Any) -> None:
        """Install a fake live AIPerfSweep CR for ``find_aiperfsweep``.

        ``raw`` should match the shape ``sweep_union._record_from_cr`` expects
        (a parsed CR dict, or an ``AIPerfSweepCR`` model). Pass whatever the
        operator's sweep_union codepath consumes — the harness does not coerce.
        """
        _LIVE_SWEEP_REGISTRY[(ns, name)] = raw

    def clear_all_seeded_data(self) -> None:
        """Remove every namespace directory under ``results_dir``.

        Use this before a Dashboard / list-page test that needs a clean slate —
        ``results_dir`` is session-scoped, so seeded data from earlier tests
        leaks into list-endpoint responses by default. The runs-index SQLite
        and any non-namespace top-level files are left alone.
        """
        import shutil

        if not self.results_dir.exists():
            return
        for child in self.results_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
        # Also reset the in-process CR registries so list endpoints don't
        # surface leftover live entries.
        _LIVE_CR_REGISTRY.clear()
        _LIVE_SWEEP_REGISTRY.clear()

    # ---- navigation helpers ------------------------------------------------

    def goto(self, hash_path: str, *, timeout_ms: int = 15_000) -> Page:
        """Navigate to ``<base_url>/#<hash_path>`` and wait for networkidle."""
        url = f"{self.base_url}/#{hash_path}"
        self.page.goto(url, wait_until="networkidle", timeout=timeout_ms)
        return self.page

    def goto_job_detail(self, ns: str, name: str, *, epoch: str | None = None) -> Page:
        path = f"/jobs/{quote(ns)}/{quote(name)}"
        if epoch is not None:
            path += f"/runs/{quote(epoch)}"
        return self.goto(path)

    def goto_sweep_detail(
        self, ns: str, name: str, *, epoch: str | None = None
    ) -> Page:
        path = f"/sweeps/{quote(ns)}/{quote(name)}"
        if epoch is not None:
            path += f"/runs/{quote(epoch)}"
        return self.goto(path)

    def goto_jobs_list(self) -> Page:
        return self.goto("/jobs")

    def goto_sweeps_list(self) -> Page:
        return self.goto("/sweeps")

    def goto_dashboard(self) -> Page:
        return self.goto("/")

    # ---- API helpers (synchronous so tests don't await) --------------------

    def api_get(self, path: str, *, timeout: float = 5.0) -> tuple[int, bytes]:
        """Issue a GET against the live operator server and return (status, body)."""
        url = f"{self.base_url}{path}"
        try:
            with _urlopen_local(url, timeout=timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    # ---- assertion shortcuts ----------------------------------------------

    def assert_no_unreachable_banner(self) -> None:
        """Fail loudly if the SPA shows 'Operator API unreachable'."""
        body = self.page.locator("body").inner_text(timeout=5_000)
        assert "Operator API unreachable" not in body, (
            f"SPA stalled on 'Operator API unreachable' banner.\n\n"
            f"--- body text ---\n{body[:2000]}"
        )

    def assert_no_console_errors(
        self, *, allow_substrings: tuple[str, ...] = ()
    ) -> None:
        """Fail if the browser logged any console.error since page setup.

        ``allow_substrings`` filters out known-noisy errors per test (e.g.
        "Failed to load resource: ... 404" when a test intentionally seeds
        only partial data and the UI requests an optional endpoint).
        """
        unexpected = [
            e for e in self.console_errors if not any(s in e for s in allow_substrings)
        ]
        assert not unexpected, "Unexpected console errors:\n  " + "\n  ".join(
            unexpected
        )


@pytest.fixture
def harness(
    _operator_server: _ServerHandle, _browser: Browser, request: pytest.FixtureRequest
) -> Iterator[Harness]:
    """Per-test Playwright + uvicorn handle. See :class:`Harness`."""
    context = _browser.new_context(viewport={"width": 1600, "height": 1000})
    page = context.new_page()

    console_errors: list[str] = []
    bad_responses: list[str] = []

    page.on(
        "console",
        lambda msg: console_errors.append(f"[{msg.type}] {msg.text}")
        if msg.type in ("error",)
        else None,
    )
    page.on(
        "pageerror",
        lambda exc: console_errors.append(f"[pageerror] {exc}"),
    )

    def _on_response(resp):
        if resp.status >= 400:
            bad_responses.append(f"{resp.status} {resp.request.method} {resp.url}")

    page.on("response", _on_response)

    # Unique ns per test — node-id has good entropy + reproducibility.
    test_tag = uuid.uuid5(uuid.NAMESPACE_URL, request.node.nodeid).hex[:12]
    ns = f"ns-{test_tag}"

    h = Harness(
        page=page,
        results_dir=_operator_server.results_dir,
        ns=ns,
        base_url=_operator_server.base_url,
        console_errors=console_errors,
        bad_responses=bad_responses,
    )
    try:
        yield h
    finally:
        # Strip any live-CR registrations this test added so the next test
        # starts with an empty registry.
        for key in list(_LIVE_CR_REGISTRY):
            if key[0] == ns:
                _LIVE_CR_REGISTRY.pop(key, None)
        for key in list(_LIVE_SWEEP_REGISTRY):
            if key[0] == ns:
                _LIVE_SWEEP_REGISTRY.pop(key, None)
        context.close()


# ---------------------------------------------------------------------------
# Apply the e2e marker to every test in this directory + auto-skip if no
# Playwright. This is a pytest collection hook; tests don't need decorators.
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    skip_marker = pytest.mark.skipif(
        not _PLAYWRIGHT_AVAILABLE, reason=_PLAYWRIGHT_REASON
    )
    e2e_marker = pytest.mark.e2e
    for item in items:
        if Path(str(item.fspath)).is_relative_to(Path(__file__).parent):
            item.add_marker(skip_marker)
            item.add_marker(e2e_marker)
