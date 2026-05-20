# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest plumbing for the chaos_dynamo (D-series) scenario suite.

Composes three layers of fixtures:

1. **Cluster + dynamo deployment** — re-exports
   ``dynamo_operator``, ``dynamo_config``, ``dynamo_server``,
   ``dynamo_endpoint_url`` from :py:mod:`tests.kubernetes.gpu.dynamo.conftest`.
   The re-export is explicit because ``tests/kubernetes/gpu/`` is the
   ``scope="package"`` boundary for the gpu suite (its ``__init__.py``
   forbids subdirectory ``__init__.py`` files), so pytest's conftest discovery
   does NOT walk from a sibling ``chaos_dynamo/`` into ``gpu/dynamo/``.

2. **Toxiproxy** — a package-scoped :py:data:`dynamo_toxiproxy` fixture that
   deploys :py:mod:`tests.kubernetes.chaos_common.fixtures.toxiproxy` (the
   expanded port-pool manifest with 17 named ports in the ``chaos-toxiproxy``
   namespace) and opens a kubectl port-forward to the admin API. Distinct
   from the legacy ``toxiproxy_injector`` in :py:mod:`tests.kubernetes.chaos.conftest`
   which uses a different namespace + port layout for the AIPerf-only suite.

3. **Unified faults registry** — overrides the
   :py:data:`tests.kubernetes.chaos_common.conftest.faults` fixture with a
   function-scoped registry pre-loaded with every concrete injector the
   D-series tests need (pod, workload, crd, network, store, process, client,
   cluster). The ``CRDInjector`` is parameterized at registration time for
   ``DynamoGraphDeployment`` / ``nvidia.com`` / ``dynamo-system``.

   Pytest fixture resolution prefers the conftest *closest* to the test
   file, so ``tests/kubernetes/chaos_dynamo/test_*.py`` resolves ``faults``
   from this module rather than the echo-only definition under
   :py:mod:`tests.kubernetes.chaos_common.conftest`. Adapter unit tests
   under ``chaos_common/`` continue to see the echo-only registry.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import aiohttp
import pytest
import pytest_asyncio

from aiperf.common.aiperf_logger import AIPerfLogger

# Re-exported dynamo fixtures. pytest discovers fixtures by attribute lookup
# on conftest modules; importing them here makes ``dynamo_operator`` etc.
# available to every test file in this package. The noqa comments on the
# fixture imports are load-bearing — ruff cannot see that pytest uses the
# names dynamically.
from tests.kubernetes.chaos.toxiproxy import (
    TOXIPROXY_ADMIN_PORT,
    ToxiproxyError,
    ToxiproxyInjector,
)
from tests.kubernetes.chaos_common.injectors.client import ClientInjector
from tests.kubernetes.chaos_common.injectors.cluster import ClusterInjector
from tests.kubernetes.chaos_common.injectors.crd import CRDInjector
from tests.kubernetes.chaos_common.injectors.network import NetworkInjector
from tests.kubernetes.chaos_common.injectors.pod import PodInjector
from tests.kubernetes.chaos_common.injectors.process import ProcessInjector
from tests.kubernetes.chaos_common.injectors.store import StoreInjector
from tests.kubernetes.chaos_common.injectors.workload import WorkloadInjector
from tests.kubernetes.chaos_common.registry import InjectorRegistry
from tests.kubernetes.gpu.dynamo.conftest import (
    dynamo_config,  # noqa: F401
    dynamo_endpoint_url,  # noqa: F401
    dynamo_operator,  # noqa: F401
    dynamo_server,  # noqa: F401
)
from tests.kubernetes.helpers.kubectl import KubectlClient

logger = AIPerfLogger(__name__)


# ============================================================================
# Toxiproxy fixture (package-scoped)
# ============================================================================

DYNAMO_TOXIPROXY_NAMESPACE = "chaos-toxiproxy"
"""Namespace declared by ``tests/kubernetes/chaos_common/fixtures/toxiproxy.yaml``."""

DYNAMO_TOXIPROXY_SERVICE = "toxiproxy"
"""Service / Deployment name inside the chaos-toxiproxy namespace."""

_TOXIPROXY_MANIFEST = (
    Path(__file__).parent.parent / "chaos_common" / "fixtures" / "toxiproxy.yaml"
)
"""Absolute path to the expanded port-pool toxiproxy manifest."""


@pytest_asyncio.fixture(scope="package", loop_scope="package")
async def dynamo_toxiproxy(
    kubectl: KubectlClient,
) -> AsyncIterator[ToxiproxyInjector]:
    """Package-scoped toxiproxy deployment for the D-series suite.

    Applies :py:data:`_TOXIPROXY_MANIFEST` (the 17-port expanded fixture),
    waits for the deployment to roll out, opens a port-forward to the admin
    REST API (:8474), and yields a ready-to-use :py:class:`ToxiproxyInjector`.
    Tests must call ``await dynamo_toxiproxy.reset()`` in their own
    ``finally`` to keep proxies/toxics from leaking across cases.

    Distinct from the legacy ``toxiproxy_injector`` in
    :py:mod:`tests.kubernetes.chaos.conftest`: that one uses a different
    namespace (``aiperf-chaos-toxiproxy``) and a smaller AIPerf-only port
    layout. The two cannot share a fixture because the AIPerf suite and the
    Dynamo suite run in separate package scopes and we want each pytest
    session to be able to install whichever one its tests need without
    pulling in the other.
    """
    injector = ToxiproxyInjector()
    await _ensure_toxiproxy_deployed(injector, kubectl)
    try:
        yield injector
    finally:
        if not _skip_cleanup():
            await _teardown_toxiproxy(injector, kubectl)
        else:
            logger.info(
                "skipping dynamo_toxiproxy teardown (AIPERF_K8S_SKIP_CLEANUP=1)"
            )


def _skip_cleanup() -> bool:
    """Honour ``AIPERF_K8S_SKIP_CLEANUP`` so chained test runs reuse infra."""
    import os

    return os.environ.get("AIPERF_K8S_SKIP_CLEANUP", "").lower() in ("1", "true", "yes")


async def _ensure_toxiproxy_deployed(
    injector: ToxiproxyInjector,
    kubectl: KubectlClient,
) -> None:
    """Apply the chaos_common toxiproxy manifest and port-forward the admin API.

    This is a parallel of :py:meth:`ToxiproxyInjector.ensure_deployed` adapted
    for the chaos_common manifest (different namespace + service name). The
    class default reads its own manifest path, so we re-do the apply +
    port-forward here to point at ours instead of subclassing.
    """
    manifest = _TOXIPROXY_MANIFEST.read_text()
    await kubectl.apply(manifest)
    ok = await kubectl.wait_for_rollout(
        "deployment",
        DYNAMO_TOXIPROXY_SERVICE,
        namespace=DYNAMO_TOXIPROXY_NAMESPACE,
        timeout=300,
    )
    if not ok:
        logs = await kubectl.get_logs(
            f"deployment/{DYNAMO_TOXIPROXY_SERVICE}",
            namespace=DYNAMO_TOXIPROXY_NAMESPACE,
        )
        raise RuntimeError(f"dynamo_toxiproxy rollout failed; logs:\n{logs}")

    pod_res = await kubectl.run(
        "get",
        "pods",
        "-n",
        DYNAMO_TOXIPROXY_NAMESPACE,
        "-l",
        f"app={DYNAMO_TOXIPROXY_SERVICE}",
        "-o",
        "jsonpath={.items[0].metadata.name}",
        check=True,
    )
    pod = pod_res.stdout.strip()
    if not pod:
        raise RuntimeError(
            f"dynamo_toxiproxy pod not found after rollout in {DYNAMO_TOXIPROXY_NAMESPACE!r}"
        )

    # Reuse the injector's private port-forward stack so its existing
    # teardown logic owns the lifecycle. This is the only place we touch
    # _pf_stack / _base_url on the injector instance.
    pf_stack: AsyncExitStack = AsyncExitStack()
    local_port = await pf_stack.enter_async_context(
        kubectl.port_forward(
            pod, TOXIPROXY_ADMIN_PORT, namespace=DYNAMO_TOXIPROXY_NAMESPACE
        )
    )
    injector._pf_stack = pf_stack  # noqa: SLF001 - intentional; mirrors ensure_deployed
    injector._base_url = f"http://127.0.0.1:{local_port}"  # noqa: SLF001

    # Reachability probe — short-lived session per attempt.
    for attempt in range(10):
        try:
            async with (
                aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=5.0)
                ) as session,
                session.get(f"{injector.base_url}/version") as resp,
            ):
                if resp.status == 200:
                    version = (await resp.text()).strip()
                    logger.info(
                        f"dynamo_toxiproxy reachable at {injector.base_url} "
                        f"(version={version})"
                    )
                    return
        except aiohttp.ClientError as exc:
            logger.debug(lambda exc=exc, a=attempt: f"toxiproxy probe {a}: {exc!r}")
        await asyncio.sleep(0.5)
    raise RuntimeError(
        f"dynamo_toxiproxy admin API did not respond at "
        f"{injector.base_url}/version after 5s; check "
        f"`kubectl get pods -n {DYNAMO_TOXIPROXY_NAMESPACE}`"
    )


async def _teardown_toxiproxy(
    injector: ToxiproxyInjector,
    kubectl: KubectlClient,
) -> None:
    """Close the port-forward stack and force-delete the namespace.

    Wrapped in try/except so a flaky cluster cannot mask the actual test
    failure; the session-scope chaos sweeper in chaos_common/conftest.py
    will catch leftover ``chaos-toxiproxy`` debris at session exit anyway.
    """
    try:
        if injector._pf_stack is not None:  # noqa: SLF001
            await injector._pf_stack.aclose()  # noqa: SLF001
            injector._pf_stack = None  # noqa: SLF001
        injector._base_url = None  # noqa: SLF001
    except Exception as exc:
        logger.warning(lambda exc=exc: f"dynamo_toxiproxy pf close failed: {exc!r}")
    try:
        await kubectl.delete_namespace(DYNAMO_TOXIPROXY_NAMESPACE, wait=False)
    except Exception as exc:
        logger.warning(lambda exc=exc: f"dynamo_toxiproxy ns delete failed: {exc!r}")


# ============================================================================
# Unified faults registry (function-scoped, overrides chaos_common.conftest)
# ============================================================================


@pytest_asyncio.fixture
async def faults(
    kubectl: KubectlClient,
    dynamo_toxiproxy: ToxiproxyInjector,
) -> AsyncIterator[InjectorRegistry]:
    """Per-test :py:class:`InjectorRegistry` wired for the D-series suite.

    Pre-registers every concrete injector the D-series scenarios use:

    * :py:class:`PodInjector` — ``pod.*`` (kill, kill_container, kill_pid)
    * :py:class:`WorkloadInjector` — ``workload.*`` (restart, scale)
    * :py:class:`CRDInjector` — ``crd.*`` and ``operator.*`` against the
      Dynamo operator (``DynamoGraphDeployment`` in ``dynamo-system``)
    * :py:class:`NetworkInjector` — ``network.*`` (latency, timeout,
      bandwidth, partition, slow_close) via the shared toxiproxy
    * :py:class:`StoreInjector` — ``store.{etcd,nats}.*`` faults
    * :py:class:`ProcessInjector` — ``process.signal`` (SIGSTOP/SIGCONT, ...)
    * :py:class:`ClientInjector` — in-process ``client.*`` faults
    * :py:class:`ClusterInjector` — ``cluster.*`` (resource_quota,
      network_policy.deny_egress, rbac.revoke)

    The ordering follows the spec's "more specific first" rule — currently
    every injector has a disjoint ``HANDLES`` tuple, so registration order
    only matters if a future injector overlaps domains. Cleanup is the
    registry's own LIFO restore; per-fixture cleanup is implicit because
    each injector is constructed fresh per test.
    """
    reg = InjectorRegistry()
    reg.register(PodInjector(kubectl))
    reg.register(WorkloadInjector(kubectl))
    reg.register(
        CRDInjector(
            kubectl,
            cr_kind="dynamographdeployment",
            cr_api_group="nvidia.com",
            operator_namespace="dynamo-system",
            operator_selector="app.kubernetes.io/name=dynamo-operator",
        )
    )
    reg.register(NetworkInjector(dynamo_toxiproxy))
    reg.register(StoreInjector(kubectl, dynamo_toxiproxy))
    reg.register(ProcessInjector(kubectl))
    reg.register(ClientInjector())
    reg.register(ClusterInjector(kubectl))
    try:
        yield reg
    finally:
        # Per-test toxiproxy reset so leftover toxics never bleed across cases.
        # Wrapped because reset() over a torn-down port-forward should not
        # mask the original test failure.
        try:
            await dynamo_toxiproxy.reset()
        except (ToxiproxyError, aiohttp.ClientError, RuntimeError) as exc:
            logger.warning(lambda exc=exc: f"faults teardown reset failed: {exc!r}")


# ============================================================================
# D-series helpers (plain async functions, not fixtures)
# ============================================================================


async def wait_for_dgd_state(
    kubectl: KubectlClient,
    name: str,
    namespace: str,
    target_state: str,
    *,
    timeout: float = 300.0,
    poll_interval: float = 2.0,
) -> str:
    """Poll a ``DynamoGraphDeployment`` until ``status.state`` matches.

    Used by D101 (operator-kill recovery), D104 (operator-reinstall recovery),
    D701 (DGD-spec mutation) and other scenarios that assert the operator
    drives the CR back to a known state after a fault.

    Args:
        kubectl: Package-scoped :py:class:`KubectlClient`.
        name: ``DynamoGraphDeployment`` resource name.
        namespace: CR namespace (typically the dynamo-server deployment ns).
        target_state: Expected value of ``status.state`` (e.g. ``"successful"``,
            ``"pending"``, ``"failed"``).
        timeout: Total seconds to wait before raising.
        poll_interval: Seconds between polls.

    Returns:
        The observed ``status.state`` once it matches ``target_state``.

    Raises:
        TimeoutError: When ``target_state`` is not observed within ``timeout``;
            includes the last observed state in the message so the failure
            report points at the actual transition that did not happen.

    Example::

        await wait_for_dgd_state(
            kubectl,
            name="dynamo-agg",
            namespace="dynamo-server",
            target_state="successful",
            timeout=180.0,
        )
    """
    deadline = asyncio.get_event_loop().time() + timeout
    last_state: str = "<unobserved>"
    while True:
        result = await kubectl.run(
            "get",
            "dynamographdeployment",
            name,
            "-n",
            namespace,
            "-o",
            "jsonpath={.status.state}",
            check=False,
        )
        if result.returncode == 0:
            last_state = result.stdout.strip() or "<empty>"
            if last_state == target_state:
                return last_state
        if asyncio.get_event_loop().time() >= deadline:
            raise TimeoutError(
                f"DynamoGraphDeployment {namespace}/{name} did not reach "
                f"state={target_state!r} within {timeout}s "
                f"(last observed state={last_state!r})"
            )
        await asyncio.sleep(poll_interval)


async def scrape_frontend_metrics(
    kubectl: KubectlClient,
    namespace: str,
    *,
    deployment_name: str = "dynamo-agg-frontend",
    metrics_port: int = 8000,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Scrape and parse the dynamo frontend's ``/metrics`` endpoint.

    Opens a short-lived ``kubectl port-forward`` to the frontend pod and
    issues a single HTTP GET. Returns a dict keyed by metric name with the
    raw parsed float value (the latest sample wins for histograms; tests
    that need bucket-level data should fetch the raw text themselves).

    Stub-grade — D803 and similar tests are expected to extend this with
    label-aware parsing as the scenarios land. The structure is here so
    test files can call ``await scrape_frontend_metrics(...)`` without
    inventing their own port-forward dance.

    Args:
        kubectl: Package-scoped :py:class:`KubectlClient`.
        namespace: Dynamo deployment namespace (e.g. ``"dynamo-server"``).
        deployment_name: Frontend Deployment name (depends on the configured
            ``DynamoMode``; defaults to the aggregated topology).
        metrics_port: Frontend container port hosting ``/metrics``.
        timeout: Per-request HTTP timeout in seconds.

    Returns:
        ``{metric_name: float, ...}`` parsed from the Prometheus text
        exposition format. Comment lines (``# HELP`` / ``# TYPE``) are
        skipped; lines that cannot be split into ``name value`` are skipped
        with a debug log.

    Raises:
        RuntimeError: When no pod matches ``deployment_name`` in ``namespace``,
            or when the ``/metrics`` GET returns a non-200 status.

    Example::

        metrics = await scrape_frontend_metrics(
            kubectl,
            namespace="dynamo-server",
            deployment_name="dynamo-disagg-frontend",
        )
        assert metrics.get("dynamo_requests_total", 0.0) > 0
    """
    pod_res = await kubectl.run(
        "get",
        "pods",
        "-n",
        namespace,
        "-l",
        f"app.kubernetes.io/component={deployment_name}",
        "-o",
        "jsonpath={.items[0].metadata.name}",
        check=False,
    )
    pod = pod_res.stdout.strip() if pod_res.returncode == 0 else ""
    if not pod:
        # Fall back to name-prefix search — the operator labels frontend pods
        # with the component name, but older Dynamo releases used a different
        # label schema. Listing by namespace + filtering on the client side is
        # cheap (one kubectl call) and keeps this helper working across both.
        pods = await kubectl.get_pods(namespace)
        for candidate in pods:
            if deployment_name in candidate.name and candidate.is_ready:
                pod = candidate.name
                break
    if not pod:
        raise RuntimeError(
            f"scrape_frontend_metrics: no ready pod matching "
            f"{deployment_name!r} in namespace {namespace!r}"
        )

    async with kubectl.port_forward(pod, metrics_port, namespace=namespace) as local:
        url = f"http://127.0.0.1:{local}/metrics"
        async with (
            aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session,
            session.get(url) as resp,
        ):
            if resp.status != 200:
                body = (await resp.read()).decode(errors="replace")[:512]
                raise RuntimeError(
                    f"scrape_frontend_metrics: GET {url} -> {resp.status}; "
                    f"body={body!r}"
                )
            text = await resp.text()

    return _parse_prometheus_text(text)


def _parse_prometheus_text(text: str) -> dict[str, float]:
    """Parse Prometheus text exposition format into a flat name -> value map.

    Histogram + summary lines (``foo_bucket{le="..."}``) collapse to the metric
    NAME without labels, with the LAST observed value winning. Sufficient for
    presence / monotonic-increase assertions; tests that need label-keyed
    series should parse ``text`` themselves.
    """
    out: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Strip label set (between { and }) so different label combinations
        # collapse to the same key — see docstring.
        name_end = len(line)
        brace = line.find("{")
        if brace != -1:
            close = line.find("}", brace + 1)
            if close == -1:
                logger.debug(
                    lambda line=line: f"prom parse: unterminated label set: {line!r}"
                )
                continue
            name_end = brace
            rest = line[close + 1 :].strip()
        else:
            # No labels: split on the first whitespace run.
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            name_end = len(parts[0])
            rest = parts[1]
        name = line[:name_end].strip()
        # Value is the first whitespace-separated token of `rest`; the
        # optional trailing timestamp is ignored.
        value_tok = rest.split(None, 1)[0] if rest else ""
        try:
            out[name] = float(value_tok)
        except ValueError:
            logger.debug(lambda line=line: f"prom parse: non-numeric value in {line!r}")
    return out


# ============================================================================
# D-series-specific helper fixtures
# ============================================================================


@pytest.fixture(scope="package")
def dynamo_deployment_namespace(dynamo_config: Any) -> str:  # noqa: ANN401
    """Resolve the namespace the D-series tests should target.

    Read from ``dynamo_config.namespace`` rather than hard-coding so the
    suite tracks whatever the deployment fixtures actually created (the
    ``--gpu-dynamo-*`` CLI flags can override the default).

    The ``Any`` on ``dynamo_config`` is intentional — pytest's resolver
    erases the dataclass to ``Any`` when wired across module boundaries,
    and importing :py:class:`DynamoConfig` here purely for a parameter
    annotation would pull the entire gpu dependency surface into chaos_dynamo.
    """
    return dynamo_config.namespace


# Symbols re-exported for static analysis. pytest only needs them as
# module-level attributes, but enumerating them here is a load-bearing
# breadcrumb for readers wondering "where does ``dynamo_operator`` come from?".
__all__: list[str] = [
    "DYNAMO_TOXIPROXY_NAMESPACE",
    "DYNAMO_TOXIPROXY_SERVICE",
    "dynamo_config",
    "dynamo_deployment_namespace",
    "dynamo_endpoint_url",
    "dynamo_operator",
    "dynamo_server",
    "dynamo_toxiproxy",
    "faults",
    "scrape_frontend_metrics",
    "wait_for_dgd_state",
]
