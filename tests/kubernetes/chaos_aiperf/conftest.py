# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest plumbing for the chaos_aiperf scenario suite.

Sibling of :py:mod:`tests.kubernetes.chaos_dynamo`. This module hosts the
unified-API ports of legacy AIPerf chaos scenarios that live in
:py:mod:`tests.kubernetes.chaos`. Composition is three layers:

1. **Re-exported legacy chaos fixtures** -- direct fixture imports from
   :py:mod:`tests.kubernetes.chaos.conftest` (importing a fixture function
   into a conftest registers it for the package) so a ported test can request
   ``chaos_injector``, ``toxiproxy_injector``, ``mock_server_injector``,
   ``operator_ready_toxiproxy_routed`` and
   ``operator_ready_apiserver_toxiproxy_routed`` exactly as legacy tests do.
   ``operator_ready``, ``kubectl`` and ``operator_job_namespace`` arrive via
   the parent package-scoped :py:mod:`tests.kubernetes.conftest`.

2. **Unified faults registry** -- overrides the echo-only ``faults`` fixture
   from :py:mod:`tests.kubernetes.chaos_common.conftest` with a per-test
   :py:class:`InjectorRegistry` pre-loaded with every concrete injector the
   ported AIPerf scenarios need. The :py:class:`CRDInjector` is parameterized
   for ``aiperfjob`` / ``aiperf.nvidia.com`` and points at
   :py:data:`DEFAULT_OPERATOR_NAMESPACE` (``aiperf-system``). The
   :py:class:`NetworkInjector` is wired to the legacy ``toxiproxy_injector``
   (namespace ``aiperf-chaos-toxiproxy``, AIPerf-only port pool covering
   the reserved C15/C16/B3 ports 20000/20002/20010) -- explicitly NOT the
   ``dynamo_toxiproxy`` from chaos_dynamo, so a chaos_aiperf run never has
   to deploy the chaos_common manifest.

3. **Async helpers** -- :py:func:`wait_for_aiperfjob_phase` polls
   ``.status.phase`` / ``.status.currentPhase`` with the same JSONPath
   shape and TimeoutError message style as the legacy
   :py:meth:`ChaosInjector.wait_for_phase`. :py:func:`scrape_aiperf_metrics`
   is a stub over the operator's Prometheus ``/metrics`` surface that
   ported metrics-shape scenarios can extend.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import pytest_asyncio

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.kubernetes.constants import DEFAULT_OPERATOR_NAMESPACE

# Re-export the legacy AIPerf chaos fixtures by importing them directly:
# importing a fixture function into a conftest module registers it for this
# package. Modern pytest rejects ``pytest_plugins`` in any non-rootdir
# conftest, so plugin-loading tests.kubernetes.chaos.conftest is no longer an
# option. ``operator_ready``/``kubectl``/``operator_job_namespace`` still
# arrive via the parent tests/kubernetes/conftest.py hierarchy.
from tests.kubernetes.chaos.conftest import (  # noqa: F401
    chaos_injector,
    mock_server_injector,
    operator_ready_apiserver_toxiproxy_routed,
    operator_ready_toxiproxy_routed,
    toxiproxy_injector,
)
from tests.kubernetes.chaos.toxiproxy import ToxiproxyError, ToxiproxyInjector
from tests.kubernetes.chaos_common.injectors.client import ClientInjector
from tests.kubernetes.chaos_common.injectors.cluster import ClusterInjector
from tests.kubernetes.chaos_common.injectors.crd import CRDInjector
from tests.kubernetes.chaos_common.injectors.network import NetworkInjector
from tests.kubernetes.chaos_common.injectors.pod import PodInjector
from tests.kubernetes.chaos_common.injectors.process import ProcessInjector
from tests.kubernetes.chaos_common.injectors.store import StoreInjector
from tests.kubernetes.chaos_common.injectors.workload import WorkloadInjector
from tests.kubernetes.chaos_common.registry import InjectorRegistry
from tests.kubernetes.helpers.kubectl import KubectlClient

logger = AIPerfLogger(__name__)


AIPERF_OPERATOR_NAMESPACE = DEFAULT_OPERATOR_NAMESPACE
"""Chart-default operator namespace used by the CRDInjector wiring below."""

AIPERF_OPERATOR_SELECTOR = "app.kubernetes.io/name=aiperf-operator"
"""kubectl ``-l`` selector that uniquely identifies the operator Pod."""


# ============================================================================
# Unified faults registry (function-scoped, overrides chaos_common.conftest)
# ============================================================================


@pytest_asyncio.fixture
async def faults(
    kubectl: KubectlClient,
    toxiproxy_injector: ToxiproxyInjector,  # noqa: F811 — pytest fixture request; shadows the re-export import by design
) -> AsyncIterator[InjectorRegistry]:
    """Per-test :py:class:`InjectorRegistry` wired for the chaos_aiperf suite.

    Pre-registers every concrete injector the ported AIPerf scenarios use:

    * :py:class:`PodInjector` -- ``pod.*`` (kill, kill_container, kill_pid).
    * :py:class:`WorkloadInjector` -- ``workload.*`` (restart,
      rolling_upgrade, scale, set_env) used by the benchmark-runtime B1/B2
      scenarios against the mock-server Deployment.
    * :py:class:`CRDInjector` -- ``crd.*`` / ``operator.*`` against the AIPerf
      operator (``aiperfjob`` in ``aiperf-system``). Used by C10 (delete +
      recreate), C11 (parallel delete subset), C12 (invalid spec).
    * :py:class:`NetworkInjector` -- ``network.*`` (latency, timeout,
      bandwidth, reset_peer, slow_close, partition) via the legacy
      ``toxiproxy_injector`` (namespace ``aiperf-chaos-toxiproxy``). Used
      by B3 latency injection and C15/C16 reachability scenarios.
    * :py:class:`StoreInjector` -- ``store.{etcd,nats}.*``. AIPerf does not
      bundle etcd or NATS as first-class dependencies, but the injector is
      registered with its defaults so scenarios that exercise them against
      a co-deployed Dynamo backend continue to work.
    * :py:class:`ProcessInjector` -- ``process.signal`` (SIGSTOP/SIGCONT, ...).
    * :py:class:`ClientInjector` -- in-process ``client.*`` faults.
    * :py:class:`ClusterInjector` -- ``cluster.*`` (resource_quota,
      network_policy.deny_egress, rbac.revoke).

    Cleanup: registry-owned faults are LIFO-restored via the unified API;
    additionally toxiproxy is ``reset()`` so leftover toxics / proxies do
    not bleed across cases. The reset is wrapped because reset() over a
    torn-down port-forward must not mask the original test exception.
    """
    reg = InjectorRegistry()
    reg.register(PodInjector(kubectl))
    reg.register(WorkloadInjector(kubectl))
    reg.register(
        CRDInjector(
            kubectl,
            cr_kind="aiperfjob",
            cr_api_group="aiperf.nvidia.com",
            operator_namespace=AIPERF_OPERATOR_NAMESPACE,
            operator_selector=AIPERF_OPERATOR_SELECTOR,
        )
    )
    reg.register(NetworkInjector(toxiproxy_injector))
    reg.register(StoreInjector(kubectl, toxiproxy_injector))
    reg.register(ProcessInjector(kubectl))
    reg.register(ClientInjector())
    reg.register(ClusterInjector(kubectl))
    try:
        yield reg
    finally:
        try:
            await toxiproxy_injector.reset()
        except (ToxiproxyError, aiohttp.ClientError, RuntimeError) as exc:
            logger.warning(lambda exc=exc: f"faults teardown reset failed: {exc!r}")


# ============================================================================
# chaos_aiperf helpers (plain async functions, not fixtures)
# ============================================================================


async def wait_for_aiperfjob_phase(
    kubectl: KubectlClient,
    namespace: str,
    name: str,
    phases: tuple[str, ...],
    *,
    timeout: float = 180.0,
    current_phase: str | None = None,
    poll_interval: float = 1.0,
) -> str:
    """Poll an AIPerfJob until ``.status.phase`` matches one of ``phases``.

    Free async helper that does not require a
    :py:class:`tests.kubernetes.chaos.chaos_injector.ChaosInjector`
    instance. The JSONPath shape (``{.status.phase}|{.status.currentPhase}``)
    and TimeoutError wording mirror :py:meth:`ChaosInjector.wait_for_phase`
    on purpose so ported scenarios produce comparable failure reports.

    When ``current_phase`` is set, the AIPerfJob's ``.status.currentPhase``
    must also match it (e.g. ``current_phase="profiling"`` to catch the
    actively-benchmarking state). Pass ``None`` to ignore sub-phase.

    Args:
        kubectl: Async kubectl wrapper pinned to the chaos cluster.
        namespace: AIPerfJob CR namespace.
        name: AIPerfJob CR name.
        phases: Acceptable terminal/intermediate phases (e.g.
            ``("Initializing", "Running")`` or ``("Completed",)``).
        timeout: Max seconds to wait before raising :py:class:`TimeoutError`.
        current_phase: Optional ``.status.currentPhase`` constraint.
        poll_interval: Seconds between polls.

    Returns:
        The phase string that was observed when the predicate first matched.

    Raises:
        TimeoutError: When the predicate is not satisfied within ``timeout``.
            Message includes the last observed ``(phase, currentPhase)`` so
            the failure report points at the actual transition that did not
            happen.

    Example::

        phase = await wait_for_aiperfjob_phase(
            kubectl,
            namespace="aiperf-test-c10",
            name="chaos-c10",
            phases=("Completed",),
            timeout=300.0,
        )
        assert phase == "Completed"
    """
    deadline = time.monotonic() + timeout
    observed_phase = ""
    observed_current_phase = ""
    polls = 0
    failed_polls = 0
    last_stderr = ""
    while time.monotonic() < deadline:
        res = await kubectl.run(
            "get",
            "aiperfjob",
            name,
            "-n",
            namespace,
            "-o",
            "jsonpath={.status.phase}|{.status.currentPhase}",
            check=False,
        )
        polls += 1
        if res.returncode != 0:
            failed_polls += 1
            last_stderr = res.stderr.strip()
        phase, _, curr = res.stdout.strip().partition("|")
        observed_phase = phase
        observed_current_phase = curr
        if phase in phases and (current_phase is None or curr == current_phase):
            return phase
        await asyncio.sleep(poll_interval)
    # An empty phase is ambiguous on its own: the CR may not exist, may not be
    # readable, or may exist with no .status yet. Say which -- otherwise all
    # three read as phase='' and no amount of re-running distinguishes them.
    if polls and failed_polls == polls:
        detail = (
            f"the CR was never readable -- all {polls} kubectl get(s) failed, "
            f"last stderr={last_stderr!r}"
        )
    elif not observed_phase:
        detail = (
            "the CR was readable but never had .status.phase set "
            "(operator did not reconcile it)"
        )
    else:
        detail = (
            f"observed phase={observed_phase!r}, "
            f"currentPhase={observed_current_phase!r}"
        )
    raise TimeoutError(
        f"AIPerfJob {namespace}/{name} did not reach phase {phases} "
        f"(currentPhase={current_phase!r}) within {timeout} s ({detail})"
    )


async def scrape_aiperf_metrics(
    kubectl: KubectlClient,
    namespace: str,
    *,
    deployment_name: str = "aiperf-operator",
    metrics_port: int = 9090,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Scrape and parse the AIPerf operator's ``/metrics`` endpoint.

    Opens a short-lived ``kubectl port-forward`` to the operator pod's
    Prometheus port and issues a single HTTP GET. Returns a dict keyed by
    metric name with the latest parsed float (label combinations collapse
    to the bare metric name; the last observed value wins). Suitable for
    presence and monotonic-increase assertions.

    Extension note: the AIPerf operator's Prometheus surface is owned by
    :py:mod:`aiperf.operator.metrics` and exposes kopf reconcile counters.
    For scenarios asserting on benchmark-runtime metrics
    (records-manager, workers, the controller HTTP API surface), extend
    this helper to accept an alternate pod selector and preserve label
    information instead of collapsing to bare names.

    Args:
        kubectl: Package-scoped :py:class:`KubectlClient`.
        namespace: Namespace hosting the operator (typically
            :py:data:`AIPERF_OPERATOR_NAMESPACE`).
        deployment_name: Operator Deployment name (chart default).
        metrics_port: Operator metrics port (chart default 9090; the
            ``OperatorEnvironment.METRICS_PORT`` env var overrides this).
        timeout: Per-request HTTP timeout in seconds.

    Returns:
        ``{metric_name: float, ...}`` parsed from the Prometheus text
        exposition format.

    Raises:
        RuntimeError: When no pod matches ``deployment_name`` in
            ``namespace``, or when the ``/metrics`` GET returns non-200.

    Example::

        metrics = await scrape_aiperf_metrics(
            kubectl,
            namespace=AIPERF_OPERATOR_NAMESPACE,
        )
        assert metrics.get("kopf_handler_invocations_total", 0.0) > 0.0
    """
    pod_res = await kubectl.run(
        "get",
        "pods",
        "-n",
        namespace,
        "-l",
        f"app.kubernetes.io/name={deployment_name}",
        "-o",
        "jsonpath={.items[0].metadata.name}",
        check=False,
    )
    pod = pod_res.stdout.strip() if pod_res.returncode == 0 else ""
    if not pod:
        raise RuntimeError(
            f"scrape_aiperf_metrics: no pod matching "
            f"app.kubernetes.io/name={deployment_name!r} in namespace {namespace!r}"
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
                    f"scrape_aiperf_metrics: GET {url} -> {resp.status}; body={body!r}"
                )
            text = await resp.text()

    return _parse_prometheus_text(text)


def _parse_prometheus_text(text: str) -> dict[str, float]:
    """Parse Prometheus text exposition format into a flat name -> value map.

    Histogram + summary lines (``foo_bucket{le="..."}``) collapse to the
    metric NAME without labels, with the LAST observed value winning.
    Sufficient for presence / monotonic-increase assertions; tests that need
    label-keyed series should parse ``text`` themselves.
    """
    out: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
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
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            name_end = len(parts[0])
            rest = parts[1]
        name = line[:name_end].strip()
        value_tok = rest.split(None, 1)[0] if rest else ""
        try:
            out[name] = float(value_tok)
        except ValueError:
            logger.debug(lambda line=line: f"prom parse: non-numeric value in {line!r}")
    return out


__all__: list[str] = [
    "AIPERF_OPERATOR_NAMESPACE",
    "AIPERF_OPERATOR_SELECTOR",
    "faults",
    "scrape_aiperf_metrics",
    "wait_for_aiperfjob_phase",
]
