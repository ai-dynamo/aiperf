# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chaos: API-disruption via toxiproxy.

Covers scenarios C15 and C16 from the chaos-expansion design doc
(``docs/superpowers/specs/2026-04-23-chaos-expansion-design.md``).

Both scenarios require toxiproxy to front a live connection between
the operator and either the apiserver (C15) or the SystemController
HTTP API (C16).

* **Apiserver (C15):** ``kubernetes_asyncio`` auto-configures from the
  in-cluster env vars (``KUBERNETES_SERVICE_HOST`` /
  ``KUBERNETES_SERVICE_PORT``) which the kubelet injects into every
  pod. Redirecting that traffic through toxiproxy requires
  re-deploying the operator Deployment with those env vars pointed at
  the toxiproxy Service. That is a cross-cutting infra change this
  test file deliberately avoids. C15 still ships as
  ``xfail(strict=False)`` with reproduction steps in the docstring.

* **SystemController HTTP (C16):** landed. The operator honors
  ``AIPERF_K8S_CONTROLLER_HTTP_URL_OVERRIDE`` (chaos-only env var)
  which swaps the per-CR JobSet pod DNS for a fixed URL in every
  :class:`ProgressClient` call. The
  ``operator_ready_controller_http_override`` fixture below redeploys
  the operator once with the override pointed at toxiproxy and
  restores a plain operator on teardown.

All tests force-delete their CR in ``finally`` and call
``toxiproxy_injector.reset()`` to clear proxies/toxics across tests.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from tests.kubernetes.chaos.chaos_injector import ChaosInjector
from tests.kubernetes.chaos.toxiproxy import (
    TOXIPROXY_NAMESPACE,
    TOXIPROXY_SERVICE,
    ToxiproxyInjector,
)
from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.operator import AIPerfJobConfig, OperatorDeployer

pytestmark = [pytest.mark.asyncio, pytest.mark.k8s_slow]


# Toxiproxy listen port reserved for the operator -> controller HTTP proxy
# (generic slot from fixtures/toxiproxy.yaml). Kept separate from the
# mock-server slot (20010) so fault-injection targets do not collide.
_CONTROLLER_PROXY_LISTEN_PORT = 20002
_CONTROLLER_PROXY_NAME = "controller"

# URL the operator uses via AIPERF_K8S_CONTROLLER_HTTP_URL_OVERRIDE. The
# toxiproxy Service exposes every reserved listen port as a ClusterIP.
_CONTROLLER_OVERRIDE_URL = (
    f"http://{TOXIPROXY_SERVICE}.{TOXIPROXY_NAMESPACE}.svc.cluster.local:"
    f"{_CONTROLLER_PROXY_LISTEN_PORT}"
)


async def _force_delete(kubectl: KubectlClient, namespace: str, name: str) -> None:
    """Best-effort CR delete; used as the unconditional finally-path."""
    await kubectl.run(
        "delete",
        "aiperfjob",
        name,
        "-n",
        namespace,
        "--ignore-not-found",
        "--wait=false",
        check=False,
    )


@pytest_asyncio.fixture(scope="module", loop_scope="package")
async def operator_ready_controller_http_override(
    kubectl: KubectlClient,
    project_root: Path,
    loaded_images,  # noqa: ANN001 - session-scoped helper, not typed in test surface
    jobset_controller: None,
    mock_server: None,
    k8s_settings,  # noqa: ANN001 - test-fixture dataclass
    operator_job_namespace: str,
) -> AsyncIterator[OperatorDeployer]:
    """Redeploy the operator with ``controller_http_url_override`` set.

    The default ``operator_ready`` fixture leaves
    ``AIPERF_K8S_CONTROLLER_HTTP_URL_OVERRIDE`` unset, so every CR
    funnels its controller HTTP calls directly to per-CR JobSet pod
    DNS. For C16 we need the operator's traffic to traverse toxiproxy
    so a toxic can blackhole it. This fixture redeploys the operator
    once with the override pointed at the chaos-namespace toxiproxy
    Service, then restores a plain operator at teardown so sibling
    package tests run with production-shaped routing.
    """
    deployer = OperatorDeployer(
        kubectl=kubectl,
        project_root=project_root,
        operator_image=k8s_settings.aiperf_image,
        default_job_namespace=operator_job_namespace,
        controller_http_url_override=_CONTROLLER_OVERRIDE_URL,
    )
    await deployer.install_crd()
    await kubectl.run("create", "namespace", operator_job_namespace, check=False)
    await deployer.deploy_operator()
    try:
        yield deployer
    finally:
        if not k8s_settings.skip_cleanup:
            await deployer.cleanup_all()
            restore = OperatorDeployer(
                kubectl=kubectl,
                project_root=project_root,
                operator_image=k8s_settings.aiperf_image,
                default_job_namespace=operator_job_namespace,
                controller_http_url_override=None,
            )
            await restore.deploy_operator()


@pytest.mark.xfail(
    strict=False,
    reason=(
        "C15 requires redirecting the operator's apiserver traffic "
        "through toxiproxy, which needs the operator Deployment "
        "re-rendered with KUBERNETES_SERVICE_HOST / "
        "KUBERNETES_SERVICE_PORT pointed at the toxiproxy Service. "
        "That is a cross-cutting infra change we do not want to make "
        "inline — shipping as xfail(strict=False) with the weaker "
        "controller-HTTP latency variant documented in the body. "
        "Flips to pass once operator chart adds an apiserver-URL "
        "override env (tracked as TODO in findings-2026-04-23-v2.md)."
    ),
)
async def test_c15_pause_apiserver_30s_recovers(
    operator_ready: OperatorDeployer,
    chaos_injector: ChaosInjector,
    toxiproxy_injector: ToxiproxyInjector,
    operator_job_namespace: str,
    kubectl: KubectlClient,
    k8s_settings,  # noqa: ANN001
) -> None:
    """30 s apiserver pause: operator reconcile retries, then CR Completes.

    Exercises kopf's built-in retry behavior when ``list_cluster_custom_object``
    and ``patch_namespaced_custom_object`` raise ``aiohttp.ClientError`` /
    ``asyncio.TimeoutError``. With a 30 s toxiproxy ``timeout`` toxic on
    the apiserver-facing proxy, every reconcile call in that window must
    be retried once the toxic is removed, and the monitor timer must
    resume without dropping the in-flight CR.

    Repro steps for the landed variant (post infra):

    1. ``await toxiproxy_injector.add_proxy("apiserver", "0.0.0.0:20000",
       "kubernetes.default.svc:443")``
    2. Operator env rewritten to ``KUBERNETES_SERVICE_HOST=toxiproxy.
       aiperf-chaos-toxiproxy.svc`` + ``KUBERNETES_SERVICE_PORT=20000``
       (via ``helpers/operator.py::OperatorDeployer.configure_env``).
    3. ``await toxiproxy_injector.add_toxic("apiserver", "timeout",
       {"timeout": 0})`` right after CR reaches Running/profiling.
    4. ``await asyncio.sleep(30); await toxiproxy_injector.remove_toxic(
       "apiserver", "timeout_downstream")``
    5. Assert ``phase=Completed`` within benchmark duration + 120 s.
    """
    name = "chaos-c15"
    longrun_config = AIPerfJobConfig(
        concurrency=2,
        request_count=None,
        benchmark_duration=120.0,
        warmup_request_count=5,
        image=k8s_settings.aiperf_image,
    )
    try:
        # Create the proxy so the fixture is at least exercised end-to-end
        # — this keeps the test meaningful even under xfail by asserting
        # the toxiproxy fixture itself is healthy. The proxy is not
        # actually in the operator's data path until the infra hook lands.
        await toxiproxy_injector.add_proxy(
            name="apiserver",
            listen="0.0.0.0:20000",
            upstream="kubernetes.default.svc:443",
        )

        await operator_ready.create_job(
            config=longrun_config, name=name, namespace=operator_job_namespace
        )
        await chaos_injector.wait_for_phase(
            operator_job_namespace,
            name,
            phases=("Running",),
            current_phase="profiling",
            timeout=180.0,
        )

        # Without operator-side apiserver URL redirection this toxic does
        # NOT actually pause any reconcile traffic — we install it for
        # protocol-parity with the intended scenario so the test body is
        # shippable code rather than a stub.
        await toxiproxy_injector.add_toxic(
            "apiserver",
            "timeout",
            {"timeout": 0},
        )
        await asyncio.sleep(30.0)
        await toxiproxy_injector.remove_toxic("apiserver", "timeout_downstream")

        # Would-be assertion once infra lands. Marked xfail above so a
        # hang here becomes an xfail rather than a hard failure.
        phase = await chaos_injector.wait_for_phase(
            operator_job_namespace,
            name,
            phases=("Completed",),
            timeout=240.0,
        )
        assert phase == "Completed", (
            f"C15: CR should resume to Completed after apiserver pause "
            f"lifts (observed phase={phase!r})"
        )
    finally:
        await _force_delete(kubectl, operator_job_namespace, name)
        await toxiproxy_injector.reset()


@pytest.mark.timeout(600)
async def test_c16_block_operator_controller_http_falls_back(
    operator_ready_controller_http_override: OperatorDeployer,
    chaos_injector: ChaosInjector,
    toxiproxy_injector: ToxiproxyInjector,
    operator_job_namespace: str,
    kubectl: KubectlClient,
    k8s_settings,  # noqa: ANN001
) -> None:
    """Block operator->controller HTTP; salvage path still Completes the CR.

    Exercises ``src/aiperf/operator/handlers/monitor.py::
    _maybe_recover_terminated_controller`` — once every ``_fetch_progress``
    call hangs on a ``timeout`` toxic, the operator can no longer observe
    controller-side progress, but the salvage path polls JobSet pod status
    and fires when the control-plane container exits on its own (which
    happens once the benchmark_duration timer elapses inside the
    controller). The salvage path then drives the CR to Completed without
    relying on controller HTTP.

    Flow:

    1. ``operator_ready_controller_http_override`` redeploys the operator
       with ``AIPERF_K8S_CONTROLLER_HTTP_URL_OVERRIDE`` pointing at
       ``toxiproxy.aiperf-chaos-toxiproxy.svc:20002``.
    2. We create the toxiproxy proxy first with a placeholder upstream —
       the controller pod's IP is not known until the JobSet spawns. The
       operator's initial progress calls will simply fail-and-retry
       against the unroutable placeholder until step 4 fixes it.
    3. Create the CR; once the controller pod has an IP, swap the proxy's
       upstream to the real ``<pod-ip>:19090`` so the operator can
       observe progress and drive the CR to Running/profiling.
    4. Add a ``timeout`` toxic. Operator HTTP now blackholes; salvage
       path must carry the CR to Completed.
    """
    name = "chaos-c16"
    longrun_config = AIPerfJobConfig(
        concurrency=2,
        request_count=None,
        benchmark_duration=120.0,
        warmup_request_count=5,
        image=k8s_settings.aiperf_image,
    )
    try:
        # Placeholder upstream: the real controller pod IP is not known
        # until the JobSet spawns. This keeps the proxy definition stable
        # while the operator retries against an unroutable peer.
        await toxiproxy_injector.add_proxy(
            name=_CONTROLLER_PROXY_NAME,
            listen=f"0.0.0.0:{_CONTROLLER_PROXY_LISTEN_PORT}",
            upstream="127.0.0.1:1",
        )

        await operator_ready_controller_http_override.create_job(
            config=longrun_config, name=name, namespace=operator_job_namespace
        )

        controller_pod = await chaos_injector.get_controller_pod_name(
            operator_job_namespace, name
        )
        # Wait for the pod to get an IP (it takes a few seconds after
        # scheduling). Poll kubectl rather than sleeping blindly.
        pod_ip = ""
        deadline = asyncio.get_event_loop().time() + 120.0
        while asyncio.get_event_loop().time() < deadline:
            res = await kubectl.run(
                "get",
                "pod",
                controller_pod,
                "-n",
                operator_job_namespace,
                "-o",
                "jsonpath={.status.podIP}",
                check=False,
            )
            pod_ip = res.stdout.strip()
            if pod_ip:
                break
            await asyncio.sleep(2.0)

        assert pod_ip, (
            f"C16: controller pod {operator_job_namespace}/{controller_pod} "
            "never received an IP within 120s; cannot rewrite toxiproxy proxy."
        )

        # Swap the proxy upstream to the real controller. Toxiproxy does
        # not support in-place upstream updates, so delete+add.
        await toxiproxy_injector.remove_proxy(_CONTROLLER_PROXY_NAME)
        await toxiproxy_injector.add_proxy(
            name=_CONTROLLER_PROXY_NAME,
            listen=f"0.0.0.0:{_CONTROLLER_PROXY_LISTEN_PORT}",
            upstream=f"{pod_ip}:19090",
        )

        # The operator can now observe controller progress. Wait until
        # the CR reaches Running/profiling before we blackhole the link.
        await chaos_injector.wait_for_phase(
            operator_job_namespace,
            name,
            phases=("Running",),
            current_phase="profiling",
            timeout=180.0,
        )

        # Blackhole every subsequent controller HTTP call. The salvage
        # path (_maybe_recover_terminated_controller) must drive the CR
        # to Completed once the controller pod exits on its own at the
        # benchmark_duration boundary.
        await toxiproxy_injector.add_toxic(
            _CONTROLLER_PROXY_NAME,
            "timeout",
            {"timeout": 30000},
        )

        phase = await chaos_injector.wait_for_phase(
            operator_job_namespace,
            name,
            phases=("Completed",),
            timeout=300.0,
        )
        assert phase == "Completed", (
            f"C16: CR should reach Completed via salvage path "
            f"(observed phase={phase!r})"
        )
    finally:
        await _force_delete(kubectl, operator_job_namespace, name)
        await toxiproxy_injector.reset()
