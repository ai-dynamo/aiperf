# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chaos: API-disruption via toxiproxy.

Covers chaos-expansion scenarios C15 and C16.

Both scenarios require toxiproxy to front a live connection between
the operator and either the apiserver (C15) or the SystemController
HTTP API (C16).

* **Apiserver (C15):** ``kubernetes_asyncio`` auto-configures from the
  in-cluster env vars (``KUBERNETES_SERVICE_HOST`` /
  ``KUBERNETES_SERVICE_PORT``) which the kubelet injects into every
  pod. The C15 fixture re-deploys the operator with those env vars
  pointed at the toxiproxy Service and sets
  ``AIPERF_K8S_APISERVER_TLS_SERVER_NAME_OVERRIDE=kubernetes.default.svc``
  so raw TCP passthrough still verifies the real apiserver certificate.
  The apiserver proxy is created before the operator deploys so kopf login
  and watch setup traverse toxiproxy from process start.

* **SystemController HTTP (C16):** landed. The operator honors
  ``AIPERF_K8S_CONTROLLER_HTTP_URL_OVERRIDE`` (chaos-only env var)
  which swaps the per-CR JobSet pod DNS for a fixed URL in controller
  API-port :class:`ProgressClient` calls. The shared package-scoped fixture
  ``operator_ready_toxiproxy_routed`` (see
  ``tests/kubernetes/chaos/conftest.py``) redeploys the operator once
  with the override pointed at toxiproxy and restores a plain operator
  on teardown. New controller-HTTP fault-injection scenarios should
  reuse that fixture rather than rolling their own.

All tests force-delete their CR in ``finally`` and call
``toxiproxy_injector.reset()`` to clear proxies/toxics across tests.
"""

from __future__ import annotations

import asyncio

import pytest

from aiperf.kubernetes.jobset import controller_dns_name
from tests.kubernetes.chaos.chaos_injector import ChaosInjector
from tests.kubernetes.chaos.conftest import CONTROLLER_HTTP_UPSTREAM_PORT
from tests.kubernetes.chaos.toxiproxy import (
    TOXIPROXY_CONTROLLER_HTTP_PORT,
    ToxiproxyInjector,
)
from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.operator import AIPerfJobConfig, OperatorDeployer

pytestmark = [pytest.mark.asyncio, pytest.mark.k8s_slow]


_CONTROLLER_PROXY_NAME = "controller"


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


async def test_c15_pause_apiserver_30s_recovers(
    operator_ready_apiserver_toxiproxy_routed: OperatorDeployer,
    chaos_injector: ChaosInjector,
    toxiproxy_injector: ToxiproxyInjector,
    operator_job_namespace: str,
    kubectl: KubectlClient,
    k8s_settings,  # noqa: ANN001
) -> None:
    """30 s apiserver pause: operator reconcile retries, then CR Completes.

    Exercises kopf's built-in retry behavior when ``list_cluster_custom_object``
    and ``patch_namespaced_custom_object`` raise ``aiohttp.ClientError`` /
    ``TimeoutError``. With a 30 s toxiproxy ``timeout`` toxic on
    the apiserver-facing proxy, every reconcile call in that window must
    be retried once the toxic is removed, and the monitor timer must
    resume without dropping the in-flight CR.

    Repro steps for the routed variant:

    1. ``await toxiproxy_injector.add_proxy("apiserver", "0.0.0.0:20000",
       "kubernetes.default.svc:443")``
    2. Operator env rewritten to ``KUBERNETES_SERVICE_HOST=toxiproxy.
       aiperf-chaos-toxiproxy.svc`` + ``KUBERNETES_SERVICE_PORT=20000`` +
       ``AIPERF_K8S_APISERVER_TLS_SERVER_NAME_OVERRIDE=kubernetes.default.svc``
       (via ``operator_ready_apiserver_toxiproxy_routed``).
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
        await operator_ready_apiserver_toxiproxy_routed.create_job(
            config=longrun_config, name=name, namespace=operator_job_namespace
        )
        await chaos_injector.wait_for_phase(
            operator_job_namespace,
            name,
            phases=("Running",),
            current_phase="profiling",
            timeout=180.0,
        )

        await toxiproxy_injector.add_toxic(
            "apiserver",
            "timeout",
            {"timeout": 0},
        )
        await asyncio.sleep(30.0)
        await toxiproxy_injector.remove_toxic("apiserver", "timeout_downstream")

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
    operator_ready_toxiproxy_routed: OperatorDeployer,
    chaos_injector: ChaosInjector,
    toxiproxy_injector: ToxiproxyInjector,
    operator_job_namespace: str,
    kubectl: KubectlClient,
    k8s_settings,  # noqa: ANN001
) -> None:
    """Block operator->controller HTTP; sidecar-result recovery Completes the CR.

    The controller pod DNS name is deterministic before the JobSet exists, so
    toxiproxy can route to the future controller Service from the start. Once
    every ``_fetch_progress`` call hangs on a ``timeout`` toxic, the operator
    can no longer observe controller-side progress directly; it must recover
    exported results through the results sidecar and mark the CR Completed.
    """
    name = "chaos-c16"
    longrun_config = AIPerfJobConfig(
        concurrency=2,
        request_count=None,
        benchmark_duration=120.0,
        warmup_request_count=5,
        image=k8s_settings.aiperf_image,
    )

    # Sanity-check: operator was deployed through the shared fixture, so
    # the env var must be present on the live pod. Catching a silent
    # fixture-ordering regression here is cheaper than diagnosing it from
    # a 10-minute timeout downstream.
    env_check = await kubectl.run(
        "set",
        "env",
        "deployment/aiperf-operator",
        "--list",
        "-n",
        OperatorDeployer.OPERATOR_NAMESPACE,
        check=True,
    )
    assert "AIPERF_K8S_CONTROLLER_HTTP_URL_OVERRIDE" in env_check.stdout, (
        "C16 precondition failed: operator is not routed through toxiproxy "
        "(AIPERF_K8S_CONTROLLER_HTTP_URL_OVERRIDE missing from deployment env); "
        "check that operator_ready_toxiproxy_routed is the fixture in use."
    )

    try:
        await toxiproxy_injector.add_proxy(
            name=_CONTROLLER_PROXY_NAME,
            listen=f"0.0.0.0:{TOXIPROXY_CONTROLLER_HTTP_PORT}",
            upstream=(
                f"{controller_dns_name(f'aiperf-{name}', operator_job_namespace)}:"
                f"{CONTROLLER_HTTP_UPSTREAM_PORT}"
            ),
        )

        await operator_ready_toxiproxy_routed.create_job(
            config=longrun_config, name=name, namespace=operator_job_namespace
        )

        # The operator can observe controller progress through toxiproxy. Wait until
        # the CR reaches Running/profiling before we blackhole the link.
        await chaos_injector.wait_for_phase(
            operator_job_namespace,
            name,
            phases=("Running",),
            current_phase="profiling",
            timeout=180.0,
        )

        # Blackhole every subsequent controller HTTP call. The sidecar-result
        # recovery path must drive the CR to Completed.
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
