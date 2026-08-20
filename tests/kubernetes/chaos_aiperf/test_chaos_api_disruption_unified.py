# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified-chaos port of ``tests/kubernetes/chaos/test_chaos_api_disruption.py``.

Covers chaos-expansion scenarios C15 and C16:

* **C15** -- 30s apiserver timeout via Toxiproxy on the operator's
  apiserver-routed Service; operator survives the pause and the CR
  Completes once the toxic lifts.
* **C16** -- operator -> SystemController HTTP blackhole via Toxiproxy
  on the controller's HTTP API; the sidecar-results recovery path
  drives the CR to Completed without further controller progress.

This file mirrors the legacy module case-for-case; only the
fault-injection surface differs:

* Legacy: ``await toxiproxy_injector.add_toxic("apiserver", "timeout",
  {"timeout": 0})``
* Unified: ``async with faults.inject("network.timeout",
  target={"proxy": "apiserver"}, attributes={"timeout": 0}): ...``

The :py:data:`faults` registry (declared in this package's ``conftest.py``)
wraps :py:class:`tests.kubernetes.chaos_common.injectors.network.NetworkInjector`
around the legacy ``toxiproxy_injector``; the unified ``async with`` block
adds the toxic on entry and removes it on exit, replacing the manual
``add_toxic`` + ``remove_toxic`` pair the legacy file used.

The heavyweight operator-redeploy fixtures
(``operator_ready_apiserver_toxiproxy_routed``,
``operator_ready_toxiproxy_routed``) are reused unchanged via the
``pytest_plugins`` declaration in this package's ``conftest.py``.
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
from tests.kubernetes.chaos_common.registry import InjectorRegistry
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


async def test_c15_pause_apiserver_30s_recovers_unified(
    operator_ready_apiserver_toxiproxy_routed: OperatorDeployer,
    chaos_injector: ChaosInjector,
    toxiproxy_injector: ToxiproxyInjector,
    faults: InjectorRegistry,
    operator_job_namespace: str,
    kubectl: KubectlClient,
    k8s_settings,  # noqa: ANN001 - test-fixture dataclass
) -> None:
    """30s apiserver pause via the unified registry: operator survives, CR Completes.

    Mirrors the legacy ``test_c15_pause_apiserver_30s_recovers``; replaces
    the explicit ``add_toxic`` / ``remove_toxic`` pair with a single
    ``async with faults.inject("network.timeout", ...): await asyncio.sleep(30)``
    block. The ``"apiserver"`` Toxiproxy proxy is already created by the
    ``operator_ready_apiserver_toxiproxy_routed`` fixture before the
    operator is deployed (see ``tests/kubernetes/chaos/conftest.py``),
    so the test body only needs to add and remove the timeout toxic.
    """
    name = "chaos-c15-unified"
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

        async with faults.inject(
            "network.timeout",
            target={"proxy": "apiserver"},
            attributes={"timeout": 0},
        ):
            await asyncio.sleep(30.0)

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
async def test_c16_block_operator_controller_http_falls_back_unified(
    operator_ready_toxiproxy_routed: OperatorDeployer,
    chaos_injector: ChaosInjector,
    toxiproxy_injector: ToxiproxyInjector,
    faults: InjectorRegistry,
    operator_job_namespace: str,
    kubectl: KubectlClient,
    k8s_settings,  # noqa: ANN001 - test-fixture dataclass
) -> None:
    """Block operator -> controller HTTP via the unified registry; sidecar recovery Completes the CR.

    Mirrors the legacy ``test_c16_block_operator_controller_http_falls_back``.
    The controller-fronting Toxiproxy proxy is created in the test body
    (its upstream is the per-CR JobSet DNS name, which depends on the CR
    name and namespace, so it cannot be hoisted into the fixture). Once
    the proxy exists, the unified ``faults.inject("network.timeout", ...)``
    block adds a 30s timeout toxic that blackholes every subsequent
    ``_fetch_progress`` call; the operator must recover exported results
    through the sidecar and mark the CR Completed.
    """
    name = "chaos-c16-unified"
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
        # recovery path must drive the CR to Completed. The unified
        # ``async with`` keeps the toxic active until the CR reaches the
        # terminal phase, then removes it on exit.
        async with faults.inject(
            "network.timeout",
            target={"proxy": _CONTROLLER_PROXY_NAME},
            attributes={"timeout": 30000},
        ):
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
