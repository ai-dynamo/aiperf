# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""C1/C3 unified-API ports - see chaos/test_chaos_cancellation.py for legacy.

Exercises the cooperative-cancellation path
(``src/aiperf/operator/handlers/lifecycle.py::on_delete``) through the
unified ``faults.inject(...)`` interface rather than the legacy
``chaos_injector`` direct calls.

The function-scope :py:data:`faults` registry inherited from
:py:mod:`tests.kubernetes.chaos_aiperf.conftest` ships only a
:py:class:`PodInjector` by default, so each C-series module that needs
CR-level faults (``crd.delete``, ``crd.delete_twice``, ``operator.kill``)
must register its own :py:class:`CRDInjector` parameterized for
AIPerfJob. This module does so via the :py:data:`faults_with_crd`
function-scope fixture; tests use that name instead of ``faults`` to make
the registration explicit at the call site.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from aiperf.common.aiperf_logger import AIPerfLogger
from tests.kubernetes.chaos.chaos_injector import (
    OPERATOR_NAMESPACE,
    OPERATOR_SELECTOR,
    ChaosInjector,
)
from tests.kubernetes.chaos_common.injectors.crd import CRDInjector
from tests.kubernetes.chaos_common.registry import InjectorRegistry
from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.operator import AIPerfJobConfig, OperatorDeployer

pytestmark = [pytest.mark.k8s_slow, pytest.mark.asyncio]
logger = AIPerfLogger(__name__)


@pytest.fixture
def longrun_config(k8s_settings) -> AIPerfJobConfig:
    """Duration-based benchmark so delete can land mid-profiling.

    Mirrors the identically named fixture in
    :py:mod:`tests.kubernetes.chaos.test_chaos_cancellation`; kept local
    because pytest does not share package-private fixtures across
    sibling test packages (``chaos/`` vs. ``chaos_aiperf/``).
    """
    return AIPerfJobConfig(
        concurrency=3,
        request_count=None,
        benchmark_duration=120.0,
        warmup_request_count=5,
        image=k8s_settings.aiperf_image,
    )


@pytest_asyncio.fixture
async def faults_with_crd(
    faults: InjectorRegistry,
    kubectl: KubectlClient,
) -> AsyncIterator[InjectorRegistry]:
    """Augment the inherited :py:data:`faults` registry with an AIPerf-shape CRDInjector.

    The conftest in :py:mod:`tests.kubernetes.chaos_aiperf.conftest`
    only pre-registers :py:class:`PodInjector`. C1/C3 dispatch on the
    ``crd.*`` namespace, so we register a :py:class:`CRDInjector`
    parameterized for ``AIPerfJob`` / ``aiperf.nvidia.com`` and the
    chart-default operator namespace + selector.
    """
    faults.register(
        CRDInjector(
            kubectl,
            cr_kind="aiperfjob",
            cr_api_group="aiperf.nvidia.com",
            operator_namespace=OPERATOR_NAMESPACE,
            operator_selector=OPERATOR_SELECTOR,
        )
    )
    yield faults


@pytest.fixture
def waiter(kubectl: KubectlClient) -> ChaosInjector:
    """Plain-helper :py:class:`ChaosInjector` for ``wait_for_*`` calls.

    The unified API only exposes fault injection (``inject`` /
    ``compose``); polling helpers like ``wait_for_phase``,
    ``wait_for_cr_gone``, and ``wait_for_pods_gone`` remain on the
    legacy :py:class:`ChaosInjector`. The task contract explicitly
    allows reusing those helpers verbatim.
    """
    return ChaosInjector(kubectl)


async def test_c1_delete_aiperfjob_mid_ramp_unified(
    operator_ready: OperatorDeployer,
    faults_with_crd: InjectorRegistry,
    waiter: ChaosInjector,
    longrun_config: AIPerfJobConfig,
    operator_job_namespace: str,
    kubectl: KubectlClient,
) -> None:
    """C1 unified port: delete AIPerfJob mid-ramp via ``faults.inject('crd.delete')``.

    Verifies the cooperative-cancellation path on
    ``src/aiperf/operator/handlers/lifecycle.py::on_delete``:
    ``request_cancellation`` latches the flag,
    ``close_progress_client`` releases the aiohttp session, and
    owner-ref GC reaps the JobSet + pods without operator intervention.
    """
    name = "chaos-c1-unified"
    try:
        await operator_ready.create_job(
            config=longrun_config, name=name, namespace=operator_job_namespace
        )
        await waiter.wait_for_phase(
            operator_job_namespace,
            name,
            phases=("Running",),
            current_phase="profiling",
            timeout=180.0,
        )

        async with faults_with_crd.inject(
            "crd.delete",
            target={"ns": operator_job_namespace, "name": name},
        ):
            # The delete is fire-and-forget; assertions about cleanup
            # latency happen inside the block so the registry teardown
            # has nothing to restore (crd.delete has no restore path).
            elapsed_cr = await waiter.wait_for_cr_gone(
                operator_job_namespace, name, timeout=10.0
            )
            assert elapsed_cr < 10.0

            await waiter.wait_for_pods_gone(
                operator_job_namespace,
                timeout=waiter.timings.pod_termination_grace,
            )
    finally:
        # Belt-and-suspenders cleanup in case the assertion chain raised
        # before the inject block could observe pod teardown.
        await kubectl.run(
            "delete",
            "aiperfjob",
            name,
            "-n",
            operator_job_namespace,
            "--ignore-not-found",
            "--wait=false",
            check=False,
        )


async def test_c3_rapid_double_delete_is_idempotent_unified(
    operator_ready: OperatorDeployer,
    faults_with_crd: InjectorRegistry,
    waiter: ChaosInjector,
    longrun_config: AIPerfJobConfig,
    operator_job_namespace: str,
    kubectl: KubectlClient,
) -> None:
    """C3 unified port: two rapid deletes via ``faults.inject('crd.delete_twice')``.

    The second kubectl delete must NotFound (404); the operator must
    not error. Guarantees ``request_cancellation`` +
    ``close_progress_client`` stay idempotent under the kopf
    "on_delete at most once per object generation" contract.
    """
    name = "chaos-c3-unified"
    try:
        await operator_ready.create_job(
            config=longrun_config, name=name, namespace=operator_job_namespace
        )
        await waiter.wait_for_phase(
            operator_job_namespace,
            name,
            phases=("Running",),
            current_phase="profiling",
            timeout=180.0,
        )

        async with faults_with_crd.inject(
            "crd.delete_twice",
            target={"ns": operator_job_namespace, "name": name},
        ) as applied:
            first_rc = applied.metadata["first_rc"]
            second_rc = applied.metadata["second_rc"]
            assert first_rc == 0, f"First delete rc={first_rc}"
            # kubectl returns non-zero on NotFound; that is the expected
            # outcome for the second call because the first already
            # removed the object.
            assert second_rc != 0, f"Second delete rc={second_rc} (expected NotFound)"
    finally:
        await kubectl.run(
            "delete",
            "aiperfjob",
            name,
            "-n",
            operator_job_namespace,
            "--ignore-not-found",
            "--wait=false",
            check=False,
        )
