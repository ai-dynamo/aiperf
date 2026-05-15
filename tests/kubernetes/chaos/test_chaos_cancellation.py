# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chaos: cancellation path (delete CR mid-benchmark).

Covers scenarios C1 and C3 from the chaos design doc.
"""

from __future__ import annotations

import pytest

from tests.kubernetes.chaos.chaos_injector import ChaosInjector
from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.operator import AIPerfJobConfig, OperatorDeployer

pytestmark = [pytest.mark.asyncio, pytest.mark.k8s_slow]


@pytest.fixture
def longrun_config(k8s_settings) -> AIPerfJobConfig:
    """Duration-based benchmark so delete can land mid-profiling."""
    return AIPerfJobConfig(
        concurrency=3,
        request_count=None,
        benchmark_duration=120.0,
        warmup_request_count=5,
        image=k8s_settings.aiperf_image,
    )


async def test_c1_delete_aiperfjob_mid_ramp(
    operator_ready: OperatorDeployer,
    chaos_injector: ChaosInjector,
    longrun_config: AIPerfJobConfig,
    operator_job_namespace: str,
    kubectl: KubectlClient,
) -> None:
    """Delete the CR during profiling; finalizer must clear promptly.

    Verifies the cooperative-cancellation path on `src/aiperf/operator/
    handlers/lifecycle.py::on_delete`: `request_cancellation` latches
    the flag, `close_progress_client` releases the aiohttp session, and
    owner-ref GC reaps the JobSet + pods without operator intervention.
    """
    name = "chaos-c1"
    try:
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

        await chaos_injector.delete_cr_no_wait(operator_job_namespace, name)

        # CR should leave the apiserver within a few seconds (finalizer
        # removal is ~100 ms in practice).
        elapsed_cr = await chaos_injector.wait_for_cr_gone(
            operator_job_namespace, name, timeout=10.0
        )
        assert elapsed_cr < 10.0

        # Pods take longer due to SIGTERM grace; 45 s window is generous.
        await chaos_injector.wait_for_pods_gone(
            operator_job_namespace, timeout=chaos_injector.timings.pod_termination_grace
        )
    finally:
        # Belt-and-suspenders cleanup in case the assertion chain raised.
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


async def test_c3_rapid_double_delete_is_idempotent(
    operator_ready: OperatorDeployer,
    chaos_injector: ChaosInjector,
    longrun_config: AIPerfJobConfig,
    operator_job_namespace: str,
    kubectl: KubectlClient,
) -> None:
    """Two rapid deletes: second must NotFound (404); operator must not error.

    Guarantees `request_cancellation` + `close_progress_client` stay
    idempotent. kopf runs `on_delete` at most once per object generation.
    """
    name = "chaos-c3"
    try:
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

        first_rc, second_rc = await chaos_injector.delete_cr_twice(
            operator_job_namespace, name
        )
        assert first_rc == 0, f"First delete rc={first_rc}"
        # kubectl returns non-zero on NotFound; that's the expected
        # outcome for the second call because the first already removed
        # the object.
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
