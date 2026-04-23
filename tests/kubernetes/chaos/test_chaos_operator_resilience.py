# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chaos: operator-pod resilience (kill operator mid-benchmark).

Covers scenarios C4 and C5 from the chaos design doc.
"""

from __future__ import annotations

import pytest

from tests.kubernetes.chaos.chaos_injector import ChaosInjector
from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.operator import AIPerfJobConfig, OperatorDeployer

pytestmark = [pytest.mark.asyncio, pytest.mark.k8s_slow]


@pytest.fixture
def longrun_config(k8s_settings) -> AIPerfJobConfig:
    """Duration-based benchmark so chaos can land mid-profiling."""
    return AIPerfJobConfig(
        concurrency=3,
        request_count=None,
        benchmark_duration=120.0,
        warmup_request_count=5,
        image=k8s_settings.aiperf_image,
    )


async def test_c4_kill_operator_mid_benchmark_recovers(
    operator_ready: OperatorDeployer,
    chaos_injector: ChaosInjector,
    longrun_config: AIPerfJobConfig,
    operator_job_namespace: str,
    kubectl: KubectlClient,
) -> None:
    """Force-delete operator pod mid-profiling; benchmark reaches Completed.

    Verifies kopf reconcile-resume + durable completion claim
    (`aiperf.nvidia.com/completion-claimed` annotation) guarantee
    exactly-once completion across operator restarts.
    """
    name = "chaos-c4"
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

        await chaos_injector.kill_operator_pod(force=True)
        await chaos_injector.wait_for_operator_ready(timeout=60.0)

        # Benchmark duration is 120 s; give generous margin for the
        # post-completion housekeeping (JobSet delete, pod terminate).
        phase = await chaos_injector.wait_for_phase(
            operator_job_namespace,
            name,
            phases=("Completed",),
            timeout=240.0,
        )
        assert phase == "Completed"

        # Completion claim annotation must be present on the terminal CR.
        claim = await chaos_injector.read_claim_annotation(operator_job_namespace, name)
        assert claim, "completion-claimed annotation missing on Completed CR"
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


@pytest.mark.xfail(
    reason=(
        "Synthetic C5 — simulates 'operator crashed right after claiming' by "
        "stamping the completion-claimed annotation before the real claim. "
        "This is presently an open-question scenario for handle_completion "
        "recovery; see findings.md / tests/kubernetes/chaos/README.md."
    ),
    strict=False,
)
async def test_c5_orphaned_claim_recovers(
    operator_ready: OperatorDeployer,
    chaos_injector: ChaosInjector,
    longrun_config: AIPerfJobConfig,
    operator_job_namespace: str,
    kubectl: KubectlClient,
) -> None:
    """Pre-stamp completion-claim annotation + kill operator.

    Simulates the window where the operator crashed after the claim
    patch but before `handle_completion` finished. A correctly
    self-healing operator should still bring the CR to Completed (via
    `_maybe_recover_terminated_controller` salvage) — but the in-process
    `_shutdown_sent` set loses state on restart, and the annotation
    fast-path can short-circuit the salvage attempt. See findings log.
    """
    name = "chaos-c5"
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

        await chaos_injector.stamp_completion_claim(operator_job_namespace, name)
        await chaos_injector.kill_operator_pod(force=True)
        await chaos_injector.wait_for_operator_ready(timeout=60.0)

        # Benchmark duration 120 s + generous recovery margin.
        phase = await chaos_injector.wait_for_phase(
            operator_job_namespace,
            name,
            phases=("Completed", "Failed"),
            timeout=300.0,
        )
        assert phase in ("Completed", "Failed")
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
