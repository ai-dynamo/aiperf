# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chaos: operator-pod resilience -- unified-API port of C4, C5, C5b.

Covers scenarios C4 (kill operator mid-benchmark; CR still completes) and
C5 / C5b (orphaned ``aiperf.nvidia.com/completion-claimed`` annotation
present at operator restart; CR converges to Completed via the monitor
orphan-recovery branch).

Port of :py:mod:`tests.kubernetes.chaos.test_chaos_operator_resilience`
onto the unified ``faults`` registry (:py:mod:`tests.kubernetes.chaos_common`).
Each scenario is the legacy test verbatim with one substitution:

* ``chaos_injector.kill_operator_pod(force=True)`` becomes
  ``async with faults.inject("operator.kill", target=...)``.
* ``chaos_injector.stamp_completion_claim(ns, name, ts)`` becomes
  ``async with faults.inject("crd.annotate", target=..., annotation_key=...,
  value=ts)``. The restore on context exit is harmless because the test's
  ``finally`` deletes the CR.
* ``chaos_injector.wait_for_phase(...)`` becomes
  ``await wait_for_aiperfjob_phase(kubectl, ...)`` (provided by
  :py:mod:`tests.kubernetes.chaos_aiperf.conftest`).
* ``chaos_injector.read_claim_annotation(...)`` stays as a direct call on a
  locally constructed :py:class:`ChaosInjector` -- reads are not fault
  injections and the helper is read-only.
"""

from __future__ import annotations

import datetime

import pytest

from tests.kubernetes.chaos.chaos_injector import (
    AIPERF_CLAIM_ANNOTATION,
    OPERATOR_NAMESPACE,
    OPERATOR_SELECTOR,
    ChaosInjector,
)
from tests.kubernetes.chaos_aiperf.conftest import wait_for_aiperfjob_phase
from tests.kubernetes.chaos_common.registry import InjectorRegistry
from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.operator import AIPerfJobConfig, OperatorDeployer

pytestmark = [pytest.mark.asyncio, pytest.mark.k8s_slow]


@pytest.fixture
def longrun_config(k8s_settings) -> AIPerfJobConfig:
    """Duration-based benchmark so chaos can land mid-profiling.

    Matches the legacy fixture in
    :py:mod:`tests.kubernetes.chaos.test_chaos_operator_resilience` --
    keeping the same shape means scenario timing carries over without
    re-tuning.
    """
    return AIPerfJobConfig(
        concurrency=3,
        request_count=None,
        benchmark_duration=120.0,
        warmup_request_count=5,
        image=k8s_settings.aiperf_image,
    )


async def _wait_for_operator_ready(
    kubectl: KubectlClient, timeout: str = "60s"
) -> None:
    """Wait for the operator Deployment to be Available again.

    Replaces ``chaos_injector.wait_for_operator_ready(...)``. Using
    ``kubectl wait`` here avoids pulling in another legacy helper just to
    poll readiness; the unified registry intentionally leaves "wait for
    pod ready" to the caller so tests stay the source
    of truth for what "recovered" means.
    """
    await kubectl.run(
        "wait",
        f"deployment/{OPERATOR_SELECTOR.split('=')[1]}",
        "-n",
        OPERATOR_NAMESPACE,
        "--for=condition=Available",
        f"--timeout={timeout}",
        check=True,
    )


async def test_c4_kill_operator_mid_benchmark_recovers_unified(
    operator_ready: OperatorDeployer,
    faults: InjectorRegistry,
    longrun_config: AIPerfJobConfig,
    operator_job_namespace: str,
    kubectl: KubectlClient,
) -> None:
    """Force-delete operator pod mid-profiling; benchmark reaches Completed.

    Verifies kopf reconcile-resume + durable completion claim
    (``aiperf.nvidia.com/completion-claimed`` annotation) guarantee
    exactly-once completion across operator restarts.
    """
    name = "chaos-c4-unified"
    try:
        await operator_ready.create_job(
            config=longrun_config, name=name, namespace=operator_job_namespace
        )
        await wait_for_aiperfjob_phase(
            kubectl,
            operator_job_namespace,
            name,
            phases=("Running",),
            current_phase="profiling",
            timeout=180.0,
        )

        async with faults.inject(
            "operator.kill",
            target={"selector": OPERATOR_SELECTOR, "ns": OPERATOR_NAMESPACE},
        ):
            # ``operator.kill`` restore is a no-op; ReplicaSet recreates the
            # Pod and the test owns "wait for ready" below.
            pass
        await _wait_for_operator_ready(kubectl)

        # Benchmark duration is 120 s; give generous margin for the
        # post-completion housekeeping (JobSet delete, pod terminate).
        phase = await wait_for_aiperfjob_phase(
            kubectl,
            operator_job_namespace,
            name,
            phases=("Completed",),
            timeout=240.0,
        )
        assert phase == "Completed"

        # Completion claim annotation must be present on the terminal CR.
        # Direct legacy call -- porting the read path is out of scope for
        # this wave; see module docstring.
        legacy_injector = ChaosInjector(kubectl=kubectl)
        claim = await legacy_injector.read_claim_annotation(
            operator_job_namespace, name
        )
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


async def test_c5_orphaned_claim_recovers_unified(
    operator_ready: OperatorDeployer,
    faults: InjectorRegistry,
    longrun_config: AIPerfJobConfig,
    operator_job_namespace: str,
    kubectl: KubectlClient,
) -> None:
    """Pre-stamp completion-claim annotation + kill operator; CR still completes.

    Simulates the window where the operator crashed after the claim patch
    but before ``handle_completion`` finished. The ``_monitor_tick``
    orphaned-claim recovery branch (see ``monitor.py::
    _recover_orphaned_completion_claim``) re-invokes ``handle_completion``
    so the CR converges to Completed -- without this, the CR would be stuck
    Running forever because every completion code path short-circuits on
    the annotation.
    """
    name = "chaos-c5-unified"
    try:
        await operator_ready.create_job(
            config=longrun_config, name=name, namespace=operator_job_namespace
        )
        await wait_for_aiperfjob_phase(
            kubectl,
            operator_job_namespace,
            name,
            phases=("Running",),
            current_phase="profiling",
            timeout=180.0,
        )

        # Match the legacy helper's timestamp format so any operator-side
        # parsing keeps working byte-for-byte.
        ts = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.000000Z")
        async with faults.inject(
            "crd.annotate",
            target={"ns": operator_job_namespace, "name": name},
            annotation_key=AIPERF_CLAIM_ANNOTATION,
            value=ts,
        ):
            async with faults.inject(
                "operator.kill",
                target={"selector": OPERATOR_SELECTOR, "ns": OPERATOR_NAMESPACE},
            ):
                pass
            await _wait_for_operator_ready(kubectl)

            # Benchmark duration 120 s + orphan-recovery margin. Phase poll
            # happens INSIDE the ``crd.annotate`` block so the annotation
            # stays in place while the operator decides what to do; the
            # restore (annotation strip) runs only after we've observed the
            # terminal phase.
            phase = await wait_for_aiperfjob_phase(
                kubectl,
                operator_job_namespace,
                name,
                phases=("Completed", "Failed"),
                timeout=300.0,
            )
        assert phase == "Completed", (
            f"Orphan-claim recovery should converge CR to Completed, got {phase}"
        )
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
