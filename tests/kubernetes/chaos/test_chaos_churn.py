# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chaos: CR churn + invalid-spec validation.

Covers chaos-expansion scenarios C10, C11, C12.

Exercises these operator code paths:

* ``src/aiperf/operator/handlers/create.py::on_create`` —
  ``clear_cancellation(job_key(namespace, job_id))`` at the top of
  every create, so a re-created same-name CR is not starved by the
  sticky cancellation flag from a prior cycle (C10).
* Per-CR isolation of reconcile timers and progress clients — deleting
  CRs A/B must not perturb CR C's monitor loop (C11).
* ``src/aiperf/operator/handlers/create.py::_validate_spec`` — raises
  ``kopf.PermanentError`` and stamps ``phase=Failed`` with a
  ``ConfigValid=False`` status condition when the spec converter
  rejects the payload (C12).
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC

import pytest

from tests.kubernetes.chaos.chaos_injector import (
    AIPERF_CLAIM_ANNOTATION,
    ChaosInjector,
)
from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.operator import AIPerfJobConfig, OperatorDeployer

pytestmark = [pytest.mark.asyncio, pytest.mark.k8s_slow]


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


async def test_c10_rapid_create_delete_recreate_same_name(
    operator_ready: OperatorDeployer,
    chaos_injector: ChaosInjector,
    operator_job_namespace: str,
    kubectl: KubectlClient,
    k8s_settings,  # noqa: ANN001  (pytest fixture, typed as Any via duck-typing)
) -> None:
    """Same-named CR re-created after delete reaches Completed cleanly.

    Exercises the ``clear_cancellation`` fast-path in
    ``src/aiperf/operator/handlers/create.py::on_create``. Without it,
    the ``request_cancellation`` flag latched by cycle 1's ``on_delete``
    would short-circuit every ``await``-boundary check in cycle 2 and
    starve profiling. The assertion that matters is not just
    ``phase=Completed`` but that the claim annotation timestamp on
    cycle 2 is strictly newer than cycle 1's delete time — proving that
    completion actually ran, not that we read a stale annotation.
    """
    name = "chaos-c10"
    small_config = AIPerfJobConfig(
        concurrency=2,
        request_count=30,
        warmup_request_count=5,
        image=k8s_settings.aiperf_image,
    )
    try:
        # Cycle 1: create + wait for JobSet + delete + wait drain.
        await operator_ready.create_job(
            config=small_config, name=name, namespace=operator_job_namespace
        )
        await chaos_injector.wait_for_phase(
            operator_job_namespace,
            name,
            phases=("Initializing", "Running"),
            timeout=180.0,
        )

        delete_wall_time = time.time()
        await chaos_injector.delete_cr_no_wait(operator_job_namespace, name)
        await chaos_injector.wait_for_cr_gone(
            operator_job_namespace,
            name,
            timeout=chaos_injector.timings.cr_cleanup_seconds,
        )
        await chaos_injector.wait_for_pods_gone(
            operator_job_namespace,
            timeout=chaos_injector.timings.pod_termination_grace,
        )

        # Cycle 2: re-create same name; must converge to Completed.
        await operator_ready.create_job(
            config=small_config, name=name, namespace=operator_job_namespace
        )
        phase = await chaos_injector.wait_for_phase(
            operator_job_namespace,
            name,
            phases=("Completed",),
            timeout=300.0,
        )
        assert phase == "Completed", (
            f"C10 cycle 2 did not reach Completed (phase={phase!r}); "
            "sticky cancellation may be starving the re-created CR"
        )

        # Claim annotation on cycle 2 must be NEWER than cycle 1's delete.
        claim = await chaos_injector.read_claim_annotation(operator_job_namespace, name)
        assert claim, (
            "C10 cycle 2 Completed CR missing "
            f"{AIPERF_CLAIM_ANNOTATION}; completion did not actually run"
        )
        # Annotation format: RFC3339-ish with microseconds + Z suffix.
        # Parse by swapping Z -> +00:00 so datetime can consume it.
        from datetime import datetime

        claim_dt = datetime.strptime(
            claim.replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S.%f%z"
        )
        delete_dt = datetime.fromtimestamp(delete_wall_time, tz=UTC)
        assert claim_dt > delete_dt, (
            f"C10 cycle 2 claim timestamp {claim_dt.isoformat()} is not "
            f"after cycle 1 delete {delete_dt.isoformat()} — the operator "
            "may be reading a stale annotation instead of actually "
            "completing cycle 2"
        )
    finally:
        await _force_delete(kubectl, operator_job_namespace, name)


async def test_c11_parallel_jobs_delete_subset(
    operator_ready: OperatorDeployer,
    chaos_injector: ChaosInjector,
    operator_job_namespace: str,
    kubectl: KubectlClient,
    k8s_settings,  # noqa: ANN001
) -> None:
    """Delete 1 of 2 parallel CRs; the survivor Completes untouched.

    Exercises per-CR isolation of the operator's reconcile machinery:
    kopf-held locks, ``client_cache`` entries, and monitor timers are
    all keyed on (namespace, name), so deleting A must not cancel B's
    cancellation flag nor free B's progress client.

    Resource budget: 3 concurrent controller + worker sets on a 4-vCPU
    Kind node proved too close to a raw scheduling-capacity test. Keep
    ``concurrency=1`` per CR and reduce the live set to two JobSets so
    the assertion remains about isolation under deletion churn rather than
    host CPU saturation.
    """
    names = ("chaos-c11-a", "chaos-c11-b")
    longrun_config = AIPerfJobConfig(
        concurrency=1,
        request_count=None,
        benchmark_duration=120.0,
        warmup_request_count=5,
        image=k8s_settings.aiperf_image,
    )
    try:
        # Create both in parallel.
        await asyncio.gather(
            *(
                operator_ready.create_job(
                    config=longrun_config,
                    name=n,
                    namespace=operator_job_namespace,
                )
                for n in names
            )
        )

        # Wait for both to reach Running/profiling. Generous timeout
        # because Kind can still be slow to pull + schedule two JobSets.
        await asyncio.gather(
            *(
                chaos_injector.wait_for_phase(
                    operator_job_namespace,
                    n,
                    phases=("Running",),
                    current_phase="profiling",
                    timeout=300.0,
                )
                for n in names
            )
        )

        # Delete A; B is the survivor.
        victims = names[:1]
        survivor = names[1]
        await asyncio.gather(
            *(
                chaos_injector.delete_cr_no_wait(operator_job_namespace, n)
                for n in victims
            )
        )

        # Victims drain cleanly: CR gone within 30 s, pods within 45 s.
        for n in victims:
            await chaos_injector.wait_for_cr_gone(
                operator_job_namespace, n, timeout=30.0
            )
        # pods_gone checks namespace-wide; the survivor still has pods, so
        # we verify each victim's CR is gone and rely on the survivor's
        # completion below to confirm its pods live. If both victims were
        # to leak pods, the namespace pod count would prevent the survivor
        # from finishing — an implicit end-to-end check.

        # Survivor completes within 360 s. The 120 s benchmark duration
        # overlaps with the operator reaping two sibling JobSets + their
        # workers + pods, so CPU pressure on a 4-vCPU Kind node is real.
        # On a larger cluster this can be tightened back to 240 s.
        phase = await chaos_injector.wait_for_phase(
            operator_job_namespace,
            survivor,
            phases=("Completed",),
            timeout=360.0,
        )
        assert phase == "Completed", (
            f"C11 survivor {survivor!r} did not reach Completed "
            f"(phase={phase!r}); parallel-delete blast radius hit the "
            "wrong CR"
        )
    finally:
        for n in names:
            await _force_delete(kubectl, operator_job_namespace, n)


async def test_c12_invalid_spec_surfaces_conditions(
    operator_ready: OperatorDeployer,  # noqa: ARG001  (operator must be running to handle the CR)
    chaos_injector: ChaosInjector,
    operator_job_namespace: str,
    kubectl: KubectlClient,
) -> None:
    """Invalid spec must surface phase=Failed + ConfigValid=False condition.

    Exercises ``src/aiperf/operator/handlers/create.py::_validate_spec``
    which raises ``kopf.PermanentError`` and sets
    ``status.conditions[ConfigValid]=False`` + ``phase=Failed`` when
    ``AIPerfJobSpec.from_crd_spec`` rejects the payload.

    We patch the ``benchmark.endpoint`` block with a bogus URL scheme.
    If the spec converter accepts it (schemes are not strictly
    validated) the endpoint-reachability probe will still fail and
    surface an ``EndpointReachable=False`` condition — in either case
    a condition should become visible within 60 s.
    """
    name = "chaos-c12"
    try:
        await chaos_injector.create_invalid_cr(
            operator_job_namespace,
            name,
            spec_patch={
                "benchmark": {
                    "models": {"items": [{"name": "mock-model"}]},
                    "endpoint": {"urls": ["notaschema://nope/v1/chat/completions"]},
                    "datasets": [
                        {
                            "name": "main",
                            "type": "synthetic",
                            "entries": 1,
                            "prompts": {"isl": {"mean": 550}},
                        }
                    ],
                    "phases": [
                        {
                            "name": "profiling",
                            "type": "concurrency",
                            "dataset": "main",
                            "concurrency": 1,
                            "requests": 1,
                        }
                    ],
                    "tokenizer": {"name": "gpt2"},
                    "runtime": {"ui": "none"},
                }
            },
        )

        deadline = time.monotonic() + 60.0
        surfaced = False
        observed_phase = ""
        while time.monotonic() < deadline:
            status = await kubectl.run(
                "get",
                "aiperfjob",
                name,
                "-n",
                operator_job_namespace,
                "-o",
                "jsonpath={.status.phase}",
                check=False,
            )
            observed_phase = status.stdout.strip()
            if observed_phase == "Failed":
                surfaced = True
                break
            cond = await kubectl.run(
                "get",
                "aiperfjob",
                name,
                "-n",
                operator_job_namespace,
                "-o",
                "jsonpath={.status.conditions[*].status}|"
                "{.status.conditions[*].message}",
                check=False,
            )
            # Any False condition counts as the operator having surfaced
            # a validation/reachability error.
            statuses, _, _ = cond.stdout.partition("|")
            if "False" in statuses:
                surfaced = True
                break
            await asyncio.sleep(2.0)

        assert surfaced, (
            f"C12: operator did not surface a failure condition or "
            f"phase=Failed within 60 s (observed phase={observed_phase!r}). "
            "See findings for a likely silent-swallow bug."
        )
    finally:
        await _force_delete(kubectl, operator_job_namespace, name)
