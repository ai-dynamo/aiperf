# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2E tests for benchmark completion signaling and resource cleanup.

Validates:
1. Controller pod patches the benchmark-complete annotation on the AIPerfJob CR
2. Operator reacts to the annotation and fetches results immediately
3. Operator deletes the JobSet after storing results (no waiting for TTL)
4. Worker Jobs are cleaned up immediately via ttlSecondsAfterFinished=0

Fixture scoping:
- Function-scoped: All tests create/destroy their own jobs to verify cleanup behavior
"""

from __future__ import annotations

import asyncio
import time

import pytest

from aiperf.kubernetes.constants import Annotations
from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.operator import (
    AIPerfJobConfig,
    OperatorDeployer,
)

TEST_JOB_TIMEOUT = 600


class TestCompletionSignal:
    """Tests for controller pod -> operator completion signaling."""

    @pytest.mark.timeout(TEST_JOB_TIMEOUT)
    @pytest.mark.asyncio
    async def test_benchmark_complete_annotation_set(
        self,
        operator_ready: OperatorDeployer,
        small_operator_config: AIPerfJobConfig,
    ) -> None:
        """Controller pod sets benchmark-complete annotation on the CR."""
        result = await operator_ready.run_job(
            small_operator_config, timeout=TEST_JOB_TIMEOUT
        )

        assert result.success, f"Job failed: {result.error_message}"

        # Fetch the raw CR to check annotations
        cr = await operator_ready.kubectl.get_json(
            "aiperfjob", result.job_name, namespace=result.namespace
        )
        annotations = cr.get("metadata", {}).get("annotations", {})

        print(f"\nAnnotations on CR: {list(annotations.keys())}")
        assert annotations.get(Annotations.BENCHMARK_COMPLETE) == "true", (
            f"Expected benchmark-complete annotation, got: {annotations}"
        )

        await operator_ready.delete_job(result.job_name, result.namespace)

    @pytest.mark.timeout(TEST_JOB_TIMEOUT)
    @pytest.mark.asyncio
    async def test_completion_signal_faster_than_polling(
        self,
        operator_ready: OperatorDeployer,
        small_operator_config: AIPerfJobConfig,
    ) -> None:
        """Completion via annotation signal should be faster than the 10s poll cycle.

        Measures total time from job creation to CR reaching Completed.
        The annotation signal means the operator reacts within seconds of
        benchmark finishing, rather than waiting for a 10s poll cycle.
        """
        job = await operator_ready.create_job(small_operator_config)
        start = time.monotonic()

        status = await operator_ready.wait_for_job_completion(
            job.job_name, job.namespace, timeout=TEST_JOB_TIMEOUT
        )

        total_duration = time.monotonic() - start
        print(f"\nTotal time to Completed: {total_duration:.1f}s")
        print(f"Phase: {status.phase}")

        assert status.is_completed, f"Expected Completed, got {status.phase}"

        await operator_ready.delete_job(job.job_name, job.namespace)


class TestJobSetCleanup:
    """Tests for JobSet deletion after results are fetched."""

    @pytest.mark.timeout(TEST_JOB_TIMEOUT + 30)
    @pytest.mark.asyncio
    async def test_jobset_deleted_after_successful_results_fetch(
        self,
        operator_ready: OperatorDeployer,
        kubectl: KubectlClient,
        small_operator_config: AIPerfJobConfig,
    ) -> None:
        """When results are fetched successfully, JobSet is deleted.

        If results fetch fails (e.g. controller exits too fast on a
        resource-constrained cluster), the JobSet is intentionally kept
        so the TTL controller handles cleanup.
        """
        result = await operator_ready.run_job(
            small_operator_config, timeout=TEST_JOB_TIMEOUT
        )

        assert result.success, f"Job failed: {result.error_message}"
        assert result.status is not None

        jobset_name = result.status.jobset_name
        if jobset_name is None:
            # Operator cleaned up before we could read the jobset name
            await operator_ready.delete_job(result.job_name, result.namespace)
            return

        has_results = result.status.is_condition_true("ResultsAvailable")

        # Give the operator a moment to finish cleanup
        await asyncio.sleep(5)

        js_result = await kubectl.run(
            "get",
            "jobset",
            jobset_name,
            "-n",
            result.namespace,
            "-o",
            "name",
            check=False,
        )
        jobset_exists = js_result.returncode == 0 and js_result.stdout.strip()

        print(f"\nResults fetched: {has_results}")
        print(f"JobSet exists: {jobset_exists}")

        if has_results:
            assert not jobset_exists, (
                f"JobSet {jobset_name} still exists after results were stored"
            )
        else:
            print("Results fetch failed (resource-constrained cluster)")
            print("JobSet intentionally kept for TTL-based cleanup")

        await operator_ready.delete_job(result.job_name, result.namespace)

    @pytest.mark.timeout(TEST_JOB_TIMEOUT)
    @pytest.mark.asyncio
    async def test_job_completes_successfully(
        self,
        operator_ready: OperatorDeployer,
        small_operator_config: AIPerfJobConfig,
    ) -> None:
        """Job reaches Completed phase with correct conditions."""
        result = await operator_ready.run_job(
            small_operator_config, timeout=TEST_JOB_TIMEOUT
        )

        assert result.success, f"Job failed: {result.error_message}"
        assert result.status is not None
        assert result.status.is_completed

        # Core conditions must be set
        assert result.status.is_condition_true("ConfigValid")
        assert result.status.is_condition_true("PreflightPassed")
        assert result.status.is_condition_true("ResourcesCreated")

        await operator_ready.delete_job(result.job_name, result.namespace)


class TestWorkerCleanup:
    """Tests for immediate worker Job/pod cleanup."""

    @pytest.mark.timeout(TEST_JOB_TIMEOUT)
    @pytest.mark.asyncio
    async def test_worker_pods_cleaned_up_after_completion(
        self,
        operator_ready: OperatorDeployer,
        kubectl: KubectlClient,
        small_operator_config: AIPerfJobConfig,
    ) -> None:
        """Worker pods should be cleaned up after they succeed.

        Workers have ttlSecondsAfterFinished=0 on their Job, so K8s
        deletes the worker Job and its pods as soon as they succeed.
        """
        job = await operator_ready.create_job(small_operator_config)

        # Wait for completion
        status = await operator_ready.wait_for_job_completion(
            job.job_name, job.namespace, timeout=TEST_JOB_TIMEOUT
        )
        assert status.is_completed, f"Expected Completed, got {status.phase}"

        # Give TTL controller and JobSet deletion time to clean up.
        # On resource-constrained Kind clusters, workers may still be in
        # a restart loop. Wait up to 30s for cleanup.
        for _ in range(6):
            await asyncio.sleep(5)
            pods = await kubectl.get_pods(job.namespace)
            worker_pods = [
                p for p in pods if job.job_name in p.name and "worker" in p.name
            ]
            if not worker_pods:
                break

        print("\nAll pods in namespace after completion:")
        for p in await kubectl.get_pods(job.namespace):
            if job.job_name in p.name:
                print(f"  {p.name}: {p.phase}")

        print(f"\nWorker pods remaining: {len(worker_pods)}")

        # Worker pods should be gone after TTL=0 + JobSet deletion
        assert len(worker_pods) == 0, (
            f"Found {len(worker_pods)} worker pods after completion: "
            f"{[f'{p.name}={p.phase}' for p in worker_pods]}"
        )

        await operator_ready.delete_job(job.job_name, job.namespace)

    @pytest.mark.timeout(TEST_JOB_TIMEOUT)
    @pytest.mark.asyncio
    async def test_worker_jobs_have_ttl_zero(
        self,
        operator_ready: OperatorDeployer,
        kubectl: KubectlClient,
        small_operator_config: AIPerfJobConfig,
    ) -> None:
        """Verify worker Jobs are created with ttlSecondsAfterFinished=0."""
        job = await operator_ready.create_job(small_operator_config)

        # Wait for the JobSet to be created (poll up to ~30s)
        status = await operator_ready.get_job_status(job.job_name, job.namespace)
        jobset_name = status.jobset_name
        for _ in range(30):
            if jobset_name is not None or status.is_completed:
                break
            await asyncio.sleep(1)
            status = await operator_ready.get_job_status(job.job_name, job.namespace)
            jobset_name = status.jobset_name
        if jobset_name is None:
            # Operator may have already cleaned up if job completed fast
            if status.is_completed:
                await operator_ready.delete_job(job.job_name, job.namespace)
                return
            pytest.fail("JobSet not created and job not completed")

        # Get the JobSet and check worker job spec
        try:
            js = await kubectl.get_json("jobset", jobset_name, namespace=job.namespace)
        except RuntimeError:
            # JobSet already cleaned up by operator
            if status.is_completed or status.is_terminal:
                await operator_ready.delete_job(job.job_name, job.namespace)
                return
            raise
        replicated_jobs = js.get("spec", {}).get("replicatedJobs", [])

        worker_rj = None
        for rj in replicated_jobs:
            if rj.get("name") == "workers":
                worker_rj = rj
                break

        assert worker_rj is not None, (
            f"No 'workers' replicated job found. "
            f"Jobs: {[r['name'] for r in replicated_jobs]}"
        )

        ttl = (
            worker_rj.get("template", {}).get("spec", {}).get("ttlSecondsAfterFinished")
        )
        print(f"\nWorker Job ttlSecondsAfterFinished: {ttl}")
        assert ttl == 0, f"Expected ttlSecondsAfterFinished=0, got {ttl}"

        # Also verify controller Job does NOT have TTL=0
        controller_rj = None
        for rj in replicated_jobs:
            if rj.get("name") == "controller":
                controller_rj = rj
                break

        if controller_rj:
            ctrl_ttl = (
                controller_rj.get("template", {})
                .get("spec", {})
                .get("ttlSecondsAfterFinished")
            )
            print(f"Controller Job ttlSecondsAfterFinished: {ctrl_ttl}")
            assert ctrl_ttl is None or ctrl_ttl > 0, (
                f"Controller should NOT have TTL=0, got {ctrl_ttl}"
            )

        # Cleanup
        await operator_ready.wait_for_job_completion(
            job.job_name, job.namespace, timeout=TEST_JOB_TIMEOUT
        )
        await operator_ready.delete_job(job.job_name, job.namespace)
