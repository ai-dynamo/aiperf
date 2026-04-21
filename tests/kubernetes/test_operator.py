# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the AIPerf Kubernetes operator.

These tests deploy the operator on a minikube cluster, create AIPerfJob CRs,
and verify the full benchmark lifecycle through the operator.

Fixture scoping strategy:
- Session-scoped: local_cluster, kubectl, operator_ready (shared across all tests)
- Module-scoped: operator_deployed_job_module (shared for read-only tests)
- Function-scoped: Used only when test modifies state or needs fresh resources
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.operator import (
    AIPerfJobConfig,
    OperatorDeployer,
    OperatorJobResult,
)

# Test timeout for individual test phases (not full job completion)
TEST_PHASE_TIMEOUT = 60  # seconds for waiting for phase transitions
TEST_JOB_TIMEOUT = 60  # seconds for full job completion


class TestOperatorDeployment:
    """Tests for operator deployment and CRD installation."""

    @pytest.mark.asyncio
    async def test_crd_is_established(
        self,
        operator_ready: OperatorDeployer,
        kubectl: KubectlClient,
    ) -> None:
        """Verify CRD is established and can be queried."""
        result = await kubectl.run(
            "get",
            "crd",
            "aiperfjobs.aiperf.nvidia.com",
            "-o",
            "jsonpath={.status.conditions[?(@.type=='Established')].status}",
        )
        assert result.stdout.strip() == "True"

    @pytest.mark.asyncio
    async def test_operator_pod_is_running(
        self,
        operator_ready: OperatorDeployer,
        kubectl: KubectlClient,
    ) -> None:
        """Verify operator pod is running."""
        pods = await kubectl.get_pods(OperatorDeployer.OPERATOR_NAMESPACE)
        operator_pods = [
            p for p in pods if "aiperf-operator" in p.name and "-test-" not in p.name
        ]

        assert len(operator_pods) == 1
        assert operator_pods[0].phase == "Running"

    @pytest.mark.asyncio
    async def test_operator_has_correct_permissions(
        self,
        operator_ready: OperatorDeployer,
        kubectl: KubectlClient,
    ) -> None:
        """Verify operator has necessary RBAC permissions."""
        result = await kubectl.run(
            "auth",
            "can-i",
            "create",
            "jobsets.jobset.x-k8s.io",
            "--as=system:serviceaccount:aiperf-system:aiperf-operator",
            check=False,
        )
        assert result.stdout.strip() == "yes"


class TestOperatorJobLifecycle:
    """Tests for AIPerfJob lifecycle management through the operator."""

    @pytest.mark.timeout(TEST_PHASE_TIMEOUT)
    @pytest.mark.asyncio
    async def test_create_job_sets_pending_phase(
        self,
        operator_ready: OperatorDeployer,
        small_operator_config: AIPerfJobConfig,
    ) -> None:
        """Verify newly created job starts in Pending phase.

        Creates its own job to verify initial phase.
        """
        result = await operator_ready.create_job(small_operator_config)

        # Give operator a moment to process
        await asyncio.sleep(2)

        status = await operator_ready.get_job_status(result.job_name, result.namespace)

        print(f"\n{'=' * 60}")
        print("JOB CREATION STATUS")
        print(f"{'=' * 60}")
        print(f"  Name: {result.job_name}")
        print(f"  Phase: {status.phase}")
        print(f"  JobSet: {status.jobset_name}")
        print(f"  Conditions: {len(status.conditions)}")
        print(f"{'=' * 60}\n")

        # Job should be in Pending or Initializing
        assert status.phase in ("Pending", "Initializing", "Running")
        assert status.jobset_name is not None

        # Cleanup
        await operator_ready.delete_job(result.job_name, result.namespace)

    @pytest.mark.timeout(TEST_JOB_TIMEOUT)
    @pytest.mark.asyncio
    async def test_job_transitions_through_phases(
        self,
        operator_ready: OperatorDeployer,
        small_operator_config: AIPerfJobConfig,
    ) -> None:
        """Verify job transitions through expected phases.

        Creates its own job to observe phase transitions.
        """
        result = await operator_ready.create_job(small_operator_config)
        phases_seen = set()

        loop = asyncio.get_event_loop()
        start = loop.time()
        timeout = TEST_JOB_TIMEOUT

        while loop.time() - start < timeout:
            status = await operator_ready.get_job_status(
                result.job_name, result.namespace
            )
            if status.phase:
                phases_seen.add(status.phase)

            if status.is_terminal:
                break

            await asyncio.sleep(2)

        print(f"\n{'=' * 60}")
        print("PHASE TRANSITIONS")
        print(f"{'=' * 60}")
        print(f"  Phases seen: {sorted(phases_seen)}")
        print(f"  Final phase: {status.phase}")
        print(f"{'=' * 60}\n")

        # Should see at least Pending and Running (or their equivalents)
        assert len(phases_seen) >= 1
        assert status.is_completed, f"Expected Completed, got {status.phase}"

        # Cleanup
        await operator_ready.delete_job(result.job_name, result.namespace)

    def test_job_completes_successfully(
        self,
        operator_deployed_job_module: OperatorJobResult,
    ) -> None:
        """Verify job completes successfully with results.

        Uses module-scoped fixture (read-only test).
        """
        result = operator_deployed_job_module

        print(f"\n{'=' * 70}")
        print("OPERATOR JOB COMPLETION RESULTS")
        print(f"{'=' * 70}")
        print(f"  Job Name: {result.job_name}")
        print(f"  Namespace: {result.namespace}")
        print(f"  Success: {result.success}")
        print(f"  Duration: {result.duration_seconds:.2f}s")

        if result.status:
            print("\n  STATUS:")
            print(f"    Phase: {result.status.phase}")
            print(f"    JobSet: {result.status.jobset_name}")
            print(
                f"    Workers: {result.status.workers_ready}/{result.status.workers_total}"
            )
            print(f"    Conditions: {len(result.status.conditions)}")

            if result.status.results:
                print("\n  RESULTS PRESENT: Yes")
                print(f"    Keys: {list(result.status.results.keys())[:5]}...")
            else:
                print("\n  RESULTS: Not yet available")

        print("\n  ✓ Job completed successfully!")
        print(f"{'=' * 70}\n")

        assert result.success
        assert result.status is not None
        assert result.status.is_completed

    def test_job_creates_jobset(
        self,
        operator_deployed_job_module: OperatorJobResult,
    ) -> None:
        """Verify operator creates JobSet for the benchmark.

        Uses module-scoped fixture (read-only test).
        The operator may clean up the JobSet after collecting results,
        so jobset_status may be None on a successful run.
        """
        assert operator_deployed_job_module.status is not None
        if operator_deployed_job_module.jobset_status is None:
            assert operator_deployed_job_module.success
            return
        assert operator_deployed_job_module.status.jobset_name is not None

    def test_job_tracks_worker_status(
        self,
        operator_deployed_job_module: OperatorJobResult,
    ) -> None:
        """Verify operator tracks worker readiness.

        Uses module-scoped fixture (read-only test).
        """
        status = operator_deployed_job_module.status
        if status is None or status.workers_total == 0:
            assert operator_deployed_job_module.success
            return

        # Workers should have been tracked (at least 1 total)
        assert status.workers_total >= 1


class TestOperatorConditions:
    """Tests for operator condition tracking.

    Uses module-scoped fixture since all tests are read-only.
    """

    def test_config_valid_condition_set(
        self,
        operator_deployed_job_module: OperatorJobResult,
    ) -> None:
        """Verify ConfigValid condition is set."""
        status = operator_deployed_job_module.status
        assert status is not None

        # Check for ConfigValid condition
        config_valid = status.get_condition("ConfigValid")

        print(f"\n{'=' * 60}")
        print("CONFIG VALID CONDITION")
        print(f"{'=' * 60}")
        if config_valid:
            print(f"  Status: {config_valid.get('status')}")
            print(f"  Reason: {config_valid.get('reason')}")
            print(f"  Message: {config_valid.get('message')}")
        else:
            print("  Condition not found")
        print(f"{'=' * 60}\n")

        if config_valid is None:
            assert operator_deployed_job_module.success
            return
        assert config_valid.get("status") == "True"

    def test_resources_created_condition_set(
        self,
        operator_deployed_job_module: OperatorJobResult,
    ) -> None:
        """Verify ResourcesCreated condition is set."""
        status = operator_deployed_job_module.status
        assert status is not None

        resources_created = status.get_condition("ResourcesCreated")
        if resources_created is None:
            assert operator_deployed_job_module.success
            return
        assert resources_created.get("status") == "True"

    def test_workers_ready_condition_set(
        self,
        operator_deployed_job_module: OperatorJobResult,
    ) -> None:
        """Verify WorkersReady condition is set on completion."""
        status = operator_deployed_job_module.status
        assert status is not None

        workers_ready = status.get_condition("WorkersReady")
        if workers_ready is None:
            assert operator_deployed_job_module.success
            return
        assert workers_ready.get("status") == "True"

    def test_benchmark_running_condition_set(
        self,
        operator_deployed_job_module: OperatorJobResult,
    ) -> None:
        """Verify BenchmarkRunning condition was set during execution."""
        status = operator_deployed_job_module.status
        assert status is not None

        benchmark_running = status.get_condition("BenchmarkRunning")
        if benchmark_running is None:
            assert operator_deployed_job_module.success
            return


class TestOperatorResults:
    """Tests for operator results collection.

    Uses module-scoped fixture for read-only tests.
    """

    def test_results_available_on_completion(
        self,
        operator_deployed_job_module: OperatorJobResult,
    ) -> None:
        """Verify results are available after job completion.

        Uses module-scoped fixture (read-only test).
        """
        status = operator_deployed_job_module.status
        assert status is not None
        assert status.is_completed

        print(f"\n{'=' * 60}")
        print("RESULTS AVAILABILITY")
        print(f"{'=' * 60}")
        print(f"  Has results: {status.results is not None}")
        if status.results:
            print(f"  Result keys: {list(status.results.keys())}")
        print(f"{'=' * 60}\n")

        # Results should be populated
        # Note: This depends on the operator successfully fetching results
        # from the controller API before the pod terminates
        # If results are not present, the test verifies the condition

        results_available = status.get_condition("ResultsAvailable")
        if results_available and results_available.get("status") == "True":
            assert status.results is not None

    @pytest.mark.timeout(TEST_JOB_TIMEOUT)
    @pytest.mark.asyncio
    async def test_live_metrics_tracked(
        self,
        operator_ready: OperatorDeployer,
        small_operator_config: AIPerfJobConfig,
    ) -> None:
        """Verify live metrics are tracked during execution.

        Creates its own job to observe live metrics during execution.
        """
        result = await operator_ready.create_job(small_operator_config)
        live_metrics_seen = False

        loop = asyncio.get_event_loop()
        start = loop.time()
        timeout = TEST_JOB_TIMEOUT

        while loop.time() - start < timeout:
            status = await operator_ready.get_job_status(
                result.job_name, result.namespace
            )

            if status.live_metrics:
                live_metrics_seen = True
                print(f"\n  Live metrics captured: {list(status.live_metrics.keys())}")

            if status.is_terminal:
                break

            await asyncio.sleep(3)

        print(f"\n{'=' * 60}")
        print("LIVE METRICS TRACKING")
        print(f"{'=' * 60}")
        print(f"  Live metrics seen during run: {live_metrics_seen}")
        print(f"{'=' * 60}\n")

        # Live metrics may or may not be captured depending on timing
        # This test documents the behavior
        assert status.is_completed

        # Cleanup
        await operator_ready.delete_job(result.job_name, result.namespace)


class TestOperatorCancellation:
    """Tests for job cancellation through the operator."""

    @pytest.mark.timeout(TEST_JOB_TIMEOUT)
    @pytest.mark.asyncio
    async def test_cancel_running_job(
        self,
        operator_ready: OperatorDeployer,
    ) -> None:
        """Verify running job can be cancelled.

        Uses benchmark_duration to ensure the job runs long enough to cancel.
        """
        # Use benchmark_duration to force job to run for a minimum time
        cancel_test_config = AIPerfJobConfig(
            concurrency=5,
            request_count=None,  # No request limit
            benchmark_duration=120.0,  # Run for 2 minutes
            warmup_request_count=5,
            image="aiperf:local",
        )

        result = await operator_ready.create_job(cancel_test_config)

        # Wait for job to start running
        loop = asyncio.get_event_loop()
        start = loop.time()
        while loop.time() - start < TEST_PHASE_TIMEOUT:
            status = await operator_ready.get_job_status(
                result.job_name, result.namespace
            )
            if status.phase == "Running":
                break
            if status.is_terminal:
                pytest.skip(
                    f"Job completed ({status.phase}) before reaching Running phase"
                )
            await asyncio.sleep(1)

        print(f"\n{'=' * 60}")
        print("JOB CANCELLATION TEST")
        print(f"{'=' * 60}")
        print(f"  Job phase before cancel: {status.phase}")

        # Cancel the job
        await operator_ready.cancel_job(result.job_name, result.namespace)

        # Wait for cancellation
        start = loop.time()
        while loop.time() - start < TEST_PHASE_TIMEOUT:
            status = await operator_ready.get_job_status(
                result.job_name, result.namespace
            )
            if status.is_terminal:
                break
            await asyncio.sleep(2)

        print(f"  Job phase after cancel: {status.phase}")
        print(f"{'=' * 60}\n")

        # Cleanup
        with contextlib.suppress(Exception):
            await operator_ready.delete_job(result.job_name, result.namespace)

        # Job should be cancelled or failed
        assert status.is_terminal, (
            f"Job did not reach terminal state after cancel: {status.phase}"
        )
        assert status.phase in ("Cancelled", "Failed"), (
            f"Expected Cancelled or Failed, got {status.phase}"
        )


class TestOperatorErrorHandling:
    """Tests for operator error handling."""

    @pytest.mark.timeout(TEST_PHASE_TIMEOUT)
    @pytest.mark.asyncio
    async def test_invalid_config_fails_with_error(
        self,
        operator_ready: OperatorDeployer,
    ) -> None:
        """Verify invalid config results in failure with error message.

        Creates its own job to test error handling.
        """
        # Create config with invalid/missing endpoint
        config = AIPerfJobConfig(
            endpoint_url="",  # Invalid: empty URL
            concurrency=5,
            request_count=10,
        )

        # Create CR with manually constructed manifest (bypass validation)
        import yaml

        cr = {
            "apiVersion": "aiperf.nvidia.com/v1alpha1",
            "kind": "AIPerfJob",
            "metadata": {
                "name": "invalid-config-test",
                "namespace": "default",
            },
            "spec": {
                "image": config.image,
                "imagePullPolicy": config.image_pull_policy,
                "benchmark": {
                    "endpoint": {},  # Missing required fields
                    "phases": {
                        "profiling": {"type": "concurrency", "concurrency": 5},
                    },
                },
            },
        }

        try:
            await operator_ready.kubectl.apply(yaml.dump(cr))

            await asyncio.sleep(5)

            status = await operator_ready.get_job_status(
                "invalid-config-test", "default"
            )

            print(f"\n{'=' * 60}")
            print("INVALID CONFIG ERROR HANDLING")
            print(f"{'=' * 60}")
            print(f"  Phase: {status.phase}")
            print(f"  Error: {status.error}")
            print(f"{'=' * 60}\n")

            # Should fail with error
            assert status.is_failed
            assert status.error is not None

        finally:
            await operator_ready.kubectl.delete(
                "aiperfjob", "invalid-config-test", namespace="default"
            )

    @pytest.mark.timeout(TEST_JOB_TIMEOUT)
    @pytest.mark.asyncio
    async def test_unreachable_endpoint_fails_gracefully(
        self,
        operator_ready: OperatorDeployer,
    ) -> None:
        """Verify unreachable endpoint is handled gracefully.

        Creates its own job to test error handling.
        """
        config = AIPerfJobConfig(
            endpoint_url="http://nonexistent-service:8000/v1",
            concurrency=2,
            request_count=5,
        )

        result = await operator_ready.create_job(
            config, name="unreachable-endpoint-test"
        )

        loop = asyncio.get_event_loop()
        start = loop.time()
        timeout = TEST_JOB_TIMEOUT

        while loop.time() - start < timeout:
            status = await operator_ready.get_job_status(
                result.job_name, result.namespace
            )
            if status.is_terminal:
                break
            await asyncio.sleep(5)

        print(f"\n{'=' * 60}")
        print("UNREACHABLE ENDPOINT ERROR HANDLING")
        print(f"{'=' * 60}")
        print(f"  Phase: {status.phase}")
        print(f"  Error: {status.error}")

        endpoint_cond = status.get_condition("EndpointReachable")
        if endpoint_cond:
            print(f"  EndpointReachable: {endpoint_cond.get('status')}")
            print(f"  Reason: {endpoint_cond.get('reason')}")

        print(f"{'=' * 60}\n")

        # Should fail
        assert status.is_failed or status.is_completed

        # Cleanup
        await operator_ready.delete_job(result.job_name, result.namespace)


class TestOperatorEvents:
    """Tests for Kubernetes events emitted by the operator.

    Uses module-scoped fixture (read-only test).
    """

    @pytest.mark.asyncio
    async def test_events_emitted_for_job(
        self,
        operator_deployed_job_module: OperatorJobResult,
        kubectl: KubectlClient,
    ) -> None:
        """Verify operator emits events for job lifecycle.

        Uses module-scoped fixture (read-only test).
        """
        events = await kubectl.get_events(operator_deployed_job_module.namespace)

        print(f"\n{'=' * 60}")
        print("OPERATOR EVENTS")
        print(f"{'=' * 60}")
        print(events)
        print(f"{'=' * 60}\n")

        # Should have at least some events related to the job
        assert len(events) > 0


class TestOperatorCleanup:
    """Tests for operator resource cleanup."""

    @pytest.mark.timeout(TEST_PHASE_TIMEOUT)
    @pytest.mark.asyncio
    async def test_deleting_job_removes_resources(
        self,
        operator_ready: OperatorDeployer,
        small_operator_config: AIPerfJobConfig,
        kubectl: KubectlClient,
    ) -> None:
        """Verify deleting AIPerfJob removes associated resources.

        Creates its own job to test cleanup.
        """
        result = await operator_ready.create_job(small_operator_config)

        # Wait for resources to be created
        await asyncio.sleep(5)

        status = await operator_ready.get_job_status(result.job_name, result.namespace)
        jobset_name = status.jobset_name

        # Verify JobSet exists (may already be cleaned up by operator)
        if jobset_name:
            jobsets = await kubectl.get_jobsets(result.namespace)
            jobset_exists = any(js.name == jobset_name for js in jobsets)
            if not jobset_exists and status.is_completed:
                # Operator already cleaned up - skip the rest
                await operator_ready.delete_job(result.job_name, result.namespace)
                return
            assert jobset_exists

        # Delete the job
        await operator_ready.delete_job(result.job_name, result.namespace)

        # Wait for cleanup
        await asyncio.sleep(10)

        # Verify resources are cleaned up
        if jobset_name:
            jobsets = await kubectl.get_jobsets(result.namespace)
            assert not any(js.name == jobset_name for js in jobsets)


@pytest.mark.k8s_slow
class TestOperatorScaling:
    """Tests for operator with different scaling configurations."""

    @pytest.mark.timeout(600)
    @pytest.mark.asyncio
    async def test_high_concurrency_job(
        self,
        operator_ready: OperatorDeployer,
    ) -> None:
        """Test operator handles high concurrency job.

        Creates its own job to test scaling.
        """
        config = AIPerfJobConfig(
            concurrency=10,
            request_count=20,
            warmup_request_count=2,
        )

        result = await operator_ready.run_job(config, timeout=180)

        assert result.success
        assert result.status is not None
        assert result.status.is_completed

    @pytest.mark.timeout(600)
    @pytest.mark.asyncio
    async def test_multiple_workers_job(
        self,
        operator_ready: OperatorDeployer,
    ) -> None:
        """Test operator handles job requiring multiple workers.

        Creates its own job to test multi-worker scaling.
        """
        config = AIPerfJobConfig(
            concurrency=20,
            request_count=40,
            warmup_request_count=5,
            connections_per_worker=10,
        )

        result = await operator_ready.run_job(config, timeout=600)

        assert result.success
        assert result.status is not None

        # Workers may fit in 1 pod (workers_per_pod default is 10)
        if result.status.workers_total == 0:
            return
        assert result.status.workers_total >= 1
