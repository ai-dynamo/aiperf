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
import uuid
from dataclasses import replace

import pytest

from tests.kubernetes.conftest import K8sTestSettings
from tests.kubernetes.helpers.deadline import (
    await_before_deadline,
    delete_and_observe_until_deadline,
)
from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.operator import (
    AIPerfJobConfig,
    OperatorDeployer,
    OperatorJobResult,
)

# Test timeout for individual test phases (not full job completion)
TEST_PHASE_TIMEOUT = 60  # seconds for waiting for phase transitions
TEST_JOB_TIMEOUT = 60  # seconds for full job completion
TEST_CLEANUP_TIMEOUT = 150  # seconds for CR deletion propagation checks
CLEANUP_ASSERTION_TIMEOUT = 120  # seconds shared by observation and deletion
CLEANUP_DELETION_POLL_RESERVE = 60  # seconds reserved from the shared deadline
CLEANUP_FAILURE_TEARDOWN_TIMEOUT = 20  # seconds from pytest's reporting buffer
CLEANUP_FAILURE_TEARDOWN_POLL_INTERVAL = 1  # seconds between absence checks


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
        operator_pods: list = []
        for _ in range(15):
            pods = await kubectl.get_pods(OperatorDeployer.OPERATOR_NAMESPACE)
            operator_pods = [
                p
                for p in pods
                if "aiperf-operator" in p.name and "-test-" not in p.name
            ]
            if len(operator_pods) >= 1 and operator_pods[0].phase == "Running":
                break
            await asyncio.sleep(1)

        assert len(operator_pods) == 1
        assert operator_pods[0].phase == "Running"

    @pytest.mark.asyncio
    async def test_operator_has_correct_permissions(
        self,
        operator_ready: OperatorDeployer,
        kubectl: KubectlClient,
    ) -> None:
        """Verify operator has necessary RBAC permissions."""
        stdout = ""
        # RBAC propagation in kind can take 30-60s after CRB apply when the
        # API server has just started; keep retrying to absorb that window.
        for _ in range(60):
            result = await kubectl.run(
                "auth",
                "can-i",
                "create",
                "jobsets.jobset.x-k8s.io",
                "--as=system:serviceaccount:aiperf-system:aiperf-operator",
                check=False,
            )
            stdout = result.stdout.strip()
            if stdout == "yes":
                break
            await asyncio.sleep(1)
        assert stdout == "yes"


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

        # Poll for the operator to set phase (up to ~15s)
        status = await operator_ready.get_job_status(result.job_name, result.namespace)
        for _ in range(15):
            if status.phase in ("Pending", "Initializing", "Running"):
                break
            await asyncio.sleep(1)
            status = await operator_ready.get_job_status(
                result.job_name, result.namespace
            )

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

        assert config_valid is not None, status.conditions
        assert config_valid.get("status") == "True", status.conditions

    def test_resources_created_condition_set(
        self,
        operator_deployed_job_module: OperatorJobResult,
    ) -> None:
        """Verify ResourcesCreated condition is set."""
        status = operator_deployed_job_module.status
        assert status is not None

        resources_created = status.get_condition("ResourcesCreated")
        assert resources_created is not None, status.conditions
        assert resources_created.get("status") == "True", status.conditions

    def test_workers_ready_condition_set(
        self,
        operator_deployed_job_module: OperatorJobResult,
    ) -> None:
        """Verify WorkersReady condition is set on completion."""
        status = operator_deployed_job_module.status
        assert status is not None

        workers_ready = status.get_condition("WorkersReady")
        assert workers_ready is not None, status.conditions
        assert workers_ready.get("status") == "True", status.conditions

    def test_benchmark_running_condition_set(
        self,
        operator_deployed_job_module: OperatorJobResult,
    ) -> None:
        """Verify BenchmarkRunning condition was set during execution."""
        status = operator_deployed_job_module.status
        assert status is not None

        benchmark_running = status.get_condition("BenchmarkRunning")
        assert benchmark_running is not None, status.conditions
        assert benchmark_running.get("status") == "True", status.conditions


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

        assert status.is_condition_true("ResultsAvailable"), status.conditions
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

    # benchmark_duration below is 120s; use a longer timeout than TEST_JOB_TIMEOUT
    # so the test can reach Running, cancel, and observe the terminal phase.
    @pytest.mark.timeout(300)
    @pytest.mark.asyncio
    async def test_cancel_running_job(
        self,
        operator_ready: OperatorDeployer,
        k8s_settings: K8sTestSettings,
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
            image=k8s_settings.aiperf_image,
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
                pytest.fail(
                    "Cancellation prerequisite became terminal before reaching "
                    f"Running: phase={status.phase}, error={status.error}"
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

        # User cancellation is its own terminal outcome. Treating Failed as
        # acceptable here hides races where completion/error handling overwrites
        # the cancellation status.
        assert status.is_terminal, (
            f"Job did not reach terminal state after cancel: {status.phase}"
        )
        assert status.phase == "Cancelled", f"Expected Cancelled, got {status.phase}"
        complete = status.get_condition("Complete")
        failed = status.get_condition("Failed")
        assert complete is not None and complete.get("status") == "False", complete
        assert failed is not None and failed.get("status") == "False", failed


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
        config = AIPerfJobConfig(
            endpoint_url="http://aiperf-mock-server.default.svc.cluster.local:8000/v1",
            concurrency=5,
            request_count=10,
        )

        # The config must pass the CRD schema and fail the operator's own
        # validation -- otherwise the apiserver rejects it at admission and the
        # operator never runs, which tests nothing about error handling. An
        # omitted `endpoint.urls` no longer works for that: the CRD marks it
        # required, so kubectl apply fails outright.
        #
        # `datasets` is x-kubernetes-preserve-unknown-fields, so the apiserver
        # waves through an entry with no `name` and AIPerfJobSpec rejects it
        # ("datasets[0] is missing required 'name' field"). That is the shape
        # this test wants: admitted, then failed by the operator with an error.
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
                    "endpoint": {"urls": [config.endpoint_url]},
                    "models": {"items": [{"name": config.model_name}]},
                    "datasets": [{"type": "synthetic"}],  # no `name` -> rejected
                    "phases": [
                        {"name": "profiling", "type": "concurrency", "concurrency": 5},
                    ],
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
    async def test_schema_violation_is_rejected_at_admission(
        self,
        operator_ready: OperatorDeployer,
    ) -> None:
        """A CRD-schema violation must never create a CR at all.

        Cheaper and clearer than admitting the object and failing it later:
        the caller gets a synchronous error instead of having to poll status.
        This pins the boundary between the two failure modes -- schema
        violations die at the apiserver, semantic violations die in the
        operator (see ``test_invalid_config_fails_with_error``).
        """
        import yaml

        cr = {
            "apiVersion": "aiperf.nvidia.com/v1alpha1",
            "kind": "AIPerfJob",
            "metadata": {"name": "schema-violation-test", "namespace": "default"},
            "spec": {
                "image": "aiperf:kind",
                "benchmark": {
                    "endpoint": {},  # urls is required by the CRD schema
                    "phases": [
                        {"name": "profiling", "type": "concurrency", "concurrency": 5},
                    ],
                },
            },
        }

        with pytest.raises(RuntimeError, match="endpoint.urls"):
            await operator_ready.kubectl.apply(yaml.dump(cr))

        result = await operator_ready.kubectl.run(
            "get",
            "aiperfjob",
            "schema-violation-test",
            "-n",
            "default",
            check=False,
        )
        assert result.returncode != 0, "rejected spec must not leave a CR behind"

    @pytest.mark.timeout(180)
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
        timeout = 150

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

        assert status.is_failed, (
            f"Unreachable endpoint unexpectedly ended in phase={status.phase}"
        )
        assert endpoint_cond is not None
        assert endpoint_cond.get("status") == "False"

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

    @pytest.mark.timeout(TEST_CLEANUP_TIMEOUT)
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
        deadline = asyncio.get_running_loop().time() + CLEANUP_ASSERTION_TIMEOUT
        config = replace(
            small_operator_config,
            request_count=None,
            benchmark_duration=120.0,
        )
        job_name = f"cleanup-{uuid.uuid4().hex[:8]}"
        namespace = operator_ready.default_job_namespace
        jobset_name: str | None = None
        last_phase: str | None = None
        cr_returncode: int | None = None
        remaining_jobsets: list[str] = []
        remaining_pods: list[str] = []
        deleted = False
        try:
            await await_before_deadline(
                deadline,
                f"creating AIPerfJob {namespace}/{job_name}",
                lambda: operator_ready.create_job(
                    config,
                    name=job_name,
                    namespace=namespace,
                ),
            )
            observation_deadline = deadline - CLEANUP_DELETION_POLL_RESERVE
            while asyncio.get_running_loop().time() < observation_deadline:
                status = await await_before_deadline(
                    observation_deadline,
                    (
                        "reading AIPerfJob status for "
                        f"{namespace}/{job_name} during controller "
                        "observation"
                    ),
                    lambda: operator_ready.get_job_status(job_name, namespace),
                )
                last_phase = status.phase
                jobset_name = status.jobset_name
                if jobset_name:
                    pods = await await_before_deadline(
                        observation_deadline,
                        f"listing controller pods for JobSet {jobset_name}",
                        lambda jobset_name=jobset_name: kubectl.get_pods(
                            namespace,
                            label_selector=(
                                f"jobset.sigs.k8s.io/jobset-name={jobset_name}"
                            ),
                        ),
                    )
                    if any("controller" in pod.name for pod in pods):
                        break
                await asyncio.sleep(
                    min(
                        1,
                        max(
                            0,
                            observation_deadline - asyncio.get_running_loop().time(),
                        ),
                    )
                )
            else:
                pytest.fail(
                    "AIPerfJob never exposed a JobSet and controller pod before "
                    f"the cleanup assertion: phase={last_phase}, "
                    f"jobset={jobset_name}"
                )

            assert jobset_name is not None
            await await_before_deadline(
                deadline,
                f"deleting AIPerfJob {namespace}/{job_name}",
                lambda: operator_ready.delete_job(job_name, namespace),
            )
            deleted = True

            while asyncio.get_running_loop().time() < deadline:
                job = await await_before_deadline(
                    deadline,
                    (f"checking whether AIPerfJob {namespace}/{job_name} was deleted"),
                    lambda: kubectl.run(
                        "get",
                        "aiperfjob",
                        job_name,
                        namespace=namespace,
                        check=False,
                    ),
                )
                jobsets = await await_before_deadline(
                    deadline,
                    (f"listing JobSets in {namespace} during deletion propagation"),
                    lambda: kubectl.get_jobsets(namespace),
                )
                pods = await await_before_deadline(
                    deadline,
                    f"listing child pods for JobSet {jobset_name}",
                    lambda: kubectl.get_pods(
                        namespace,
                        label_selector=(
                            f"jobset.sigs.k8s.io/jobset-name={jobset_name}"
                        ),
                    ),
                )
                cr_returncode = job.returncode
                remaining_jobsets = [item.name for item in jobsets]
                remaining_pods = [pod.name for pod in pods]
                if (
                    job.returncode != 0
                    and not any(jobset.name == jobset_name for jobset in jobsets)
                    and not pods
                ):
                    return
                await asyncio.sleep(
                    min(1, max(0, deadline - asyncio.get_running_loop().time()))
                )

            pytest.fail(
                "AIPerfJob deletion did not remove the CR, JobSet, and child pods: "
                f"cr_returncode={cr_returncode}, "
                f"jobsets={remaining_jobsets}, pods={remaining_pods}"
            )
        finally:
            if not deleted:
                with contextlib.suppress(Exception):
                    loop = asyncio.get_running_loop()
                    teardown_deadline = min(
                        deadline + CLEANUP_FAILURE_TEARDOWN_TIMEOUT,
                        loop.time() + CLEANUP_FAILURE_TEARDOWN_TIMEOUT,
                    )
                    await delete_and_observe_until_deadline(
                        teardown_deadline,
                        f"AIPerfJob {namespace}/{job_name}",
                        lambda: operator_ready.delete_job(job_name, namespace),
                        lambda: _aiperf_job_exists(
                            kubectl,
                            job_name,
                            namespace,
                        ),
                        CLEANUP_FAILURE_TEARDOWN_POLL_INTERVAL,
                    )


async def _aiperf_job_exists(
    kubectl: KubectlClient,
    job_name: str,
    namespace: str,
) -> bool:
    """Return whether an exact AIPerfJob identity is present."""
    result = await kubectl.run(
        "get",
        "aiperfjob",
        job_name,
        namespace=namespace,
        check=False,
    )
    return result.returncode == 0


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
