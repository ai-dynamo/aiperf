# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for Helm-based AIPerf operator deployment.

These tests deploy the operator using Helm on a minikube cluster, create AIPerfJob CRs,
and verify the full benchmark lifecycle through the operator.

Fixture scoping strategy:
- Module-scoped: local_cluster, kubectl, helm_deployer (shared across all tests)
- Function-scoped: Used only when test modifies state or needs fresh resources
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from dataclasses import replace

import pytest

from tests.kubernetes.helpers.deadline import (
    await_before_deadline,
    delete_and_observe_until_deadline,
)
from tests.kubernetes.helpers.helm import HelmDeployer
from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.operator import AIPerfJobConfig, OperatorJobResult

# Test timeout for individual test phases (not full job completion)
TEST_PHASE_TIMEOUT = 60  # seconds for waiting for phase transitions
TEST_JOB_TIMEOUT = 60  # seconds for full job completion
TEST_CLEANUP_TIMEOUT = 150  # seconds for CR deletion propagation checks
CLEANUP_ASSERTION_TIMEOUT = 120  # seconds shared by observation and deletion
CLEANUP_DELETION_POLL_RESERVE = 60  # seconds reserved from the shared deadline
CLEANUP_FAILURE_TEARDOWN_TIMEOUT = 20  # seconds from pytest's reporting buffer
CLEANUP_FAILURE_TEARDOWN_POLL_INTERVAL = 1  # seconds between absence checks


class TestHelmChartDeployment:
    """Tests for Helm chart installation and configuration."""

    @pytest.mark.asyncio
    async def test_chart_installs_successfully(
        self,
        helm_deployed: HelmDeployer,
    ) -> None:
        """Verify Helm chart installs without errors."""
        status = await helm_deployed.get_release_status()
        assert status == "deployed"

    @pytest.mark.asyncio
    async def test_crd_is_established(
        self,
        helm_deployed: HelmDeployer,
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
        helm_deployed: HelmDeployer,
        kubectl: KubectlClient,
    ) -> None:
        """Verify operator pod is running."""
        pods = await kubectl.get_pods(helm_deployed.OPERATOR_NAMESPACE)
        operator_pods = [
            p for p in pods if "aiperf-operator" in p.name and "-test-" not in p.name
        ]

        assert len(operator_pods) == 1
        assert operator_pods[0].phase == "Running"

    @pytest.mark.asyncio
    async def test_service_account_created(
        self,
        helm_deployed: HelmDeployer,
        kubectl: KubectlClient,
    ) -> None:
        """Verify service account is created by Helm."""
        result = await kubectl.run(
            "get",
            "serviceaccount",
            "-n",
            helm_deployed.OPERATOR_NAMESPACE,
            "-o",
            "jsonpath={.items[*].metadata.name}",
        )
        sa_names = result.stdout.strip().split()
        assert any("aiperf" in name for name in sa_names)

    @pytest.mark.asyncio
    async def test_cluster_role_created(
        self,
        helm_deployed: HelmDeployer,
        kubectl: KubectlClient,
    ) -> None:
        """Verify ClusterRole is created with correct permissions."""
        result = await kubectl.run(
            "get",
            "clusterrole",
            "-o",
            "jsonpath={.items[*].metadata.name}",
        )
        role_names = result.stdout.strip().split()
        assert any("aiperf" in name for name in role_names)

    @pytest.mark.asyncio
    async def test_operator_has_correct_permissions(
        self,
        helm_deployed: HelmDeployer,
        kubectl: KubectlClient,
    ) -> None:
        """Verify operator has necessary RBAC permissions."""
        # Get the service account name from the deployment
        result = await kubectl.run(
            "get",
            "deployment",
            "-n",
            helm_deployed.OPERATOR_NAMESPACE,
            "-o",
            "jsonpath={.items[0].spec.template.spec.serviceAccountName}",
        )
        sa_name = result.stdout.strip()

        # Check permission to create JobSets
        result = await kubectl.run(
            "auth",
            "can-i",
            "create",
            "jobsets.jobset.x-k8s.io",
            f"--as=system:serviceaccount:{helm_deployed.OPERATOR_NAMESPACE}:{sa_name}",
            check=False,
        )
        assert result.stdout.strip() == "yes"


class TestHelmChartUpgrade:
    """Tests for Helm chart upgrade functionality."""

    @pytest.mark.asyncio
    async def test_upgrade_with_new_values(
        self,
        helm_deployed: HelmDeployer,
        kubectl: KubectlClient,
    ) -> None:
        """Verify chart can be upgraded with new values."""
        # Upgrade with different resource limits
        new_values = replace(
            helm_deployed.values,
            resources_limits_memory="768Mi",
        )

        await helm_deployed.upgrade_chart(values=new_values)

        # Verify deployment updated
        status = await helm_deployed.get_release_status()
        assert status == "deployed"

        # Verify new memory limit is applied
        result = await kubectl.run(
            "get",
            "deployment",
            helm_deployed.RELEASE_NAME,
            "-n",
            helm_deployed.OPERATOR_NAMESPACE,
            "-o",
            "jsonpath={.spec.template.spec.containers[0].resources.limits.memory}",
        )
        assert result.stdout.strip() == "768Mi"


class TestHelmJobLifecycle:
    """Tests for AIPerfJob lifecycle management with Helm-deployed operator."""

    @pytest.mark.timeout(TEST_PHASE_TIMEOUT)
    @pytest.mark.asyncio
    async def test_create_job_sets_pending_phase(
        self,
        helm_deployed: HelmDeployer,
        small_helm_config: AIPerfJobConfig,
    ) -> None:
        """Verify newly created job starts in Pending phase."""
        result = await helm_deployed.create_job(small_helm_config)

        # Poll for the operator to set phase (up to ~15s)
        status = await helm_deployed.get_job_status(result.job_name, result.namespace)
        for _ in range(15):
            if status.phase in ("Pending", "Initializing", "Running"):
                break
            await asyncio.sleep(1)
            status = await helm_deployed.get_job_status(
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
        await helm_deployed.delete_job(result.job_name, result.namespace)

    @pytest.mark.timeout(TEST_JOB_TIMEOUT)
    @pytest.mark.asyncio
    async def test_job_transitions_through_phases(
        self,
        helm_deployed: HelmDeployer,
        small_helm_config: AIPerfJobConfig,
    ) -> None:
        """Verify job transitions through expected phases."""
        result = await helm_deployed.create_job(small_helm_config)
        phases_seen = set()

        loop = asyncio.get_event_loop()
        start = loop.time()
        timeout = TEST_JOB_TIMEOUT

        while loop.time() - start < timeout:
            status = await helm_deployed.get_job_status(
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

        assert len(phases_seen) >= 1
        assert status.is_completed, f"Expected Completed, got {status.phase}"

        # Cleanup
        await helm_deployed.delete_job(result.job_name, result.namespace)

    def test_job_completes_successfully(
        self,
        helm_deployed_job_module: OperatorJobResult,
    ) -> None:
        """Verify job completes successfully with results."""
        result = helm_deployed_job_module

        print(f"\n{'=' * 70}")
        print("HELM OPERATOR JOB COMPLETION RESULTS")
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

        print(f"{'=' * 70}\n")

        assert result.success
        assert result.status is not None
        assert result.status.is_completed

    def test_job_creates_jobset(
        self,
        helm_deployed_job_module: OperatorJobResult,
    ) -> None:
        """Verify operator creates JobSet for the benchmark.

        The operator may clean up the JobSet after collecting results,
        so jobset_status may be None on a successful run.
        """
        assert helm_deployed_job_module.status is not None
        if helm_deployed_job_module.jobset_status is None:
            assert helm_deployed_job_module.success

    def test_job_tracks_worker_status(
        self,
        helm_deployed_job_module: OperatorJobResult,
    ) -> None:
        """Verify operator tracks worker readiness."""
        status = helm_deployed_job_module.status
        if status is None or status.workers_total == 0:
            assert helm_deployed_job_module.success
            return
        assert status.workers_total >= 1


class TestHelmConditions:
    """Tests for operator condition tracking with Helm deployment."""

    def test_config_valid_condition_set(
        self,
        helm_deployed_job_module: OperatorJobResult,
    ) -> None:
        """Verify ConfigValid condition is set."""
        status = helm_deployed_job_module.status
        assert status is not None

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
        helm_deployed_job_module: OperatorJobResult,
    ) -> None:
        """Verify ResourcesCreated condition is set."""
        status = helm_deployed_job_module.status
        assert status is not None

        resources_created = status.get_condition("ResourcesCreated")
        assert resources_created is not None, status.conditions
        assert resources_created.get("status") == "True", status.conditions

    def test_workers_ready_condition_set(
        self,
        helm_deployed_job_module: OperatorJobResult,
    ) -> None:
        """Verify WorkersReady condition is set on completion."""
        status = helm_deployed_job_module.status
        assert status is not None

        workers_ready = status.get_condition("WorkersReady")
        assert workers_ready is not None, status.conditions
        assert workers_ready.get("status") == "True", status.conditions

    def test_benchmark_running_condition_set(
        self,
        helm_deployed_job_module: OperatorJobResult,
    ) -> None:
        """Verify BenchmarkRunning condition was set during execution."""
        status = helm_deployed_job_module.status
        assert status is not None

        benchmark_running = status.get_condition("BenchmarkRunning")
        assert benchmark_running is not None, status.conditions
        assert benchmark_running.get("status") == "True", status.conditions


class TestHelmResults:
    """Tests for operator results collection with Helm deployment."""

    def test_results_available_on_completion(
        self,
        helm_deployed_job_module: OperatorJobResult,
    ) -> None:
        """Verify results are available after job completion."""
        status = helm_deployed_job_module.status
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


class TestHelmErrorHandling:
    """Tests for operator error handling with Helm deployment."""

    @pytest.mark.timeout(TEST_PHASE_TIMEOUT)
    @pytest.mark.asyncio
    async def test_invalid_config_fails_with_error(
        self,
        helm_deployed: HelmDeployer,
    ) -> None:
        """Verify invalid config results in failure with error message."""
        import yaml

        cr = {
            "apiVersion": "aiperf.nvidia.com/v1alpha1",
            "kind": "AIPerfJob",
            "metadata": {
                "name": "invalid-config-test",
                "namespace": "default",
            },
            "spec": {
                "image": "aiperf:local",
                "imagePullPolicy": "Never",
                "benchmark": {
                    "endpoint": {
                        "urls": [
                            "http://aiperf-mock-server.default.svc.cluster.local:8000/v1"
                        ]
                    },
                    "models": {"items": [{"name": "mock-model"}]},
                    "datasets": [{"type": "synthetic"}],
                    "phases": [
                        {"name": "profiling", "type": "concurrency", "concurrency": 5},
                    ],
                },
            },
        }

        try:
            await helm_deployed.kubectl.apply(yaml.dump(cr))

            loop = asyncio.get_event_loop()
            deadline = loop.time() + TEST_PHASE_TIMEOUT
            while True:
                status = await helm_deployed.get_job_status(
                    "invalid-config-test", "default"
                )
                if status.is_terminal or loop.time() >= deadline:
                    break
                await asyncio.sleep(1)

            print(f"\n{'=' * 60}")
            print("INVALID CONFIG ERROR HANDLING")
            print(f"{'=' * 60}")
            print(f"  Phase: {status.phase}")
            print(f"  Error: {status.error}")
            print(f"{'=' * 60}\n")

            assert status.is_failed, (
                "invalid config did not reach Failed before timeout; "
                f"status={status.raw_status!r}"
            )
            assert status.error is not None

        finally:
            await helm_deployed.kubectl.delete(
                "aiperfjob", "invalid-config-test", namespace="default"
            )

    # The preflight's endpoint reachability check has its own retry/backoff
    # ladder that can push this past TEST_JOB_TIMEOUT (60s); give it room.
    @pytest.mark.timeout(180)
    @pytest.mark.asyncio
    async def test_unreachable_endpoint_fails_gracefully(
        self,
        helm_deployed: HelmDeployer,
    ) -> None:
        """Verify unreachable endpoint is handled gracefully."""
        config = AIPerfJobConfig(
            endpoint_url="http://nonexistent-service:8000/v1",
            concurrency=2,
            request_count=5,
        )

        result = await helm_deployed.create_job(
            config, name="unreachable-endpoint-test"
        )

        loop = asyncio.get_event_loop()
        start = loop.time()
        timeout = 150

        while loop.time() - start < timeout:
            status = await helm_deployed.get_job_status(
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
        print(f"{'=' * 60}\n")

        assert status.is_failed, (
            f"Unreachable endpoint unexpectedly ended in phase={status.phase}"
        )
        endpoint_cond = status.get_condition("EndpointReachable")
        assert endpoint_cond is not None
        assert endpoint_cond.get("status") == "False"

        # Cleanup
        await helm_deployed.delete_job(result.job_name, result.namespace)


class TestHelmEvents:
    """Tests for Kubernetes events emitted by Helm-deployed operator."""

    @pytest.mark.asyncio
    async def test_events_emitted_for_job(
        self,
        helm_deployed_job_module: OperatorJobResult,
        kubectl: KubectlClient,
    ) -> None:
        """Verify operator emits events for job lifecycle."""
        events = await kubectl.get_events(helm_deployed_job_module.namespace)

        print(f"\n{'=' * 60}")
        print("OPERATOR EVENTS")
        print(f"{'=' * 60}")
        print(events)
        print(f"{'=' * 60}\n")

        assert len(events) > 0


class TestHelmCleanup:
    """Tests for operator resource cleanup with Helm deployment."""

    @pytest.mark.timeout(TEST_CLEANUP_TIMEOUT)
    @pytest.mark.asyncio
    async def test_deleting_job_removes_resources(
        self,
        helm_deployed: HelmDeployer,
        small_helm_config: AIPerfJobConfig,
        kubectl: KubectlClient,
    ) -> None:
        """Verify deleting AIPerfJob removes associated resources."""
        deadline = asyncio.get_running_loop().time() + CLEANUP_ASSERTION_TIMEOUT
        config = replace(
            small_helm_config,
            request_count=None,
            benchmark_duration=120.0,
        )
        job_name = f"cleanup-{uuid.uuid4().hex[:8]}"
        namespace = helm_deployed.default_job_namespace
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
                lambda: helm_deployed.create_job(
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
                    lambda: helm_deployed.get_job_status(job_name, namespace),
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
                lambda: helm_deployed.delete_job(job_name, namespace),
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
                        lambda: helm_deployed.delete_job(job_name, namespace),
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
class TestHelmScaling:
    """Tests for Helm-deployed operator with different scaling configurations."""

    @pytest.mark.timeout(600)
    @pytest.mark.asyncio
    async def test_high_concurrency_job(
        self,
        helm_deployed: HelmDeployer,
    ) -> None:
        """Test operator handles high concurrency job."""
        config = AIPerfJobConfig(
            concurrency=10,
            request_count=20,
            warmup_request_count=2,
        )

        result = await helm_deployed.run_job(config, timeout=180)

        assert result.success
        assert result.status is not None
        assert result.status.is_completed

    @pytest.mark.timeout(600)
    @pytest.mark.asyncio
    async def test_multiple_workers_job(
        self,
        helm_deployed: HelmDeployer,
    ) -> None:
        """Test operator handles job requiring multiple workers."""
        config = AIPerfJobConfig(
            concurrency=20,
            request_count=40,
            warmup_request_count=5,
            connections_per_worker=10,
        )

        result = await helm_deployed.run_job(config, timeout=600)

        assert result.success
        assert result.status is not None
        # Benchmark succeeded (result.success). The operator's
        # workers_total counter is observational — with the mock server the
        # benchmark can complete faster than the status monitor polls, and
        # CompletedBeforeMonitor is set. Accept either a mid-flight
        # observation or fast-completion verified by Completed phase + stored
        # results.
        if result.status.workers_total < 1:
            assert result.status.phase == "Completed"
            assert any(
                c.get("type") == "ResultsAvailable" and c.get("status") == "True"
                for c in (result.status.conditions or [])
            )


class TestHelmUninstall:
    """Tests for Helm chart uninstallation.

    This class stays last because it removes the module-scoped Helm release.
    The suite's primary operator uses a separate namespace and is restored by
    the Helm fixture before later modules run.
    """

    @pytest.mark.timeout(120)
    @pytest.mark.asyncio
    async def test_uninstall_removes_operator(
        self,
        helm_deployed: HelmDeployer,
        kubectl: KubectlClient,
    ) -> None:
        """Verify Helm uninstall removes operator pods."""
        await helm_deployed.uninstall_chart(wait=False)

        for _ in range(30):
            pods = await kubectl.get_pods(helm_deployed.OPERATOR_NAMESPACE)
            operator_pods = [p for p in pods if "aiperf-operator" in p.name]
            if not operator_pods:
                break
            await asyncio.sleep(2)

        pods = await kubectl.get_pods(helm_deployed.OPERATOR_NAMESPACE)
        operator_pods = [p for p in pods if "aiperf-operator" in p.name]
        assert len(operator_pods) == 0, (
            f"Expected no operator pods after uninstall, found: {operator_pods}"
        )
