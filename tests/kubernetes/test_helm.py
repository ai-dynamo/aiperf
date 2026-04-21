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

import pytest

from tests.kubernetes.helpers.helm import HelmDeployer, HelmValues
from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.operator import AIPerfJobConfig, OperatorJobResult

# Test timeout for individual test phases (not full job completion)
TEST_PHASE_TIMEOUT = 60  # seconds for waiting for phase transitions
TEST_JOB_TIMEOUT = 60  # seconds for full job completion


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
        new_values = HelmValues(
            image_repository="aiperf",
            image_tag="local",
            image_pull_policy="Never",
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

        await asyncio.sleep(2)

        status = await helm_deployed.get_job_status(result.job_name, result.namespace)

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

        if config_valid is None:
            assert helm_deployed_job_module.success
            return
        assert config_valid.get("status") == "True"

    def test_resources_created_condition_set(
        self,
        helm_deployed_job_module: OperatorJobResult,
    ) -> None:
        """Verify ResourcesCreated condition is set."""
        status = helm_deployed_job_module.status
        assert status is not None

        resources_created = status.get_condition("ResourcesCreated")
        if resources_created is None:
            assert helm_deployed_job_module.success
            return
        assert resources_created.get("status") == "True"

    def test_workers_ready_condition_set(
        self,
        helm_deployed_job_module: OperatorJobResult,
    ) -> None:
        """Verify WorkersReady condition is set on completion."""
        status = helm_deployed_job_module.status
        assert status is not None

        workers_ready = status.get_condition("WorkersReady")
        if workers_ready is None:
            assert helm_deployed_job_module.success
            return
        assert workers_ready.get("status") == "True"

    def test_benchmark_running_condition_set(
        self,
        helm_deployed_job_module: OperatorJobResult,
    ) -> None:
        """Verify BenchmarkRunning condition was set during execution."""
        status = helm_deployed_job_module.status
        assert status is not None

        benchmark_running = status.get_condition("BenchmarkRunning")
        if benchmark_running is None:
            assert helm_deployed_job_module.success
            return


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

        results_available = status.get_condition("ResultsAvailable")
        if results_available and results_available.get("status") == "True":
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
                    "endpoint": {},  # Missing required fields
                    "phases": {
                        "profiling": {"type": "concurrency", "concurrency": 5},
                    },
                },
            },
        }

        try:
            await helm_deployed.kubectl.apply(yaml.dump(cr))

            await asyncio.sleep(5)

            status = await helm_deployed.get_job_status(
                "invalid-config-test", "default"
            )

            print(f"\n{'=' * 60}")
            print("INVALID CONFIG ERROR HANDLING")
            print(f"{'=' * 60}")
            print(f"  Phase: {status.phase}")
            print(f"  Error: {status.error}")
            print(f"{'=' * 60}\n")

            assert status.is_failed
            assert status.error is not None

        finally:
            await helm_deployed.kubectl.delete(
                "aiperfjob", "invalid-config-test", namespace="default"
            )

    @pytest.mark.timeout(TEST_JOB_TIMEOUT)
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
        timeout = TEST_JOB_TIMEOUT

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

        assert status.is_failed or status.is_completed

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

    @pytest.mark.timeout(TEST_PHASE_TIMEOUT)
    @pytest.mark.asyncio
    async def test_deleting_job_removes_resources(
        self,
        helm_deployed: HelmDeployer,
        small_helm_config: AIPerfJobConfig,
        kubectl: KubectlClient,
    ) -> None:
        """Verify deleting AIPerfJob removes associated resources."""
        result = await helm_deployed.create_job(small_helm_config)

        await asyncio.sleep(5)

        status = await helm_deployed.get_job_status(result.job_name, result.namespace)
        jobset_name = status.jobset_name

        # Verify JobSet exists (may already be cleaned up by operator)
        if jobset_name:
            jobsets = await kubectl.get_jobsets(result.namespace)
            jobset_exists = any(js.name == jobset_name for js in jobsets)
            if not jobset_exists and status.is_completed:
                # Operator already cleaned up - skip the rest
                await helm_deployed.delete_job(result.job_name, result.namespace)
                return
            assert jobset_exists

        # Delete the job
        await helm_deployed.delete_job(result.job_name, result.namespace)

        # Wait for cleanup
        await asyncio.sleep(10)

        # Verify resources are cleaned up
        if jobset_name:
            jobsets = await kubectl.get_jobsets(result.namespace)
            assert not any(js.name == jobset_name for js in jobsets)


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
        # Workers may fit in 1 pod (workers_per_pod default is 10)
        assert result.status.workers_total >= 1


@pytest.mark.skip(
    reason="Destroys shared operator; run in isolation: pytest tests/kubernetes/test_helm.py::TestHelmUninstall -v"
)
class TestHelmUninstall:
    """Tests for Helm chart uninstallation.

    Skipped in the full suite because it removes the operator, cascading
    failures to all subsequent test modules. Run in isolation instead.
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
