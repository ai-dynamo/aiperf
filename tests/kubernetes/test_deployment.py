# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for basic Kubernetes deployment functionality."""

from __future__ import annotations

import asyncio

import pytest

from aiperf.kubernetes.constants import Containers
from tests.kubernetes.helpers.benchmark import (
    BenchmarkConfig,
    BenchmarkDeployer,
)
from tests.kubernetes.helpers.cluster import LocalCluster
from tests.kubernetes.helpers.kubectl import KubectlClient


class TestClusterSetup:
    """Tests for cluster setup and prerequisites."""

    @pytest.mark.asyncio
    async def test_cluster_exists(self, local_cluster: LocalCluster) -> None:
        """Verify the local cluster is running."""
        assert await local_cluster.exists()
        assert len(await local_cluster.get_nodes()) > 0

    @pytest.mark.asyncio
    async def test_kubectl_connection(self, kubectl: KubectlClient) -> None:
        """Verify kubectl can connect to the cluster."""
        result = await kubectl.run("cluster-info", check=False)
        assert result.returncode == 0
        assert "Kubernetes control plane" in result.stdout

    @pytest.mark.asyncio
    async def test_jobset_crd_installed(
        self, kubectl: KubectlClient, jobset_controller: None
    ) -> None:
        """Verify JobSet CRD is installed."""
        result = await kubectl.run("get", "crd", "jobsets.jobset.x-k8s.io", check=False)
        assert result.returncode == 0

    @pytest.mark.asyncio
    async def test_mock_server_running(
        self, kubectl: KubectlClient, mock_server: None
    ) -> None:
        """Verify mock server is running."""
        pods = await kubectl.get_pods("default")
        mock_pods = [p for p in pods if "mock-server" in p.name]

        assert len(mock_pods) > 0
        assert all(p.is_ready for p in mock_pods)


class TestBenchmarkDeployment:
    """Tests for benchmark deployment."""

    @pytest.mark.asyncio
    async def test_deploy_creates_namespace(
        self,
        benchmark_deployer: BenchmarkDeployer,
        small_benchmark_config: BenchmarkConfig,
        kubectl: KubectlClient,
    ) -> None:
        """Verify deployment creates a new namespace."""
        result = await benchmark_deployer.deploy(
            config=small_benchmark_config,
            wait_for_completion=False,  # Don't wait, just check deployment
            timeout=60,
        )

        print(f"\n{'=' * 60}")
        print("NAMESPACE VERIFICATION RESULTS")
        print(f"{'=' * 60}")
        print(f"  Namespace: {result.namespace}")
        print(f"  Starts with 'aiperf-': {result.namespace.startswith('aiperf-')}")
        print(
            f"  Exists in cluster: {await kubectl.namespace_exists(result.namespace)}"
        )
        print("  ✓ Namespace created successfully!")
        print(f"{'=' * 60}\n")

        assert result.namespace is not None
        assert result.namespace.startswith("aiperf-")
        assert await kubectl.namespace_exists(result.namespace)

        # Cleanup
        await benchmark_deployer.cleanup(result)

    @pytest.mark.asyncio
    async def test_deploy_creates_jobset(
        self,
        benchmark_deployer: BenchmarkDeployer,
        small_benchmark_config: BenchmarkConfig,
        kubectl: KubectlClient,
    ) -> None:
        """Verify deployment creates a JobSet."""
        result = await benchmark_deployer.deploy(
            config=small_benchmark_config,
            wait_for_completion=False,
            timeout=60,
        )

        jobsets = await kubectl.get_jobsets(result.namespace)

        print(f"\n{'=' * 60}")
        print("JOBSET VERIFICATION RESULTS")
        print(f"{'=' * 60}")
        print(f"  Namespace: {result.namespace}")
        print(f"  JobSets found: {len(jobsets)}")
        if jobsets:
            print(f"  JobSet name: {jobsets[0].name}")
            print(f"  Expected name: {result.jobset_name}")
            print(f"  Names match: {jobsets[0].name == result.jobset_name}")
        print("  ✓ JobSet created successfully!")
        print(f"{'=' * 60}\n")

        assert len(jobsets) == 1
        assert jobsets[0].name == result.jobset_name

        # Cleanup
        await benchmark_deployer.cleanup(result)

    @pytest.mark.asyncio
    async def test_deploy_creates_controller_pod(
        self,
        benchmark_deployer: BenchmarkDeployer,
        small_benchmark_config: BenchmarkConfig,
        kubectl: KubectlClient,
    ) -> None:
        """Verify deployment creates controller pod with the expected service containers.

        The controller pod runs one container per control-plane service plus a
        results sidecar. GPU telemetry and server metrics sidecars are included
        because both default to enabled in the generated manifest.
        """
        result = await benchmark_deployer.deploy(
            config=small_benchmark_config,
            wait_for_completion=False,
            timeout=60,
        )

        # Wait for pods to be created
        await asyncio.sleep(10)

        pods = await kubectl.get_pods(result.namespace)
        controller_pods = [p for p in pods if "controller" in p.name]

        print(f"\n{'=' * 60}")
        print("CONTROLLER POD VERIFICATION RESULTS")
        print(f"{'=' * 60}")
        print(f"  Namespace: {result.namespace}")
        print(f"  JobSet: {result.jobset_name}")
        print(f"  Controller pods found: {len(controller_pods)}")

        assert len(controller_pods) == 1

        controller = controller_pods[0]
        # Multi-container controller pod: one container per control-plane
        # service plus the results sidecar. gpu_telemetry and server_metrics
        # default to enabled in the generated manifest, so include them.
        expected_containers = {
            Containers.CONTROL_PLANE,
            Containers.DATASET_MANAGER,
            Containers.TIMING_MANAGER,
            Containers.RECORDS_MANAGER,
            Containers.API,
            Containers.GPU_TELEMETRY_MANAGER,
            Containers.SERVER_METRICS_MANAGER,
            Containers.RESULTS_SIDECAR,
        }

        actual_containers = set(controller.containers.keys())

        print(f"  Pod name: {controller.name}")
        print(f"  Pod phase: {controller.phase}")
        print(f"  Containers: {sorted(actual_containers)}")
        print(f"  Expected: {sorted(expected_containers)}")
        print(f"{'=' * 60}\n")

        assert expected_containers == actual_containers, (
            f"Expected controller containers {sorted(expected_containers)}, "
            f"got: {sorted(actual_containers)}"
        )

        # Cleanup
        await benchmark_deployer.cleanup(result)

    @pytest.mark.asyncio
    async def test_deploy_creates_worker_pod(
        self,
        benchmark_deployer: BenchmarkDeployer,
        small_benchmark_config: BenchmarkConfig,
        kubectl: KubectlClient,
    ) -> None:
        """Verify deployment creates worker pod with per-service containers.

        The worker pod keeps a worker-group-manager container for shared pod
        infrastructure, while workers and record processors run in their own
        sibling containers.
        """
        result = await benchmark_deployer.deploy(
            config=small_benchmark_config,
            wait_for_completion=False,
            timeout=60,
        )

        # Wait for pods to be created
        await asyncio.sleep(10)

        pods = await kubectl.get_pods(result.namespace)
        worker_pods = [
            p for p in pods if "worker" in p.name and "controller" not in p.name
        ]

        print(f"\n{'=' * 60}")
        print("WORKER POD VERIFICATION RESULTS")
        print(f"{'=' * 60}")
        print(f"  Namespace: {result.namespace}")
        print(f"  JobSet: {result.jobset_name}")
        print(f"  Worker pods found: {len(worker_pods)}")

        assert len(worker_pods) >= 1

        worker = worker_pods[0]
        actual_containers = set(worker.containers.keys())
        worker_containers = {
            name for name in actual_containers if name.startswith("worker-")
        }
        record_processor_containers = {
            name for name in actual_containers if name.startswith("record-processor-")
        }

        print(f"  Pod name: {worker.name}")
        print(f"  Pod phase: {worker.phase}")
        print(f"  Containers: {sorted(actual_containers)}")
        print(
            "  Expected: worker-group-manager + worker-* + record-processor-* containers"
        )
        print("  ✓ Multi-container worker pod verified!")
        print(f"{'=' * 60}\n")

        assert "worker-group-manager" in actual_containers
        assert worker_containers, (
            f"Expected at least one worker-* container, got: {actual_containers}"
        )
        assert record_processor_containers, (
            f"Expected at least one record-processor-* container, got: {actual_containers}"
        )

        # Cleanup
        await benchmark_deployer.cleanup(result)


class TestConfigVariations:
    """Tests for different benchmark configurations."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("concurrency", [1, 2, 5, 10])
    async def test_different_concurrency_levels(
        self,
        benchmark_deployer: BenchmarkDeployer,
        concurrency: int,
    ) -> None:
        """Test benchmarks with different concurrency levels."""
        config = BenchmarkConfig(
            concurrency=concurrency,
            request_count=10,  # Small for speed
            warmup_request_count=2,
        )

        result = await benchmark_deployer.deploy(config, timeout=90)

        assert result.success, (
            f"Benchmark failed with concurrency={concurrency}: {result.error_message}"
        )
        assert result.metrics is not None
        assert result.metrics.request_count == 10

    @pytest.mark.asyncio
    @pytest.mark.parametrize("request_count", [10, 25, 50])
    async def test_different_request_counts(
        self,
        benchmark_deployer: BenchmarkDeployer,
        request_count: int,
    ) -> None:
        """Test benchmarks with different request counts."""
        config = BenchmarkConfig(
            concurrency=3,
            request_count=request_count,
            warmup_request_count=2,
        )

        result = await benchmark_deployer.deploy(config, timeout=90)

        assert result.success
        assert result.metrics is not None
        assert result.metrics.request_count == request_count

    @pytest.mark.asyncio
    async def test_custom_sequence_lengths(
        self,
        benchmark_deployer: BenchmarkDeployer,
    ) -> None:
        """Test benchmark with custom input/output sequence lengths."""
        config = BenchmarkConfig(
            concurrency=2,
            request_count=10,
            warmup_request_count=2,
            input_sequence_min=100,
            input_sequence_max=200,
            output_tokens_min=20,
            output_tokens_max=100,
        )

        result = await benchmark_deployer.deploy(config, timeout=90)

        assert result.success
        assert result.metrics is not None
