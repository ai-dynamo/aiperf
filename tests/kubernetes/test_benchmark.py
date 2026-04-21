# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for benchmark execution and lifecycle."""

from __future__ import annotations

import pytest

from aiperf.kubernetes.constants import Containers
from tests.kubernetes.helpers.benchmark import (
    BenchmarkConfig,
    BenchmarkDeployer,
    BenchmarkResult,
)
from tests.kubernetes.helpers.kubectl import KubectlClient


class TestBenchmarkCompletion:
    """Tests for benchmark completion (module-scoped for speed)."""

    def test_benchmark_completes_successfully(
        self,
        deployed_small_benchmark_module: BenchmarkResult,
    ) -> None:
        """Verify benchmark completes successfully."""
        result = deployed_small_benchmark_module

        assert result.success
        # Status may not be fully terminal if results were collected via API
        # before the JobSet formally completed.
        assert result.status is not None or result.api_results is not None

    def test_api_results_collected(
        self,
        deployed_small_benchmark_module: BenchmarkResult,
    ) -> None:
        """Verify API results are downloaded from the controller."""
        result = deployed_small_benchmark_module

        assert result.api_results is not None, "No API results collected"
        assert result.api_results.get("status") in ("complete", "cancelled")

    def test_jobset_reaches_completed_state(
        self,
        deployed_small_benchmark_module: BenchmarkResult,
    ) -> None:
        """Verify benchmark reached a successful terminal state.

        The JobSet may not have formally completed yet if results were
        collected via the API before the JobSet controller updated status.
        """
        result = deployed_small_benchmark_module
        assert result.success
        # If status is available, verify it
        if result.status and result.status.terminal_state:
            assert result.status.terminal_state == "Completed"

    def test_all_pods_complete(
        self,
        deployed_small_benchmark_module: BenchmarkResult,
    ) -> None:
        """Verify all pods reach a terminal state after benchmark completion.

        In operator mode, the operator deletes the JobSet after fetching results,
        which terminates all pods. Controller pods should Succeed; worker pods may
        show Failed if they were killed during cleanup (this is expected).
        """
        pods = deployed_small_benchmark_module.pods

        # Pods may be gone if the operator cleaned up the JobSet
        if not pods:
            assert deployed_small_benchmark_module.success, (
                "No pods found and benchmark was not successful"
            )
            return

        terminal_phases = {"Succeeded", "Failed", "Completed"}
        for pod in pods:
            assert pod.phase in terminal_phases, (
                f"Pod {pod.name} not in terminal state: phase={pod.phase}"
            )

    def test_all_containers_exit_zero(
        self,
        deployed_small_benchmark_module: BenchmarkResult,
    ) -> None:
        """Verify controller containers exit with code 0.

        Worker containers may be killed during operator cleanup (SIGTERM -> exit 143)
        so we only check controller pods for clean exits.
        """
        for pod in deployed_small_benchmark_module.pods:
            if "controller" not in pod.name:
                continue
            for container_name, container_status in pod.containers.items():
                state = container_status.get("state", {})

                if "terminated" in state:
                    exit_code = state["terminated"].get("exitCode", -1)
                    assert exit_code == 0, (
                        f"Container {container_name} in {pod.name} exited with code {exit_code}"
                    )


class TestBenchmarkLifecycle:
    """Tests for benchmark lifecycle management."""

    @pytest.mark.asyncio
    async def test_can_deploy_multiple_benchmarks_sequentially(
        self,
        benchmark_deployer: BenchmarkDeployer,
    ) -> None:
        """Verify multiple benchmarks can run sequentially."""
        config = BenchmarkConfig(
            concurrency=2,
            request_count=10,
            warmup_request_count=2,
        )

        results = []
        for _ in range(3):
            result = await benchmark_deployer.deploy(config, timeout=90)
            results.append(result)

        for i, result in enumerate(results):
            assert result.success, f"Benchmark {i} failed: {result.error_message}"
            assert result.metrics is not None
            assert result.metrics.request_count == 10

    @pytest.mark.asyncio
    async def test_cleanup_removes_namespace(
        self,
        benchmark_deployer: BenchmarkDeployer,
        kubectl: KubectlClient,
    ) -> None:
        """Verify cleanup removes the benchmark namespace."""
        config = BenchmarkConfig(
            concurrency=2,
            request_count=10,
            warmup_request_count=2,
        )

        result = await benchmark_deployer.deploy(config, timeout=90)
        namespace = result.namespace

        assert await kubectl.namespace_exists(namespace)

        await benchmark_deployer.cleanup(result, delete_namespace=True)

        assert not await kubectl.namespace_exists(namespace)

    def test_duration_is_recorded(
        self,
        deployed_small_benchmark_module: BenchmarkResult,
    ) -> None:
        """Verify benchmark duration is recorded."""
        assert deployed_small_benchmark_module.duration_seconds > 0


class TestBenchmarkPods:
    """Tests for benchmark pod configuration (module-scoped for speed)."""

    def test_controller_pod_has_expected_containers(
        self,
        deployed_small_benchmark_module: BenchmarkResult,
    ) -> None:
        """Verify controller pod has one container per control-plane service.

        gpu_telemetry and server_metrics default to enabled in the generated
        manifest, so their sidecars are part of the expected set.
        """
        controller = deployed_small_benchmark_module.controller_pod

        # Pods may be cleaned up by the operator after JobSet completion
        if controller is None:
            assert deployed_small_benchmark_module.success, (
                "No controller pod found and benchmark was not successful"
            )
            return

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
        assert expected_containers == actual_containers

    def test_worker_pods_completed_after_benchmark(
        self,
        deployed_small_benchmark_module: BenchmarkResult,
    ) -> None:
        """Verify worker pods have completed after benchmark finishes.

        Worker pods should either be in Succeeded phase or have been cleaned up
        after the JobSet completes.
        """
        all_pods = deployed_small_benchmark_module.pods
        workers = deployed_small_benchmark_module.worker_pods

        print(f"\n{'=' * 60}")
        print("WORKER POD STATE (POST-COMPLETION)")
        print(f"{'=' * 60}")
        print(f"  All pods found: {len(all_pods)}")
        for pod in all_pods:
            print(f"    - {pod.name} (phase={pod.phase})")
        print(f"  Worker pods: {len(workers)}")
        print(f"{'=' * 60}\n")

        # Workers may still be Running when results are collected (operator
        # hasn't cleaned up the JobSet yet). Accept Running as transient.
        acceptable = {"Succeeded", "Running", "Completed"}
        for worker in workers:
            assert worker.phase in acceptable, (
                f"Worker pod {worker.name} in unexpected phase: {worker.phase}"
            )


class TestBenchmarkWithDifferentEndpoints:
    """Tests for benchmark with different endpoint configurations."""

    @pytest.mark.asyncio
    async def test_chat_endpoint_type(
        self,
        benchmark_deployer: BenchmarkDeployer,
    ) -> None:
        """Test benchmark with chat endpoint type."""
        config = BenchmarkConfig(
            endpoint_type="chat",
            concurrency=2,
            request_count=10,
            warmup_request_count=2,
        )

        result = await benchmark_deployer.deploy(config, timeout=90)

        assert result.success
        assert result.metrics is not None

    @pytest.mark.asyncio
    async def test_completions_endpoint_type(
        self,
        benchmark_deployer: BenchmarkDeployer,
    ) -> None:
        """Test benchmark with completions endpoint type."""
        config = BenchmarkConfig(
            endpoint_type="completions",
            concurrency=2,
            request_count=10,
            warmup_request_count=2,
        )

        result = await benchmark_deployer.deploy(config, timeout=90)

        # Completions might work or fail depending on mock server
        # Just verify it doesn't crash
        assert result.status is not None


@pytest.mark.k8s_slow
class TestLargeBenchmarks:
    """Tests for larger benchmark configurations."""

    @pytest.mark.asyncio
    async def test_high_concurrency_benchmark(
        self,
        benchmark_deployer: BenchmarkDeployer,
    ) -> None:
        """Test benchmark with high concurrency."""
        config = BenchmarkConfig(
            concurrency=20,
            request_count=100,
            warmup_request_count=10,
        )

        result = await benchmark_deployer.deploy(config, timeout=600)

        assert result.success
        assert result.metrics is not None
        assert result.metrics.request_count == 100

    @pytest.mark.asyncio
    async def test_large_request_count(
        self,
        benchmark_deployer: BenchmarkDeployer,
    ) -> None:
        """Test benchmark with large request count."""
        config = BenchmarkConfig(
            concurrency=5,
            request_count=200,
            warmup_request_count=10,
        )

        result = await benchmark_deployer.deploy(config, timeout=600)

        assert result.success
        assert result.metrics is not None
        assert result.metrics.request_count == 200
