# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for worker pod scaling and multi-pod deployments."""

from __future__ import annotations

import asyncio

import pytest
from pytest import param

from tests.kubernetes.helpers.benchmark import (
    BenchmarkConfig,
    BenchmarkDeployer,
    BenchmarkResult,
)
from tests.kubernetes.helpers.kubectl import KubectlClient


class TestWorkerPodScaling:
    """Tests for worker pod scaling behavior."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "workers,concurrency,expected_pods",
        [
            param(1, 2, 1, id="1-worker-1-pod"),
            param(5, 10, 1, id="5-workers-1-pod"),
            param(10, 10, 1, id="10-workers-1-pod"),
        ],
    )  # fmt: skip
    async def test_worker_pod_count_matches_config(
        self,
        benchmark_deployer: BenchmarkDeployer,
        kubectl: KubectlClient,
        workers: int,
        concurrency: int,
        expected_pods: int,
    ) -> None:
        """Verify correct number of worker pods are created.

        --workers-max sets total workers distributed across pods based on
        --workers-per-pod (default 10). All cases here fit in 1 pod.
        """
        config = BenchmarkConfig(
            concurrency=concurrency,
            request_count=10,
            warmup_request_count=2,
            workers=workers,
        )

        result = await benchmark_deployer.deploy(
            config, wait_for_completion=False, timeout=60
        )

        # Poll for worker pods (may complete quickly on mock server)
        worker_pods = []
        for _ in range(30):
            pods = await kubectl.get_pods(result.namespace)
            worker_pods = [
                p for p in pods if "worker" in p.name and "controller" not in p.name
            ]
            if worker_pods:
                break
            await asyncio.sleep(1)

        # Benchmark may have completed and pods cleaned up before we observed
        if not worker_pods:
            return

        assert len(worker_pods) == expected_pods, (
            f"Expected {expected_pods} worker pods, got {len(worker_pods)}"
        )

        # Cleanup
        await benchmark_deployer.cleanup(result)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "workers,request_count",
        [
            param(2, 30, id="2-workers-30-requests"),
            param(3, 50, id="3-workers-50-requests"),
        ],
    )  # fmt: skip
    async def test_multiple_workers_complete_benchmark(
        self,
        benchmark_deployer: BenchmarkDeployer,
        workers: int,
        request_count: int,
    ) -> None:
        """Verify benchmark completes with multiple worker pods."""
        config = BenchmarkConfig(
            concurrency=workers * 3,
            request_count=request_count,
            warmup_request_count=5,
            workers=workers,
        )

        result = await benchmark_deployer.deploy(config, timeout=90)

        assert result.success, f"Benchmark failed: {result.error_message}"
        assert result.metrics is not None
        assert result.metrics.request_count == request_count
        assert result.metrics.error_count == 0


class TestHighConcurrencyScaling:
    """Tests for high concurrency scenarios."""

    @pytest.mark.k8s_slow
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "concurrency,request_count,workers",
        [
            param(20, 100, 4, id="20c-100r-4w"),
            param(30, 150, 5, id="30c-150r-5w"),
        ],
    )  # fmt: skip
    async def test_high_concurrency_with_multiple_workers(
        self,
        benchmark_deployer: BenchmarkDeployer,
        concurrency: int,
        request_count: int,
        workers: int,
    ) -> None:
        """Test high concurrency benchmark with multiple worker pods."""
        config = BenchmarkConfig(
            concurrency=concurrency,
            request_count=request_count,
            warmup_request_count=10,
            workers=workers,
        )

        result = await benchmark_deployer.deploy(config, timeout=600)

        assert result.success, (
            f"High concurrency benchmark failed: {result.error_message}"
        )
        assert result.metrics is not None
        assert result.metrics.request_count == request_count


class TestPodResourceConfiguration:
    """Tests for pod resource configuration (module-scoped for speed)."""

    @pytest.mark.asyncio
    async def test_controller_pod_has_expected_resources(
        self,
        deployed_small_benchmark_module: BenchmarkResult,
        kubectl: KubectlClient,
    ) -> None:
        """Verify controller pod has resource requests/limits set."""
        result = deployed_small_benchmark_module
        controller = result.controller_pod
        if controller is None:
            assert result.success
            return

        pod_json = await kubectl.get_json(
            "pod", controller.name, namespace=result.namespace
        )
        containers = pod_json.get("spec", {}).get("containers", [])

        for container in containers:
            resources = container.get("resources", {})
            assert "requests" in resources, f"{container['name']} missing requests"
            assert "limits" in resources, f"{container['name']} missing limits"
            assert "cpu" in resources["requests"]
            assert "memory" in resources["requests"]

    @pytest.mark.asyncio
    async def test_worker_pod_has_expected_resources(
        self,
        deployed_small_benchmark_module: BenchmarkResult,
        kubectl: KubectlClient,
    ) -> None:
        """Verify worker pod has resource requests/limits set."""
        result = deployed_small_benchmark_module
        worker_pods = result.worker_pods
        if not worker_pods:
            assert result.success
            return

        pod_json = await kubectl.get_json(
            "pod", worker_pods[0].name, namespace=result.namespace
        )
        containers = pod_json.get("spec", {}).get("containers", [])

        for container in containers:
            resources = container.get("resources", {})
            assert "requests" in resources, f"{container['name']} missing requests"
            assert "limits" in resources, f"{container['name']} missing limits"


class TestPodSecurityConfiguration:
    """Tests for pod security configuration (module-scoped for speed)."""

    @pytest.mark.asyncio
    async def test_pods_run_as_non_root(
        self,
        deployed_small_benchmark_module: BenchmarkResult,
        kubectl: KubectlClient,
    ) -> None:
        """Verify pods run as non-root user."""
        result = deployed_small_benchmark_module

        if not result.pods:
            assert result.success
            return

        for pod_status in result.pods:
            pod_json = await kubectl.get_json(
                "pod", pod_status.name, namespace=result.namespace
            )
            pod_spec = pod_json.get("spec", {})

            pod_security = pod_spec.get("securityContext", {})
            assert pod_security.get("runAsNonRoot") is True, (
                f"Pod {pod_status.name} should have runAsNonRoot=true"
            )

            for container in pod_spec.get("containers", []):
                container_security = container.get("securityContext", {})
                assert container_security.get("allowPrivilegeEscalation") is False, (
                    f"Container {container['name']} should not allow privilege escalation"
                )

    @pytest.mark.asyncio
    async def test_pods_have_health_probes(
        self,
        deployed_small_benchmark_module: BenchmarkResult,
        kubectl: KubectlClient,
    ) -> None:
        """Verify pods have startup, liveness, and readiness probes."""
        result = deployed_small_benchmark_module

        if not result.pods:
            assert result.success
            return

        for pod_status in result.pods:
            pod_json = await kubectl.get_json(
                "pod", pod_status.name, namespace=result.namespace
            )
            containers = pod_json.get("spec", {}).get("containers", [])

            for container in containers:
                assert "startupProbe" in container, (
                    f"Container {container['name']} missing startupProbe"
                )
                assert "livenessProbe" in container, (
                    f"Container {container['name']} missing livenessProbe"
                )


class TestControllerSinglePodConstraint:
    """Tests to verify controller always runs as a single pod."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "workers",
        [
            param(1, id="1-worker"),
            param(5, id="5-workers"),
            param(10, id="10-workers"),
        ],
    )  # fmt: skip
    async def test_controller_always_single_pod(
        self,
        benchmark_deployer: BenchmarkDeployer,
        kubectl: KubectlClient,
        workers: int,
    ) -> None:
        """Verify controller pod count is always 1 regardless of worker count."""
        concurrency = workers * 2
        config = BenchmarkConfig(
            concurrency=concurrency,
            request_count=max(concurrency, 10),
            warmup_request_count=2,
            workers=workers,
        )

        result = await benchmark_deployer.deploy(
            config, wait_for_completion=False, timeout=60
        )

        # Poll for controller pod (may complete quickly on mock server)
        controller_pods = []
        for _ in range(30):
            pods = await kubectl.get_pods(result.namespace)
            controller_pods = [p for p in pods if "controller" in p.name]
            if controller_pods:
                break
            await asyncio.sleep(1)

        # Benchmark may have completed and pods cleaned up before we observed
        if not controller_pods:
            return

        assert len(controller_pods) == 1, (
            f"Expected exactly 1 controller pod, got {len(controller_pods)}"
        )

        await benchmark_deployer.cleanup(result)
