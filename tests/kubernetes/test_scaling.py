# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for worker pod scaling and multi-pod deployments."""

from __future__ import annotations

import asyncio

import pytest
from pytest import param

from aiperf.kubernetes.constants import Containers
from tests.kubernetes.helpers.benchmark import (
    BenchmarkConfig,
    BenchmarkDeployer,
    BenchmarkResult,
)
from tests.kubernetes.helpers.kubectl import KubectlClient


def _is_probe_exempt_container(name: str) -> bool:
    """Return whether latency-sensitive runtime code intentionally omits probes."""
    return name in {Containers.RECORDS_MANAGER, Containers.WORKER_GROUP_MANAGER} or (
        name.startswith(("worker-", "record-processor-"))
    )


async def _deploy_live_benchmark(
    benchmark_deployer: BenchmarkDeployer,
    kubectl: KubectlClient,
) -> BenchmarkResult:
    """Deploy a benchmark and retain its live controller and worker pods."""
    result = await benchmark_deployer.deploy(
        BenchmarkConfig(
            concurrency=2,
            request_count=1_000_000,
            warmup_request_count=2,
        ),
        wait_for_completion=False,
        timeout=60,
    )
    try:
        assert result.jobset_name, "Benchmark deployment did not create a JobSet"
        selector = f"jobset.sigs.k8s.io/jobset-name={result.jobset_name}"
        deadline = asyncio.get_running_loop().time() + 60.0
        while asyncio.get_running_loop().time() < deadline:
            pods = await kubectl.get_pods(result.namespace, label_selector=selector)
            if any("controller" in pod.name for pod in pods) and any(
                "worker" in pod.name and "controller" not in pod.name for pod in pods
            ):
                result.pods = pods
                return result
            await asyncio.sleep(1)
        pytest.fail(
            "Benchmark did not expose both controller and worker pods before "
            "manifest assertions"
        )
    except BaseException:
        await benchmark_deployer.cleanup(result)
        raise


class TestWorkerPodScaling:
    """Tests for worker pod scaling behavior."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "workers,concurrency,expected_pods",
        [
            param(1, 2, 1, id="1-worker-1-pod"),
            param(5, 10, 1, id="5-workers-1-pod"),
            param(10, 10, 1, id="10-workers-1-pod"),
            param(11, 22, 2, id="11-workers-2-pods"),
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

        --total-workers sets total workers distributed across pods based on
        --workers-per-pod (default 10).
        """
        config = BenchmarkConfig(
            concurrency=concurrency,
            request_count=max(10, concurrency * 2),
            warmup_request_count=2,
            concurrency_ramp_duration=3.0,
            workers=workers,
        )

        result = await benchmark_deployer.deploy(
            config, wait_for_completion=False, timeout=60
        )
        try:
            assert result.jobset_name, "Benchmark deployment did not create a JobSet"
            jobset = await kubectl.get_json(
                "jobset", result.jobset_name, namespace=result.namespace
            )
            worker_jobs = [
                replicated_job
                for replicated_job in jobset.get("spec", {}).get("replicatedJobs", [])
                if replicated_job.get("name") == "workers"
            ]
            assert len(worker_jobs) == 1, (
                "Expected exactly one 'workers' replicated job, "
                f"found {len(worker_jobs)}"
            )
            assert worker_jobs[0].get("replicas", 1) == expected_pods, (
                f"Expected {expected_pods} JobSet worker replicas, "
                f"got {worker_jobs[0].get('replicas', 1)}"
            )

            # Scope to this JobSet because xdist workers share the cluster.
            worker_pods = []
            for _ in range(60):
                pods = await kubectl.get_pods(result.namespace)
                worker_pods = [
                    pod
                    for pod in pods
                    if pod.name.startswith(result.jobset_name)
                    and "worker" in pod.name
                    and "controller" not in pod.name
                ]
                if len(worker_pods) == expected_pods:
                    break
                await asyncio.sleep(1)

            assert len(worker_pods) == expected_pods, (
                f"Expected {expected_pods} worker pods, got {len(worker_pods)}: "
                f"{[pod.name for pod in worker_pods]}"
            )
        finally:
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

        result = await benchmark_deployer.deploy(config)

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
        benchmark_deployer: BenchmarkDeployer,
        kubectl: KubectlClient,
    ) -> None:
        """Verify the default burstable controller pod has requests only."""
        result = await _deploy_live_benchmark(benchmark_deployer, kubectl)
        try:
            controller = result.controller_pod
            assert controller is not None, "No controller pod was observed"

            pod_json = await kubectl.get_json(
                "pod", controller.name, namespace=result.namespace
            )
            containers = pod_json.get("spec", {}).get("containers", [])

            for container in containers:
                resources = container.get("resources", {})
                assert "requests" in resources, f"{container['name']} missing requests"
                assert "cpu" in resources["requests"]
                assert "memory" in resources["requests"]
                assert "limits" not in resources, (
                    f"{container['name']} unexpectedly has limits in burstable mode"
                )
        finally:
            await benchmark_deployer.cleanup(result)

    @pytest.mark.asyncio
    async def test_worker_pod_has_expected_resources(
        self,
        benchmark_deployer: BenchmarkDeployer,
        kubectl: KubectlClient,
    ) -> None:
        """Verify the default burstable worker pod has requests only."""
        result = await _deploy_live_benchmark(benchmark_deployer, kubectl)
        try:
            worker_pods = result.worker_pods
            assert worker_pods, "No worker pod was observed"

            pod_json = await kubectl.get_json(
                "pod", worker_pods[0].name, namespace=result.namespace
            )
            containers = pod_json.get("spec", {}).get("containers", [])

            for container in containers:
                resources = container.get("resources", {})
                assert "requests" in resources, f"{container['name']} missing requests"
                assert "cpu" in resources["requests"]
                assert "memory" in resources["requests"]
                assert "limits" not in resources, (
                    f"{container['name']} unexpectedly has limits in burstable mode"
                )
        finally:
            await benchmark_deployer.cleanup(result)


class TestPodSecurityConfiguration:
    """Tests for pod security configuration (module-scoped for speed)."""

    @pytest.mark.asyncio
    async def test_pods_run_as_non_root(
        self,
        benchmark_deployer: BenchmarkDeployer,
        kubectl: KubectlClient,
    ) -> None:
        """Verify pods run as non-root user."""
        result = await _deploy_live_benchmark(benchmark_deployer, kubectl)
        try:
            assert result.pods, "No benchmark pods were observed"
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
                    assert (
                        container_security.get("allowPrivilegeEscalation") is False
                    ), (
                        f"Container {container['name']} should not allow privilege escalation"
                    )
        finally:
            await benchmark_deployer.cleanup(result)

    @pytest.mark.asyncio
    async def test_pods_have_health_probes(
        self,
        benchmark_deployer: BenchmarkDeployer,
        kubectl: KubectlClient,
    ) -> None:
        """Verify pods have startup, liveness, and readiness probes."""
        result = await _deploy_live_benchmark(benchmark_deployer, kubectl)
        try:
            assert result.pods, "No benchmark pods were observed"
            for pod_status in result.pods:
                pod_json = await kubectl.get_json(
                    "pod", pod_status.name, namespace=result.namespace
                )
                containers = pod_json.get("spec", {}).get("containers", [])

                for container in containers:
                    # Only require probes on containers that expose a health port
                    has_health_port = any(
                        p.get("name") == "health" for p in container.get("ports", [])
                    )
                    if not has_health_port:
                        continue
                    probes = {
                        "startupProbe",
                        "livenessProbe",
                        "readinessProbe",
                    } & container.keys()
                    if _is_probe_exempt_container(container["name"]):
                        assert not probes
                        continue
                    assert probes, (
                        f"Container {container['name']} exposes a health port "
                        f"but has no startup/liveness/readiness probe"
                    )
        finally:
            await benchmark_deployer.cleanup(result)


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
            request_count=1_000_000,
            warmup_request_count=2,
            workers=workers,
        )

        result = await benchmark_deployer.deploy(
            config, wait_for_completion=False, timeout=60
        )
        try:
            assert result.jobset_name, "Benchmark deployment did not create a JobSet"
            selector = f"jobset.sigs.k8s.io/jobset-name={result.jobset_name}"
            controller_pods = []
            for _ in range(30):
                pods = await kubectl.get_pods(result.namespace, label_selector=selector)
                controller_pods = [pod for pod in pods if "controller" in pod.name]
                if controller_pods:
                    break
                await asyncio.sleep(1)

            assert controller_pods, "No controller pod was observed"
            assert len(controller_pods) == 1, (
                f"Expected exactly 1 controller pod, got {len(controller_pods)}"
            )
        finally:
            await benchmark_deployer.cleanup(result)
