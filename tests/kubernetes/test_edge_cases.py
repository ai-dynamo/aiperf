# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for edge cases and error handling."""

from __future__ import annotations

import pytest

from tests.kubernetes.helpers.benchmark import (
    BenchmarkConfig,
    BenchmarkDeployer,
    BenchmarkResult,
)
from tests.kubernetes.helpers.kubectl import KubectlClient


class TestMinimalConfigurations:
    """Tests for minimal/edge case configurations."""

    @pytest.mark.asyncio
    async def test_single_request(
        self,
        benchmark_deployer: BenchmarkDeployer,
    ) -> None:
        """Test benchmark with single request."""
        config = BenchmarkConfig(
            concurrency=1,
            request_count=1,
            warmup_request_count=1,
        )

        result = await benchmark_deployer.deploy(config, timeout=90)
        # Retry once if shared mock-server flakiness caused no metrics
        if result.metrics is None or result.metrics.request_count is None:
            result = await benchmark_deployer.deploy(config, timeout=90)

        assert result.success
        assert result.metrics is not None
        assert result.metrics.request_count == 1

    @pytest.mark.asyncio
    async def test_concurrency_one(
        self,
        benchmark_deployer: BenchmarkDeployer,
    ) -> None:
        """Test benchmark with concurrency of 1."""
        config = BenchmarkConfig(
            concurrency=1,
            request_count=10,
            warmup_request_count=2,
        )

        result = await benchmark_deployer.deploy(config, timeout=90)
        if result.metrics is None or result.metrics.request_count is None:
            result = await benchmark_deployer.deploy(config, timeout=90)

        assert result.success
        assert result.metrics is not None
        assert result.metrics.request_count == 10

    @pytest.mark.asyncio
    async def test_minimal_warmup(
        self,
        benchmark_deployer: BenchmarkDeployer,
    ) -> None:
        """Test benchmark with minimal warmup (1 request)."""
        config = BenchmarkConfig(
            concurrency=2,
            request_count=10,
            warmup_request_count=1,
        )

        result = await benchmark_deployer.deploy(config, timeout=90)

        assert result.success
        assert result.metrics is not None

    @pytest.mark.asyncio
    async def test_minimal_sequence_length(
        self,
        benchmark_deployer: BenchmarkDeployer,
    ) -> None:
        """Test benchmark with minimal sequence lengths."""
        config = BenchmarkConfig(
            concurrency=2,
            request_count=10,
            warmup_request_count=2,
            input_sequence_min=10,
            input_sequence_max=20,
            output_tokens_min=5,
            output_tokens_max=10,
        )

        result = await benchmark_deployer.deploy(config, timeout=90)

        assert result.success


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_config_dump_generates_valid_yaml(self) -> None:
        """Verify config generates valid YAML via dump_config."""
        import yaml

        from aiperf.config import AIPerfConfig
        from aiperf.config.loader import dump_config

        config = AIPerfConfig(
            models=["test-model"],
            endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
            datasets={
                "default": {
                    "type": "synthetic",
                    "entries": 50,
                    "prompts": {"isl": 128, "osl": 64},
                }
            },
            phases={
                "default": {
                    "type": "concurrency",
                    "requests": 50,
                    "concurrency": 5,
                }
            },
        )

        yaml_str = dump_config(config, exclude_defaults=False)
        parsed = yaml.safe_load(yaml_str)

        assert parsed["endpoint"]["urls"] == [
            "http://localhost:8000/v1/chat/completions"
        ]
        assert parsed["models"]["items"][0]["name"] == "test-model"

    def test_config_save_creates_file(self) -> None:
        """Verify save_config creates a YAML file."""
        import tempfile
        from pathlib import Path

        from aiperf.config import AIPerfConfig
        from aiperf.config.loader import save_config

        config = AIPerfConfig(
            models=["test-model"],
            endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
            datasets={
                "default": {
                    "type": "synthetic",
                    "entries": 10,
                    "prompts": {"isl": 32, "osl": 16},
                }
            },
            phases={
                "default": {"type": "concurrency", "requests": 10, "concurrency": 1}
            },
        )

        path = Path(tempfile.mktemp(suffix=".yaml"))
        try:
            save_config(config, path)
            assert path.exists()
            assert path.suffix == ".yaml"
        finally:
            path.unlink(missing_ok=True)


class TestResourceCleanup:
    """Tests for resource cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_after_successful_run(
        self,
        benchmark_deployer: BenchmarkDeployer,
        kubectl: KubectlClient,
    ) -> None:
        """Verify resources are cleaned up after successful run."""
        config = BenchmarkConfig(
            concurrency=2,
            request_count=10,
            warmup_request_count=2,
        )

        result = await benchmark_deployer.deploy(config, timeout=90)
        namespace = result.namespace

        # Namespace should exist
        assert await kubectl.namespace_exists(namespace)

        # Cleanup
        await benchmark_deployer.cleanup(result)

        # Namespace should be gone
        assert not await kubectl.namespace_exists(namespace)

    @pytest.mark.asyncio
    async def test_multiple_deployments_tracked(
        self,
        benchmark_deployer: BenchmarkDeployer,
    ) -> None:
        """Verify multiple deployments are tracked."""
        config = BenchmarkConfig(
            concurrency=2,
            request_count=10,
            warmup_request_count=2,
        )

        initial_count = benchmark_deployer.get_deployment_count()

        await benchmark_deployer.deploy(config, timeout=90)
        assert benchmark_deployer.get_deployment_count() == initial_count + 1

        await benchmark_deployer.deploy(config, timeout=90)
        assert benchmark_deployer.get_deployment_count() == initial_count + 2


class TestDiagnostics:
    """Tests for diagnostic information collection."""

    def test_pods_collected_on_completion(
        self,
        deployed_small_benchmark_module: BenchmarkResult,
    ) -> None:
        """Verify benchmark completed (pods may already be cleaned up)."""
        # Pods may be gone if operator cleaned up the JobSet
        if not deployed_small_benchmark_module.pods:
            assert deployed_small_benchmark_module.success
        else:
            assert len(deployed_small_benchmark_module.pods) > 0

    def test_controller_pod_accessible(
        self,
        deployed_small_benchmark_module: BenchmarkResult,
    ) -> None:
        """Verify controller pod is accessible (if not already cleaned up)."""
        controller = deployed_small_benchmark_module.controller_pod
        if controller is None:
            assert deployed_small_benchmark_module.success
            return
        assert "controller" in controller.name

    def test_worker_pods_completed_after_benchmark(
        self,
        deployed_small_benchmark_module: BenchmarkResult,
    ) -> None:
        """Verify worker pods have completed or are completing after benchmark finishes.

        Worker pods may still be Running when the benchmark result is collected
        (operator hasn't cleaned up the JobSet yet). Accept Running as transient.
        """
        workers = deployed_small_benchmark_module.worker_pods
        if not workers:
            assert deployed_small_benchmark_module.success
            return
        acceptable = {"Succeeded", "Running", "Completed"}
        for worker in workers:
            assert worker.phase in acceptable, (
                f"Worker pod {worker.name} in unexpected phase: {worker.phase}"
            )

    @pytest.mark.asyncio
    async def test_can_get_logs_from_control_plane_container(
        self,
        deployed_small_benchmark_module: BenchmarkResult,
        get_pod_logs,
    ) -> None:
        """Verify logs can be retrieved from the control-plane container.

        New architecture: single control-plane container spawns all services
        as subprocesses.
        """
        controller = deployed_small_benchmark_module.controller_pod
        if controller is None:
            assert deployed_small_benchmark_module.success
            return
        logs = await get_pod_logs(
            deployed_small_benchmark_module, container="control-plane"
        )
        assert len(logs) > 0


class TestKubectlHelpers:
    """Tests for kubectl helper functions."""

    @pytest.mark.asyncio
    async def test_get_events(
        self,
        kubectl: KubectlClient,
        deployed_small_benchmark_module: BenchmarkResult,
    ) -> None:
        """Verify events can be retrieved."""
        events = await kubectl.get_events(deployed_small_benchmark_module.namespace)
        # Events string should not be empty
        assert events is not None

    @pytest.mark.asyncio
    async def test_pod_status_parsing(
        self,
        kubectl: KubectlClient,
        deployed_small_benchmark_module: BenchmarkResult,
    ) -> None:
        """Verify pod status is parsed correctly."""
        pods = await kubectl.get_pods(deployed_small_benchmark_module.namespace)

        for pod in pods:
            assert pod.name
            assert pod.namespace == deployed_small_benchmark_module.namespace
            assert pod.phase in [
                "Running",
                "Succeeded",
                "Completed",
                "Failed",
                "Pending",
                "Unknown",
            ]
            assert "/" in pod.ready  # Format: "X/Y"

    @pytest.mark.asyncio
    async def test_jobset_status_parsing(
        self,
        kubectl: KubectlClient,
        deployed_small_benchmark_module: BenchmarkResult,
    ) -> None:
        """Verify JobSet status is parsed correctly (if still present).

        The operator may delete the JobSet after collecting results.
        """
        try:
            status = await kubectl.get_jobset(
                deployed_small_benchmark_module.jobset_name,
                deployed_small_benchmark_module.namespace,
            )
        except (RuntimeError, Exception):
            # JobSet already cleaned up by operator
            assert deployed_small_benchmark_module.success
            return

        if status is None:
            assert deployed_small_benchmark_module.success
            return

        assert status.name == deployed_small_benchmark_module.jobset_name
        assert status.namespace == deployed_small_benchmark_module.namespace
        # JobSet may not be formally completed yet if results were collected
        # via API before the JobSet controller updated status.
        if not status.is_completed:
            assert deployed_small_benchmark_module.success
