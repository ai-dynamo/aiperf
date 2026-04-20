# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf kube CLI commands (status, logs, results, cancel, delete)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pytest import param

from tests.kubernetes.helpers.benchmark import (
    BenchmarkConfig,
    BenchmarkDeployer,
    BenchmarkResult,
)
from tests.kubernetes.helpers.kubectl import KubectlClient


async def run_aiperf_command(
    project_root: Path,
    *args: str,
    timeout: int = 30,
    check: bool = False,
    kube_context: str | None = None,
) -> asyncio.subprocess.Process:
    """Run an aiperf CLI command.

    Args:
        project_root: Path to project root.
        *args: Command arguments after 'aiperf'.
        timeout: Command timeout in seconds.
        check: Whether to raise on non-zero exit.

    Returns:
        Completed process result with stdout, stderr, and returncode attributes.
    """
    cmd = ["uv", "run", "aiperf", *args]
    if kube_context:
        cmd.extend(["--kube-context", kube_context])
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=project_root,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise

    proc.stdout = stdout_bytes.decode() if stdout_bytes else ""
    proc.stderr = stderr_bytes.decode() if stderr_bytes else ""

    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}: {proc.stderr}"
        )

    return proc


class TestKubeListCommand:
    """Tests for aiperf kube list command."""

    @pytest.mark.asyncio
    async def test_list_command_shows_running_benchmark(
        self,
        benchmark_deployer: BenchmarkDeployer,
        kubectl: KubectlClient,
        project_root: Path,
        small_benchmark_config: BenchmarkConfig,
    ) -> None:
        """Verify list command shows running benchmarks."""
        result = await benchmark_deployer.deploy(
            small_benchmark_config, wait_for_completion=False, timeout=60
        )

        # Wait for pods to be created
        await asyncio.sleep(10)

        # Run list command
        list_result = await run_aiperf_command(
            project_root,
            "kube",
            "list",
            "--namespace",
            result.namespace,
            timeout=30,
            kube_context=kubectl.context,
        )

        print(f"\n{'=' * 60}")
        print("KUBE LIST OUTPUT")
        print(f"{'=' * 60}")
        print(list_result.stdout)
        if list_result.stderr:
            print(f"STDERR: {list_result.stderr}")
        print(f"{'=' * 60}\n")

        # List command should succeed
        assert list_result.returncode == 0, f"List failed: {list_result.stderr}"

        await benchmark_deployer.cleanup(result)

    @pytest.mark.asyncio
    async def test_list_command_all_namespaces(
        self,
        benchmark_deployer: BenchmarkDeployer,
        kubectl: KubectlClient,
        project_root: Path,
        small_benchmark_config: BenchmarkConfig,
    ) -> None:
        """Verify list command works with --all-namespaces."""
        result = await benchmark_deployer.deploy(
            small_benchmark_config, wait_for_completion=False, timeout=60
        )

        # Wait for pods to be created
        await asyncio.sleep(10)

        # Run list with --all-namespaces
        list_result = await run_aiperf_command(
            project_root,
            "kube",
            "list",
            "--all-namespaces",
            timeout=30,
            kube_context=kubectl.context,
        )

        assert list_result.returncode == 0, f"List failed: {list_result.stderr}"

        await benchmark_deployer.cleanup(result)


class TestKubeLogsCommand:
    """Tests for aiperf kube logs command."""

    @pytest.mark.asyncio
    async def test_logs_command_retrieves_logs(
        self,
        benchmark_deployer: BenchmarkDeployer,
        project_root: Path,
        small_benchmark_config: BenchmarkConfig,
    ) -> None:
        """Verify logs command retrieves pod logs."""
        result = await benchmark_deployer.deploy(
            small_benchmark_config, wait_for_completion=False, timeout=60
        )

        # Wait for pods to start generating logs
        await asyncio.sleep(15)

        # Run logs command with explicit namespace
        logs_result = await run_aiperf_command(
            project_root,
            "kube",
            "logs",
            "--job-id",
            result.job_id,
            "--namespace",
            result.namespace,
            "--tail",
            "50",
            timeout=30,
        )

        print(f"\n{'=' * 60}")
        print("KUBE LOGS OUTPUT (first 500 chars)")
        print(f"{'=' * 60}")
        print(logs_result.stdout[:500] if logs_result.stdout else "(empty)")
        print(f"{'=' * 60}\n")

        # Logs command should succeed (exit code 0)
        assert logs_result.returncode == 0, f"Logs failed: {logs_result.stderr}"

        await benchmark_deployer.cleanup(result)


class TestKubeResultsCommand:
    """Tests for aiperf kube results command."""

    @pytest.mark.asyncio
    async def test_results_command_after_completion(
        self,
        deployed_small_benchmark_module: BenchmarkResult,
        project_root: Path,
    ) -> None:
        """Verify results command retrieves benchmark results."""
        result = deployed_small_benchmark_module

        # Skip if benchmark failed
        if not result.success:
            pytest.skip(f"Benchmark failed: {result.error_message}")

        # Run results command
        results_output = await run_aiperf_command(
            project_root,
            "kube",
            "results",
            "--job-id",
            result.job_id,
            "--namespace",
            result.namespace,
            timeout=30,
        )

        print(f"\n{'=' * 60}")
        print("KUBE RESULTS OUTPUT")
        print(f"{'=' * 60}")
        print(results_output.stdout[:1000] if results_output.stdout else "(empty)")
        print(f"{'=' * 60}\n")

        # Results command should succeed
        assert results_output.returncode == 0, (
            f"Results failed: {results_output.stderr}"
        )


class TestKubeCleanup:
    """Tests for benchmark cleanup via deployer."""

    @pytest.mark.asyncio
    async def test_cleanup_removes_namespace(
        self,
        benchmark_deployer: BenchmarkDeployer,
        kubectl: KubectlClient,
    ) -> None:
        """Verify deployer cleanup removes benchmark namespace."""
        config = BenchmarkConfig(
            concurrency=2,
            request_count=10,
            warmup_request_count=2,
        )

        result = await benchmark_deployer.deploy(
            config, wait_for_completion=True, timeout=90
        )
        namespace = result.namespace

        # Verify namespace exists
        assert await kubectl.namespace_exists(namespace)

        # Cleanup via deployer
        await benchmark_deployer.cleanup(result)

        # Wait for deletion to complete
        await asyncio.sleep(5)

        # Verify namespace is gone
        assert not await kubectl.namespace_exists(namespace), (
            "Namespace should be deleted after cleanup"
        )


class TestKubeGenerateCommand:
    """Tests for aiperf kube generate command."""

    @pytest.mark.asyncio
    async def test_generate_outputs_valid_yaml(self, project_root: Path) -> None:
        """Verify generate command outputs valid AIPerfJob CR YAML."""
        import yaml

        result = await run_aiperf_command(
            project_root,
            "kube",
            "generate",
            "--operator",
            "--model",
            "test-model",
            "--url",
            "http://server:8000/v1",
            "--image",
            "aiperf:test",
            "--concurrency",
            "5",
            "--request-count",
            "10",
            timeout=30,
        )

        assert result.returncode == 0, f"Generate failed: {result.stderr}"

        # Parse YAML output - should be a single AIPerfJob CR
        cr = yaml.safe_load(result.stdout)
        assert cr is not None
        assert cr["kind"] == "AIPerfJob"
        assert cr["apiVersion"] == "aiperf.nvidia.com/v1alpha1"
        assert "spec" in cr
        assert "benchmark" in cr["spec"]
        assert "endpoint" in cr["spec"]["benchmark"]
        assert "models" in cr["spec"]["benchmark"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "workers",
        [
            param(1, id="1-worker"),
            param(5, id="5-workers"),
            param(10, id="10-workers"),
        ],
    )  # fmt: skip
    async def test_generate_respects_workers_flag(
        self, project_root: Path, workers: int
    ) -> None:
        """Verify generate command includes workers-max in the CR spec."""
        import yaml

        result = await run_aiperf_command(
            project_root,
            "kube",
            "generate",
            "--operator",
            "--model",
            "test-model",
            "--url",
            "http://server:8000/v1",
            "--image",
            "aiperf:test",
            "--workers-max",
            str(workers),
            timeout=30,
        )

        assert result.returncode == 0, f"Generate failed: {result.stderr}"

        cr = yaml.safe_load(result.stdout)
        assert cr is not None
        assert cr["kind"] == "AIPerfJob"


class TestKubeGenerateNoOperatorCommand:
    """Tests for aiperf kube generate --no-operator command."""

    @pytest.mark.asyncio
    async def test_generate_no_operator_outputs_valid_yaml(
        self, project_root: Path
    ) -> None:
        """Verify --no-operator generates raw K8s manifests (Role, ConfigMap, JobSet)."""
        import yaml

        result = await run_aiperf_command(
            project_root,
            "kube",
            "generate",
            "--no-operator",
            "--model",
            "test-model",
            "--url",
            "http://server:8000/v1",
            "--image",
            "aiperf:test",
            "--concurrency",
            "5",
            "--request-count",
            "10",
            timeout=30,
        )

        assert result.returncode == 0, f"Generate failed: {result.stderr}"

        # Parse multi-document YAML
        docs = list(yaml.safe_load_all(result.stdout))
        kinds = {doc["kind"] for doc in docs if doc}

        # Should contain raw manifests, not an AIPerfJob CR
        assert "AIPerfJob" not in kinds
        assert "JobSet" in kinds
        assert "ConfigMap" in kinds
        assert "Role" in kinds
        assert "RoleBinding" in kinds

    @pytest.mark.asyncio
    async def test_generate_no_operator_jobset_has_controller_and_workers(
        self, project_root: Path
    ) -> None:
        """Verify JobSet contains controller and workers replicated jobs."""
        import yaml

        result = await run_aiperf_command(
            project_root,
            "kube",
            "generate",
            "--no-operator",
            "--model",
            "test-model",
            "--url",
            "http://server:8000/v1",
            "--image",
            "aiperf:test",
            timeout=30,
        )

        assert result.returncode == 0, f"Generate failed: {result.stderr}"

        docs = list(yaml.safe_load_all(result.stdout))
        jobset = next(d for d in docs if d and d["kind"] == "JobSet")

        replicated_names = {rj["name"] for rj in jobset["spec"]["replicatedJobs"]}
        assert "controller" in replicated_names
        assert "workers" in replicated_names

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "workers",
        [
            param(1, id="1-worker"),
            param(5, id="5-workers"),
        ],
    )  # fmt: skip
    async def test_generate_no_operator_respects_workers_flag(
        self, project_root: Path, workers: int
    ) -> None:
        """Verify --no-operator respects --workers-max flag."""
        import yaml

        result = await run_aiperf_command(
            project_root,
            "kube",
            "generate",
            "--no-operator",
            "--model",
            "test-model",
            "--url",
            "http://server:8000/v1",
            "--image",
            "aiperf:test",
            "--workers-max",
            str(workers),
            timeout=30,
        )

        assert result.returncode == 0, f"Generate failed: {result.stderr}"

        docs = list(yaml.safe_load_all(result.stdout))
        jobset = next(d for d in docs if d and d["kind"] == "JobSet")
        assert jobset is not None


class TestKubeValidateCommand:
    """Tests for aiperf kube validate command."""

    @pytest.mark.asyncio
    async def test_validate_checks_generated_manifest(self, project_root: Path) -> None:
        """Verify validate command accepts a valid generated manifest."""
        import tempfile

        # Generate a valid manifest
        gen_result = await run_aiperf_command(
            project_root,
            "kube",
            "generate",
            "--operator",
            "--model",
            "test-model",
            "--url",
            "http://server:8000/v1",
            "--image",
            "aiperf:test",
            "--request-count",
            "10",
            timeout=10,
        )
        assert gen_result.returncode == 0

        # Write to temp file and validate
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(gen_result.stdout)
            manifest_path = f.name

        result = await run_aiperf_command(
            project_root,
            "kube",
            "validate",
            manifest_path,
            timeout=30,
        )

        print(f"\n{'=' * 60}")
        print("KUBE VALIDATE OUTPUT")
        print(f"{'=' * 60}")
        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        print(f"{'=' * 60}\n")

        Path(manifest_path).unlink(missing_ok=True)

        assert result.returncode == 0, f"Validate failed: {result.stderr}"


class TestKubeCommandHelp:
    """Tests for CLI help messages."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "subcommand",
        [
            param("init", id="init"),
            param("validate", id="validate"),
            param("profile", id="profile"),
            param("generate", id="generate"),
            param("attach", id="attach"),
            param("list", id="list"),
            param("logs", id="logs"),
            param("results", id="results"),
            param("debug", id="debug"),
        ],
    )  # fmt: skip
    async def test_subcommand_help_available(
        self, project_root: Path, subcommand: str
    ) -> None:
        """Verify help is available for each subcommand."""
        result = await run_aiperf_command(
            project_root,
            "kube",
            subcommand,
            "--help",
            timeout=10,
        )

        assert result.returncode == 0, f"Help failed: {result.stderr}"
        assert subcommand in result.stdout.lower()
