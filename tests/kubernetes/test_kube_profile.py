# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf kube profile command with auto-attach behavior."""

from __future__ import annotations

import asyncio

import pytest


async def _run_command(project_root, *args, timeout=10):
    """Run a CLI command asynchronously."""
    cmd = ["uv", "run", "aiperf", *args]
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
    return proc


@pytest.mark.k8s
@pytest.mark.asyncio
async def test_kube_profile_command_exists(project_root):
    """Test that kube profile command is available and shows help."""
    result = await _run_command(project_root, "kube", "profile", "--help")

    assert result.returncode == 0, f"Command failed: {result.stderr}"
    assert "profile" in result.stdout.lower()
    assert "--detach" in result.stdout
    assert "--attach-port" in result.stdout
    assert "--no-wait" in result.stdout


@pytest.mark.k8s
@pytest.mark.asyncio
async def test_kube_profile_has_all_benchmark_parameters(project_root):
    """Test that profile command accepts all standard benchmark parameters."""
    result = await _run_command(project_root, "kube", "profile", "--help")

    assert result.returncode == 0

    # Check for key benchmark parameters
    assert "--model" in result.stdout or "--model-names" in result.stdout
    assert "--url" in result.stdout or "urls" in result.stdout.lower()
    assert "--concurrency" in result.stdout
    assert "--request-count" in result.stdout
    assert "--endpoint-type" in result.stdout

    # Check for Kubernetes-specific parameters
    assert "--image" in result.stdout
    assert "--workers-max" in result.stdout
    assert "--namespace" in result.stdout


@pytest.mark.k8s
@pytest.mark.asyncio
async def test_kube_generate_with_cli_params_works(project_root):
    """Test that kube generate works with CLI parameters (not config file)."""
    result = await _run_command(
        project_root,
        "kube",
        "generate",
        "--operator",
        "--model",
        "test-model",
        "--url",
        "http://server:8000",
        "--image",
        "aiperf:local",
        "--workers-max",
        "2",
        "--concurrency",
        "5",
        "--request-count",
        "10",
        timeout=30,
    )

    # Should succeed and output AIPerfJob CR YAML
    assert result.returncode == 0, f"Command failed: {result.stderr}"
    assert "kind: AIPerfJob" in result.stdout
    assert "apiVersion: aiperf.nvidia.com/v1alpha1" in result.stdout


@pytest.mark.k8s
@pytest.mark.asyncio
async def test_kube_generate_no_operator_with_cli_params_works(project_root):
    """Test that kube generate --no-operator outputs raw K8s manifests."""
    result = await _run_command(
        project_root,
        "kube",
        "generate",
        "--no-operator",
        "--model",
        "test-model",
        "--url",
        "http://server:8000",
        "--image",
        "aiperf:local",
        "--workers-max",
        "2",
        "--concurrency",
        "5",
        "--request-count",
        "10",
        timeout=30,
    )

    assert result.returncode == 0, f"Command failed: {result.stderr}"
    assert "kind: JobSet" in result.stdout
    assert "kind: ConfigMap" in result.stdout
    assert "kind: AIPerfJob" not in result.stdout
