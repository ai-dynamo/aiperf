# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live acceptance coverage for operator-managed AIPerfSweep execution."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import orjson
import pytest
import yaml

from aiperf.kubernetes.subproc import run_command
from aiperf.operator.environment import OperatorEnvironment
from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.operator import AIPerfJobConfig, OperatorDeployer

_TERMINAL_PHASES = frozenset({"Succeeded", "Failed", "Cancelled", "PartiallyFailed"})


def _grid_sweep_config() -> dict[str, Any]:
    benchmark = AIPerfJobConfig(
        concurrency=1,
        request_count=5,
        warmup_request_count=0,
    ).to_flat_spec()
    benchmark["artifacts"] = {"raw": True, "prefix": "sweep-e2e"}
    benchmark["sweep"] = {
        "type": "grid",
        "parameters": {"phases.profiling.concurrency": [1, 2]},
    }
    benchmark["randomSeed"] = 42
    return benchmark


def _multi_run_config() -> dict[str, Any]:
    benchmark = AIPerfJobConfig(
        concurrency=1,
        request_count=3,
        warmup_request_count=0,
    ).to_flat_spec()
    benchmark["multiRun"] = {"numRuns": 2, "cooldownSeconds": 0}
    benchmark["randomSeed"] = 42
    return benchmark


def _adaptive_sweep_config() -> dict[str, Any]:
    benchmark = AIPerfJobConfig(
        concurrency=1,
        request_count=3,
        warmup_request_count=0,
    ).to_flat_spec()
    benchmark["sweep"] = {
        "type": "adaptive_search",
        "searchSpace": [
            {
                "path": "phases.profiling.concurrency",
                "lo": 1,
                "hi": 2,
                "kind": "int",
            }
        ],
        "objectives": [
            {
                "metric": "output_token_throughput",
                "stat": "avg",
                "direction": "maximize",
            }
        ],
        "maxIterations": 2,
        "nInitialPoints": 1,
        "randomSeed": 42,
    }
    return benchmark


def _sobol_sweep_config() -> dict[str, Any]:
    benchmark = AIPerfJobConfig(
        concurrency=1,
        request_count=3,
        warmup_request_count=0,
    ).to_flat_spec()
    benchmark["sweep"] = {
        "type": "sobol",
        "samples": 2,
        "seed": 42,
        "dimensions": [
            {
                "path": "phases.profiling.concurrency",
                "lo": 1,
                "hi": 2,
                "kind": "int",
            }
        ],
    }
    return benchmark


async def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write a temporary authored config without blocking the event loop."""
    content = yaml.safe_dump(data, sort_keys=False)
    await asyncio.to_thread(path.write_text, content, encoding="utf-8")


async def _submit_sweep(
    *,
    config_path: Path,
    name: str,
    namespace: str,
    image: str,
    kube_context: str,
) -> None:
    result = await run_command(
        [
            "uv",
            "run",
            "aiperf",
            "kube",
            "sweep",
            "--config",
            str(config_path),
            "--name",
            name,
            "--namespace",
            namespace,
            "--image",
            image,
            "--image-pull-policy",
            "Never",
            "--kube-context",
            kube_context,
            "--detach",
        ],
        timeout=90,
    )
    assert result.ok, (
        f"aiperf kube sweep failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


async def _wait_for_durable_sweep(
    *,
    kubectl: KubectlClient,
    operator: OperatorDeployer,
    name: str,
    namespace: str,
    timeout: int,
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    last_doc: dict[str, Any] = {}

    while loop.time() < deadline:
        try:
            last_doc = await kubectl.get_json("aiperfsweep", name, namespace=namespace)
        except RuntimeError:
            await asyncio.sleep(2)
            continue

        status = last_doc.get("status", {})
        phase = status.get("phase")
        if phase in _TERMINAL_PHASES and phase != "Succeeded":
            return last_doc

        aggregate_ref = status.get("aggregateRef", {})
        durable_path = aggregate_ref.get("apiPath", "")
        if (
            phase == "Succeeded"
            and status.get("resultsAvailable") is True
            and "/epochs/" in durable_path
        ):
            return last_doc
        await asyncio.sleep(2)

    operator_logs = await operator.get_operator_logs(tail=100)
    pytest.fail(
        f"AIPerfSweep {namespace}/{name} did not produce durable results within "
        f"{timeout}s. Last status: {last_doc.get('status', {})}\n"
        f"Operator logs (last 100):\n{operator_logs}"
    )


async def _download_sweep_results(
    *,
    destination: Path,
    name: str,
    namespace: str,
    kube_context: str,
) -> None:
    result = await run_command(
        [
            "uv",
            "run",
            "aiperf",
            "kube",
            "results",
            name,
            "--namespace",
            namespace,
            "--output",
            str(destination),
            "--all",
            "--kube-context",
            kube_context,
        ],
        timeout=90,
    )
    assert result.ok, (
        f"aiperf kube results failed:\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


async def _wait_for_sweep_deleted(
    *,
    kubectl: KubectlClient,
    operator: OperatorDeployer,
    name: str,
    namespace: str,
    timeout: float,
) -> None:
    """Wait for the operator's parent-sweep TTL reaper to delete the CR."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        result = await kubectl.run(
            "get",
            "aiperfsweep",
            name,
            "-n",
            namespace,
            "--ignore-not-found",
            "-o",
            "name",
            check=False,
        )
        if result.returncode == 0 and not result.stdout.strip():
            return
        await asyncio.sleep(1)

    operator_logs = await operator.get_operator_logs(tail=100)
    pytest.fail(
        f"AIPerfSweep {namespace}/{name} still exists {timeout:.0f}s after "
        "ttlSecondsAfterFinished was set to zero.\n"
        f"Operator logs (last 100):\n{operator_logs}"
    )


@pytest.mark.timeout(600)
@pytest.mark.asyncio
async def test_grid_sweep_completes_and_harvests_aggregate(
    operator_ready: OperatorDeployer,
    kubectl: KubectlClient,
    k8s_settings: Any,
    operator_job_namespace: str,
    tmp_path: Path,
) -> None:
    """Run two child jobs and require the parent aggregate to become durable."""
    name = f"grid-{uuid.uuid4().hex[:8]}"
    config_path = tmp_path / "grid-sweep.yaml"
    await _write_yaml(config_path, _grid_sweep_config())

    await _submit_sweep(
        config_path=config_path,
        name=name,
        namespace=operator_job_namespace,
        image=k8s_settings.aiperf_image,
        kube_context=kubectl.context,
    )
    try:
        doc = await _wait_for_durable_sweep(
            kubectl=kubectl,
            operator=operator_ready,
            name=name,
            namespace=operator_job_namespace,
            timeout=k8s_settings.benchmark_timeout,
        )

        status = doc.get("status", {})
        assert status.get("phase") == "Succeeded", status
        assert status.get("totalVariations") == 2, status
        assert status.get("maxTotalRuns") == 2, status
        assert status.get("completedRuns") == 2, status
        assert status.get("failedRuns") == 0, status
        assert status.get("runStates") == {
            "pending": 0,
            "running": 0,
            "completed": 2,
            "failed": 0,
            "cancelled": 0,
        }
        assert status.get("aggregation", {}).get("phase") == "Complete", status
        assert status.get("resultsAvailable") is True, status

        aggregate = status.get("aggregate", {})
        parent = aggregate.get("parent", {})
        assert parent.get("phase") == "Succeeded", aggregate
        assert len(parent.get("childRuns", [])) == 2, aggregate
        assert len(aggregate.get("children", {}).get("children", [])) == 2, aggregate

        children_result = await kubectl.run(
            "get",
            "aiperfjobs",
            "-n",
            operator_job_namespace,
            "-l",
            f"aiperf.nvidia.com/sweep={name}",
            "-o",
            "json",
        )
        children = orjson.loads(children_result.stdout).get("items", [])
        assert len(children) == 2
        assert {
            child.get("metadata", {})
            .get("labels", {})
            .get("aiperf.nvidia.com/variation-index")
            for child in children
        } == {"00", "01"}
        assert all(
            child.get("status", {}).get("phase") == "Completed" for child in children
        )

        destination = tmp_path / "sweep-results"
        await _download_sweep_results(
            destination=destination,
            name=name,
            namespace=operator_job_namespace,
            kube_context=kubectl.context,
        )
        downloaded_manifest = orjson.loads(
            await asyncio.to_thread((destination / "sweep_manifest.json").read_bytes)
        )
        assert len(downloaded_manifest.get("children", [])) == 2

        raw_files = sorted(destination.rglob("sweep-e2e_raw.jsonl"))
        assert len(raw_files) == 2, sorted(str(path) for path in destination.rglob("*"))
        for raw_file in raw_files:
            raw_content = await asyncio.to_thread(raw_file.read_bytes)
            records = [
                orjson.loads(line) for line in raw_content.splitlines() if line.strip()
            ]
            assert len(records) == 5, raw_file

        ttl_patch = await kubectl.run(
            "patch",
            "aiperfsweep",
            name,
            "-n",
            operator_job_namespace,
            "--type=merge",
            "-p",
            '{"spec":{"ttlSecondsAfterFinished":0}}',
            check=False,
        )
        assert ttl_patch.returncode == 0, ttl_patch.stderr
        await _wait_for_sweep_deleted(
            kubectl=kubectl,
            operator=operator_ready,
            name=name,
            namespace=operator_job_namespace,
            timeout=max(30.0, OperatorEnvironment.MONITOR.INTERVAL * 3),
        )
    finally:
        await kubectl.run(
            "delete",
            "aiperfsweep",
            name,
            "-n",
            operator_job_namespace,
            "--ignore-not-found",
            "--wait=false",
            check=False,
        )


@pytest.mark.timeout(600)
@pytest.mark.asyncio
async def test_adaptive_sweep_runs_shared_planner_and_archives_history(
    operator_ready: OperatorDeployer,
    kubectl: KubectlClient,
    k8s_settings: Any,
    operator_job_namespace: str,
    tmp_path: Path,
) -> None:
    """Run the canonical adaptive planner through two Kubernetes children."""
    name = f"adaptive-{uuid.uuid4().hex[:8]}"
    config_path = tmp_path / "adaptive-sweep.yaml"
    await _write_yaml(config_path, _adaptive_sweep_config())

    await _submit_sweep(
        config_path=config_path,
        name=name,
        namespace=operator_job_namespace,
        image=k8s_settings.aiperf_image,
        kube_context=kubectl.context,
    )
    try:
        doc = await _wait_for_durable_sweep(
            kubectl=kubectl,
            operator=operator_ready,
            name=name,
            namespace=operator_job_namespace,
            timeout=k8s_settings.benchmark_timeout,
        )

        status = doc.get("status", {})
        assert status.get("phase") == "Succeeded", status
        assert status.get("totalVariations") == 2, status
        assert status.get("maxTotalRuns") == 2, status
        assert status.get("completedRuns") == 2, status
        assert status.get("failedRuns") == 0, status
        assert status.get("resultsAvailable") is True, status

        children_result = await kubectl.run(
            "get",
            "aiperfjobs",
            "-n",
            operator_job_namespace,
            "-l",
            f"aiperf.nvidia.com/sweep={name}",
            "-o",
            "json",
        )
        children = orjson.loads(children_result.stdout).get("items", [])
        assert len(children) == 2
        assert {
            child.get("metadata", {})
            .get("labels", {})
            .get("aiperf.nvidia.com/variation-index")
            for child in children
        } == {"00", "01"}
        assert all(
            child.get("status", {}).get("phase") == "Completed" for child in children
        )

        destination = tmp_path / "adaptive-results"
        await _download_sweep_results(
            destination=destination,
            name=name,
            namespace=operator_job_namespace,
            kube_context=kubectl.context,
        )
        history_files = list(destination.rglob("search_history.json"))
        assert len(history_files) == 1, sorted(
            str(path) for path in destination.rglob("*")
        )
        history = orjson.loads(await asyncio.to_thread(history_files[0].read_bytes))
        assert len(history.get("iterations", [])) == 2, history
        assert history.get("convergence_reason") == "max_iterations", history
        assert history.get("best_trials"), history
        assert history.get("config", {}).get("planner") == "bayesian", history
    finally:
        await kubectl.run(
            "delete",
            "aiperfsweep",
            name,
            "-n",
            operator_job_namespace,
            "--ignore-not-found",
            "--wait=false",
            check=False,
        )


@pytest.mark.timeout(600)
@pytest.mark.asyncio
async def test_sobol_sweep_archives_sampling_design_with_parent_epoch(
    operator_ready: OperatorDeployer,
    kubectl: KubectlClient,
    k8s_settings: Any,
    operator_job_namespace: str,
    tmp_path: Path,
) -> None:
    """Preserve the canonical QMC design without shared-PVC root collisions."""
    name = f"sobol-{uuid.uuid4().hex[:8]}"
    config_path = tmp_path / "sobol-sweep.yaml"
    await _write_yaml(config_path, _sobol_sweep_config())

    await _submit_sweep(
        config_path=config_path,
        name=name,
        namespace=operator_job_namespace,
        image=k8s_settings.aiperf_image,
        kube_context=kubectl.context,
    )
    try:
        doc = await _wait_for_durable_sweep(
            kubectl=kubectl,
            operator=operator_ready,
            name=name,
            namespace=operator_job_namespace,
            timeout=k8s_settings.benchmark_timeout,
        )

        status = doc.get("status", {})
        assert status.get("phase") == "Succeeded", status
        assert status.get("totalVariations") == 2, status
        assert status.get("completedRuns") == 2, status
        assert status.get("failedRuns") == 0, status
        assert status.get("resultsAvailable") is True, status

        destination = tmp_path / "sobol-results"
        await _download_sweep_results(
            destination=destination,
            name=name,
            namespace=operator_job_namespace,
            kube_context=kubectl.context,
        )
        design_files = list(destination.rglob("sampling_design.json"))
        assert len(design_files) == 1, sorted(
            str(path) for path in destination.rglob("*")
        )
        design = orjson.loads(await asyncio.to_thread(design_files[0].read_bytes))
        assert design.get("type") == "sobol", design
        assert design.get("samples") == 2, design
        assert design.get("seed") == 42, design
        assert len(design.get("samples_mapped", [])) == 2, design
    finally:
        await kubectl.run(
            "delete",
            "aiperfsweep",
            name,
            "-n",
            operator_job_namespace,
            "--ignore-not-found",
            "--wait=false",
            check=False,
        )


@pytest.mark.timeout(600)
@pytest.mark.asyncio
async def test_multi_run_without_parameter_axis_uses_one_cell_sweep(
    operator_ready: OperatorDeployer,
    kubectl: KubectlClient,
    k8s_settings: Any,
    operator_job_namespace: str,
    tmp_path: Path,
) -> None:
    """Require two authored trials even when there is no parameter sweep."""
    name = f"multi-{uuid.uuid4().hex[:8]}"
    config_path = tmp_path / "multi-run.yaml"
    await _write_yaml(config_path, _multi_run_config())

    await _submit_sweep(
        config_path=config_path,
        name=name,
        namespace=operator_job_namespace,
        image=k8s_settings.aiperf_image,
        kube_context=kubectl.context,
    )
    try:
        doc = await _wait_for_durable_sweep(
            kubectl=kubectl,
            operator=operator_ready,
            name=name,
            namespace=operator_job_namespace,
            timeout=k8s_settings.benchmark_timeout,
        )

        status = doc.get("status", {})
        assert status.get("phase") == "Succeeded", status
        assert status.get("totalVariations") == 1, status
        assert status.get("maxTotalRuns") == 2, status
        assert status.get("completedRuns") == 2, status
        assert status.get("failedRuns") == 0, status
        assert status.get("resultsAvailable") is True, status

        children_result = await kubectl.run(
            "get",
            "aiperfjobs",
            "-n",
            operator_job_namespace,
            "-l",
            f"aiperf.nvidia.com/sweep={name}",
            "-o",
            "json",
        )
        children = orjson.loads(children_result.stdout).get("items", [])
        assert len(children) == 2
        assert {
            child.get("metadata", {})
            .get("labels", {})
            .get("aiperf.nvidia.com/variation-index")
            for child in children
        } == {"00"}
        assert {
            child.get("metadata", {})
            .get("labels", {})
            .get("aiperf.nvidia.com/trial-index")
            for child in children
        } == {"0", "1"}
        assert all(
            child.get("status", {}).get("phase") == "Completed" for child in children
        )

        destination = tmp_path / "multi-run-results"
        await _download_sweep_results(
            destination=destination,
            name=name,
            namespace=operator_job_namespace,
            kube_context=kubectl.context,
        )
        downloaded_manifest = orjson.loads(
            await asyncio.to_thread((destination / "sweep_manifest.json").read_bytes)
        )
        assert len(downloaded_manifest.get("children", [])) == 2
    finally:
        await kubectl.run(
            "delete",
            "aiperfsweep",
            name,
            "-n",
            operator_job_namespace,
            "--ignore-not-found",
            "--wait=false",
            check=False,
        )
