# tests/kubernetes/chaos/test_sweep_controller_kill.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chaos test: kill the sweep-controller pod mid-sweep, assert idempotent resume.

Submits a 4-cell sweep, watches the sweep-controller's existing `/results`
output files through the results sidecar, force-deletes the sweep-controller
pod, asserts JobSet creates a replacement, and verifies the final aggregate
includes all 8 child results without re-running the children that were already
created.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import orjson
import pytest

from aiperf.kubernetes.subproc import run_command
from tests.kubernetes.helpers.kubectl import KubectlClient
from tests.kubernetes.helpers.operator import OperatorDeployer

pytestmark = [pytest.mark.asyncio, pytest.mark.k8s_slow]

EXPECTED_CHILD_RUNS = 8
OUTPUT_READY_THRESHOLD = 2
_TERMINAL_SWEEP_PHASES = frozenset(
    {"Succeeded", "Failed", "Cancelled", "PartiallyFailed"}
)


def _build_sweep_manifest(*, name: str, namespace: str, image: str) -> str:
    body: dict[str, Any] = {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfSweep",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "image": image,
            "imagePullPolicy": "Never",
            "sweep": {
                "type": "grid",
                "parameters": {"phases.profiling.concurrency": [1, 2, 3, 4]},
            },
            "multiRun": {"numRuns": 2},
            "benchmark": {
                "models": {"items": [{"name": "mock-model"}]},
                "endpoint": {
                    "urls": [
                        "http://aiperf-mock-server.default.svc.cluster.local:8000/v1"
                    ]
                },
                "datasets": [
                    {
                        "name": "main",
                        "type": "synthetic",
                        "entries": 64,
                        "prompts": {"isl": {"mean": 128}},
                    }
                ],
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "concurrency": 1,
                        "duration": 12,
                    }
                ],
                "tokenizer": {"name": "gpt2"},
                "runtime": {"ui": "none"},
            },
        },
    }
    return orjson.dumps(body, option=orjson.OPT_INDENT_2).decode()


async def _get_sweep_controller_pod(
    kubectl: KubectlClient,
    *,
    namespace: str,
    sweep_name: str,
    timeout: float = 120.0,
) -> str:
    deadline = asyncio.get_event_loop().time() + timeout
    selector = (
        f"jobset.sigs.k8s.io/jobset-name=aiperf-{sweep_name},"
        "jobset.sigs.k8s.io/replicatedjob-name=controller"
    )
    while asyncio.get_event_loop().time() < deadline:
        res = await kubectl.run(
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            selector,
            "-o",
            "jsonpath={.items[0].metadata.name}",
            check=False,
        )
        pod = res.stdout.strip()
        if pod:
            return pod
        await asyncio.sleep(1.0)
    raise TimeoutError(
        f"no sweep-controller pod found for AIPerfSweep {namespace}/{sweep_name}"
    )


async def _get_sweep_controller_pod_uid(
    kubectl: KubectlClient,
    *,
    pod: str,
    namespace: str,
) -> str:
    res = await kubectl.run(
        "get",
        "pod",
        pod,
        "-n",
        namespace,
        "-o",
        "jsonpath={.metadata.uid}",
        check=False,
    )
    return res.stdout.strip()


async def _wait_for_replacement_sweep_controller_pod(
    kubectl: KubectlClient,
    *,
    namespace: str,
    sweep_name: str,
    deleted_uid: str,
    timeout: float = 120.0,
) -> str:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            pod = await _get_sweep_controller_pod(
                kubectl, namespace=namespace, sweep_name=sweep_name, timeout=5.0
            )
        except TimeoutError:
            await asyncio.sleep(1.0)
            continue
        uid = await _get_sweep_controller_pod_uid(kubectl, pod=pod, namespace=namespace)
        if uid and uid != deleted_uid:
            return pod
        await asyncio.sleep(1.0)
    raise TimeoutError(
        f"JobSet did not create a replacement sweep-controller pod for "
        f"AIPerfSweep {namespace}/{sweep_name} within {timeout} s"
    )


async def _get_sweep_status(
    kubectl: KubectlClient,
    *,
    namespace: str,
    sweep_name: str,
) -> dict[str, Any]:
    res = await kubectl.run(
        "get",
        "aiperfsweep",
        sweep_name,
        "-n",
        namespace,
        "-o",
        "json",
        check=True,
    )
    return orjson.loads(res.stdout).get("status", {})


def _durable_aggregate_api_path(*, namespace: str, sweep_name: str, epoch: str) -> str:
    return (
        f"/api/v1/sweeps/{namespace}/{sweep_name}/epochs/{epoch}/"
        "artifacts/aggregate.json"
    )


async def _wait_for_durable_sweep(
    kubectl: KubectlClient,
    *,
    namespace: str,
    sweep_name: str,
    timeout: float = 900.0,
) -> dict[str, Any]:
    deadline = asyncio.get_event_loop().time() + timeout
    last_status: dict[str, Any] = {}
    while asyncio.get_event_loop().time() < deadline:
        last_status = await _get_sweep_status(
            kubectl, namespace=namespace, sweep_name=sweep_name
        )
        phase = last_status.get("phase")
        if phase in _TERMINAL_SWEEP_PHASES and phase != "Succeeded":
            raise AssertionError(
                f"AIPerfSweep {namespace}/{sweep_name} reached phase={phase!r}; "
                f"status={last_status!r}"
            )
        epoch = str(last_status.get("runEpoch") or "")
        aggregate_ref = last_status.get("aggregateRef") or {}
        if (
            phase == "Succeeded"
            and epoch
            and (last_status.get("aggregation") or {}).get("phase") == "Complete"
            and last_status.get("resultsAvailable") is True
            and aggregate_ref.get("apiPath")
            == _durable_aggregate_api_path(
                namespace=namespace,
                sweep_name=sweep_name,
                epoch=epoch,
            )
        ):
            return last_status
        await asyncio.sleep(5.0)
    raise TimeoutError(
        f"AIPerfSweep {namespace}/{sweep_name} did not publish its durable "
        "aggregate; "
        f"last status={last_status!r}"
    )


async def _download_sweep_results(
    *,
    destination: Path,
    name: str,
    namespace: str,
    epoch: str,
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
            "--run",
            epoch,
            "--output",
            str(destination),
            "--all",
            "--kube-context",
            kube_context,
        ],
        timeout=300,
    )
    assert result.ok, (
        f"aiperf kube results failed:\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


async def _read_json(path: Path) -> dict[str, Any]:
    payload = orjson.loads(await asyncio.to_thread(path.read_bytes))
    assert isinstance(payload, dict), path
    return payload


def _assert_durable_sweep_status(
    status: dict[str, Any], *, namespace: str, sweep_name: str
) -> str:
    assert status.get("phase") == "Succeeded"
    assert status.get("completedRuns") == EXPECTED_CHILD_RUNS
    assert status.get("maxTotalRuns") == EXPECTED_CHILD_RUNS
    assert (status.get("aggregation") or {}).get("phase") == "Complete"
    assert status.get("resultsAvailable") is True
    aggregate_children = ((status.get("aggregate") or {}).get("children") or {}).get(
        "children", []
    )
    assert len(aggregate_children) == EXPECTED_CHILD_RUNS

    run_epoch = str(status.get("runEpoch") or "")
    assert run_epoch
    expected_api_path = _durable_aggregate_api_path(
        namespace=namespace,
        sweep_name=sweep_name,
        epoch=run_epoch,
    )
    aggregate_ref = status.get("aggregateRef") or {}
    assert aggregate_ref.get("apiPath") == expected_api_path
    assert str(aggregate_ref.get("url") or "").endswith(expected_api_path)
    return run_epoch


async def _assert_downloaded_sweep_archive(
    destination: Path, *, run_epoch: str
) -> None:
    downloaded_manifest = await _read_json(destination / "sweep_manifest.json")
    archived_parent = await _read_json(destination / "aggregate.json")
    archived_children = await _read_json(destination / "children.json")

    assert str(downloaded_manifest.get("sweepRunEpoch")) == run_epoch
    assert archived_children.get("sweep_run_epoch") == run_epoch
    assert len(archived_parent.get("childRuns") or []) == EXPECTED_CHILD_RUNS

    manifest_children = downloaded_manifest.get("children") or []
    children_on_disk = archived_children.get("children") or []
    assert len(manifest_children) == EXPECTED_CHILD_RUNS
    assert len(children_on_disk) == EXPECTED_CHILD_RUNS
    expected_cells = {
        (variation, trial) for variation in range(4) for trial in range(2)
    }
    downloaded_cells = {
        (int(child["variationIndex"]), int(child["trialIndex"]))
        for child in manifest_children
    }
    archived_cells = {
        (int(child["variation_index"]), int(child["trial_index"]))
        for child in children_on_disk
    }
    assert downloaded_cells == expected_cells
    assert archived_cells == expected_cells

    archived_by_name = {child["name"]: child for child in children_on_disk}
    for child in manifest_children:
        variation_index = int(child["variationIndex"])
        trial_index = int(child["trialIndex"])
        child_name = child["name"]
        child_epoch = str(child.get("childRunEpoch") or "")
        assert child_epoch
        assert (
            str(archived_by_name[child_name].get("child_run_epoch") or "")
            == child_epoch
        )
        child_dir = destination / f"v{variation_index}-t{trial_index}"
        assert (child_dir / "metrics.json").is_file(), child_dir
        assert (child_dir / "profile_export_aiperf.json").is_file(), child_dir


def _parse_find_file_count(stdout: str) -> int:
    stripped = stdout.strip()
    return int(stripped) if stripped.isdigit() else 0


async def _count_child_sweep_markers(
    kubectl: KubectlClient,
    *,
    pod: str,
    namespace: str,
    exec_container: str,
) -> int:
    res = await kubectl.run(
        "exec",
        pod,
        "-c",
        exec_container,
        "-n",
        namespace,
        "--",
        "/bin/bash",
        "-c",
        "find /results -mindepth 3 -maxdepth 3 -name sweep.json -type f | wc -l",
        check=False,
    )
    return _parse_find_file_count(res.stdout)


async def _wait_for_child_sweep_markers(
    kubectl: KubectlClient,
    *,
    pod: str,
    namespace: str,
    minimum: int,
    timeout: float = 300.0,
) -> int:
    deadline = asyncio.get_event_loop().time() + timeout
    last_count = 0
    while asyncio.get_event_loop().time() < deadline:
        last_count = await _count_child_sweep_markers(
            kubectl,
            pod=pod,
            namespace=namespace,
            exec_container="results-sidecar",
        )
        if last_count >= minimum:
            return last_count
        await asyncio.sleep(1.0)
    raise TimeoutError(
        f"sweep-controller pod {namespace}/{pod} did not write at least "
        f"{minimum} child sweep markers under /results; last_count={last_count}"
    )


async def _get_child_uids(
    kubectl: KubectlClient,
    *,
    namespace: str,
    sweep_name: str,
) -> dict[str, str]:
    res = await kubectl.run(
        "get",
        "aiperfjob",
        "-n",
        namespace,
        "-l",
        f"aiperf.nvidia.com/sweep={sweep_name}",
        "-o",
        "jsonpath={range .items[*]}{.metadata.name}={.metadata.uid}{'\\n'}{end}",
        check=False,
    )
    children: dict[str, str] = {}
    for line in res.stdout.splitlines():
        name, _, uid = line.partition("=")
        if name and uid:
            children[name.strip()] = uid.strip()
    return children


async def test_parse_find_file_count_handles_wc_output() -> None:
    assert _parse_find_file_count("2\n") == OUTPUT_READY_THRESHOLD
    assert _parse_find_file_count("      7\n") == 7
    assert _parse_find_file_count("find: /results: No such file or directory\n") == 0


@pytest.mark.timeout(1200)
async def test_sweep_controller_kill_resumes_correctly(
    operator_ready: OperatorDeployer,
    operator_job_namespace: str,
    kubectl: KubectlClient,
    k8s_settings,  # noqa: ANN001 - test-fixture dataclass
    tmp_path: Path,
) -> None:
    assert operator_ready is not None
    sweep_name = f"chaos-sweep-kill-{uuid.uuid4().hex[:6]}"
    manifest = _build_sweep_manifest(
        name=sweep_name,
        namespace=operator_job_namespace,
        image=k8s_settings.aiperf_image,
    )
    try:
        await kubectl.apply(manifest, namespace=operator_job_namespace)
        pod = await _get_sweep_controller_pod(
            kubectl, namespace=operator_job_namespace, sweep_name=sweep_name
        )
        marker_count = await _wait_for_child_sweep_markers(
            kubectl,
            pod=pod,
            namespace=operator_job_namespace,
            minimum=OUTPUT_READY_THRESHOLD,
            timeout=360.0,
        )
        assert marker_count >= OUTPUT_READY_THRESHOLD
        completed_before_kill = await _get_child_uids(
            kubectl, namespace=operator_job_namespace, sweep_name=sweep_name
        )
        assert len(completed_before_kill) >= OUTPUT_READY_THRESHOLD

        deleted_uid = await _get_sweep_controller_pod_uid(
            kubectl, pod=pod, namespace=operator_job_namespace
        )
        await kubectl.run(
            "delete",
            "pod",
            pod,
            "-n",
            operator_job_namespace,
            "--grace-period=0",
            "--force",
            "--ignore-not-found",
            check=False,
        )
        replacement_pod = await _wait_for_replacement_sweep_controller_pod(
            kubectl,
            namespace=operator_job_namespace,
            sweep_name=sweep_name,
            deleted_uid=deleted_uid,
            timeout=120.0,
        )
        assert replacement_pod

        final_status = await _wait_for_durable_sweep(
            kubectl,
            namespace=operator_job_namespace,
            sweep_name=sweep_name,
            timeout=900.0,
        )
        run_epoch = _assert_durable_sweep_status(
            final_status,
            namespace=operator_job_namespace,
            sweep_name=sweep_name,
        )

        destination = tmp_path / "sweep-results"
        await _download_sweep_results(
            destination=destination,
            name=sweep_name,
            namespace=operator_job_namespace,
            epoch=run_epoch,
            kube_context=kubectl.context,
        )
        await _assert_downloaded_sweep_archive(destination, run_epoch=run_epoch)

        children_after = await _get_child_uids(
            kubectl, namespace=operator_job_namespace, sweep_name=sweep_name
        )
        assert len(children_after) == EXPECTED_CHILD_RUNS
        for child_name, uid in completed_before_kill.items():
            assert children_after.get(child_name) == uid
    finally:
        await kubectl.run(
            "delete",
            "aiperfsweep",
            sweep_name,
            "-n",
            operator_job_namespace,
            "--wait=false",
            "--ignore-not-found",
            check=False,
        )
