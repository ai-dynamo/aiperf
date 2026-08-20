# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Kubernetes end-to-end benchmark deployer."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import param

from tests.kubernetes.helpers.benchmark import (
    BenchmarkConfig,
    BenchmarkDeployer,
    BenchmarkResult,
    _CollectionOutcome,
)


async def _return_result(result: BenchmarkResult, _timeout: int) -> BenchmarkResult:
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("timeout", "expected_timeout"),
    [
        param(None, 417, id="configured-default"),
        param(23, 23, id="per-deploy-override"),
    ],
)  # fmt: skip
async def test_deploy_routes_effective_timeout_to_completion_wait(
    tmp_path: Path,
    timeout: int | None,
    expected_timeout: int,
) -> None:
    """The deployer uses its configured timeout unless a call overrides it."""
    kubectl = MagicMock()
    kubectl.apply = AsyncMock(return_value="aiperfjob.aiperf.nvidia.com/bench created")
    kubectl.get_jobsets = AsyncMock(
        return_value=[SimpleNamespace(name="aiperf-bench-timeout")]
    )
    deployer = BenchmarkDeployer(
        kubectl=kubectl,
        project_root=tmp_path,
        default_timeout=417,
    )
    manifest = """apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: bench-timeout
  namespace: bench-timeout
spec: {}
"""

    with (
        patch.object(deployer, "_generate_manifest", AsyncMock(return_value=manifest)),
        patch.object(deployer, "_ensure_clean_namespace", AsyncMock()),
        patch.object(
            deployer,
            "_wait_and_collect",
            AsyncMock(side_effect=_return_result),
        ) as wait_and_collect,
        patch.object(BenchmarkResult, "print_results"),
    ):
        await deployer.deploy(BenchmarkConfig(), timeout=timeout)

    assert wait_and_collect.await_args.args[1] == expected_timeout


@pytest.mark.asyncio
async def test_deploy_delayed_jobset_tracks_aiperfjob_manifest_name(
    tmp_path: Path,
) -> None:
    """A delayed JobSet must not redirect durable polling to the namespace."""
    kubectl = MagicMock()
    kubectl.apply = AsyncMock(
        return_value="aiperfjob.aiperf.nvidia.com/bench-delayed created"
    )
    kubectl.get_jobsets = AsyncMock(return_value=[])
    deployer = BenchmarkDeployer(kubectl=kubectl, project_root=tmp_path)
    manifest = """apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: bench-delayed
  namespace: bench-ns
spec: {}
"""

    with (
        patch.object(deployer, "_generate_manifest", AsyncMock(return_value=manifest)),
        patch.object(deployer, "_ensure_clean_namespace", AsyncMock()),
        patch.object(
            deployer, "_wait_and_collect", AsyncMock(side_effect=_return_result)
        ),
        patch.object(BenchmarkResult, "print_results"),
        patch("tests.kubernetes.helpers.benchmark.asyncio.sleep", AsyncMock()),
    ):
        result = await deployer.deploy(BenchmarkConfig())

    assert result.job_id == "bench-delayed"
    assert result.namespace == "bench-ns"
    assert result.jobset_name == ""


@pytest.mark.asyncio
async def test_deploy_runs_hooks_on_their_declared_sides_of_apply(
    tmp_path: Path,
) -> None:
    """Preflight prerequisites exist before apply; observers run afterward."""
    calls: list[str] = []
    kubectl = MagicMock()

    async def apply(_manifest: str) -> str:
        calls.append("apply")
        return "aiperfjob.aiperf.nvidia.com/bench-hooks created"

    async def clean(_namespace: str) -> None:
        calls.append("clean")

    async def pre_apply(_namespace: str) -> None:
        calls.append("pre-apply")

    async def pre_wait(_namespace: str) -> None:
        calls.append("pre-wait")

    kubectl.apply = AsyncMock(side_effect=apply)
    kubectl.get_jobsets = AsyncMock(
        side_effect=lambda _namespace: (
            calls.append("jobset-poll") or [SimpleNamespace(name="aiperf-bench-hooks")]
        )
    )
    deployer = BenchmarkDeployer(kubectl=kubectl, project_root=tmp_path)
    manifest = """apiVersion: aiperf.nvidia.com/v1alpha1
kind: AIPerfJob
metadata:
  name: bench-hooks
  namespace: bench-hooks
spec: {}
"""

    with (
        patch.object(deployer, "_generate_manifest", AsyncMock(return_value=manifest)),
        patch.object(deployer, "_ensure_clean_namespace", side_effect=clean),
    ):
        await deployer.deploy(
            BenchmarkConfig(),
            wait_for_completion=False,
            pre_apply_hook=pre_apply,
            pre_wait_hook=pre_wait,
        )

    assert calls == ["clean", "pre-apply", "apply", "jobset-poll", "pre-wait"]


@pytest.mark.asyncio
async def test_collect_from_cr_backfills_delayed_jobset_name(tmp_path: Path) -> None:
    """Durable status supplies the JobSet identity after delayed creation."""
    kubectl = MagicMock()
    kubectl.get_json = AsyncMock(
        return_value={
            "metadata": {"name": "bench", "namespace": "bench-ns"},
            "status": {
                "phase": "Completed",
                "jobSetName": "aiperf-bench",
                "results": {"request_count": {"avg": 4}},
            },
        }
    )
    deployer = BenchmarkDeployer(kubectl, tmp_path)
    result = BenchmarkResult(
        namespace="bench-ns",
        jobset_name="",
        job_id="bench",
        config=BenchmarkConfig(),
    )

    outcome = await deployer._collect_from_cr(result, timeout=10)

    assert outcome is not None
    assert outcome.success
    assert result.jobset_name == "aiperf-bench"


@pytest.mark.asyncio
async def test_collect_terminal_outcome_cr_completion_cancels_api_wait(
    tmp_path: Path,
) -> None:
    """Durable CR completion wins without waiting for the controller API."""
    deployer = BenchmarkDeployer(MagicMock(), tmp_path)
    result = BenchmarkResult(
        namespace="bench-ns",
        jobset_name="aiperf-bench",
        job_id="bench",
        config=BenchmarkConfig(),
    )
    api_started = asyncio.Event()
    api_cancelled = asyncio.Event()

    async def wait_for_api(
        _result: BenchmarkResult, _timeout: int
    ) -> _CollectionOutcome | None:
        api_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            api_cancelled.set()
            raise

    async def wait_for_cr(
        _result: BenchmarkResult, _timeout: int
    ) -> _CollectionOutcome:
        await api_started.wait()
        return _CollectionOutcome(
            source="CR",
            api_results={"status": "complete"},
            success=True,
        )

    with (
        patch.object(deployer, "_collect_from_api", side_effect=wait_for_api),
        patch.object(deployer, "_collect_from_cr", side_effect=wait_for_cr),
    ):
        outcome = await deployer._collect_terminal_outcome(result, timeout=10)

    assert outcome is not None
    assert outcome.source == "CR"
    assert outcome.success
    assert api_cancelled.is_set()


@pytest.mark.asyncio
async def test_collect_terminal_outcome_api_failure_still_waits_for_cr(
    tmp_path: Path,
) -> None:
    """A dead controller API does not disable the durable CR collector."""
    deployer = BenchmarkDeployer(MagicMock(), tmp_path)
    result = BenchmarkResult(
        namespace="bench-ns",
        jobset_name="aiperf-bench",
        job_id="bench",
        config=BenchmarkConfig(),
    )

    async def fail_api(
        _result: BenchmarkResult, _timeout: int
    ) -> _CollectionOutcome | None:
        raise RuntimeError("controller pod was deleted")

    async def complete_from_cr(
        _result: BenchmarkResult, _timeout: int
    ) -> _CollectionOutcome:
        await asyncio.sleep(0)
        return _CollectionOutcome(
            source="CR",
            api_results={"status": "complete", "results": {"request_count": 4}},
            success=True,
        )

    with (
        patch.object(deployer, "_collect_from_api", side_effect=fail_api),
        patch.object(deployer, "_collect_from_cr", side_effect=complete_from_cr),
    ):
        outcome = await deployer._collect_terminal_outcome(result, timeout=10)

    assert outcome is not None
    assert outcome.source == "CR"
    assert outcome.api_results["status"] == "complete"


@pytest.mark.asyncio
async def test_collect_from_cr_waits_for_results_after_completed_phase(
    tmp_path: Path,
) -> None:
    """A transient result-less Completed status retains the metrics grace."""
    kubectl = MagicMock()
    kubectl.get_json = AsyncMock(
        side_effect=[
            {
                "metadata": {"name": "bench", "namespace": "bench-ns"},
                "status": {"phase": "Completed"},
            },
            {
                "metadata": {"name": "bench", "namespace": "bench-ns"},
                "status": {
                    "phase": "Completed",
                    "results": {"request_count": {"avg": 4}},
                },
            },
        ]
    )
    deployer = BenchmarkDeployer(kubectl, tmp_path)
    result = BenchmarkResult(
        namespace="bench-ns",
        jobset_name="aiperf-bench",
        job_id="bench",
        config=BenchmarkConfig(),
    )

    outcome = await deployer._collect_from_cr(
        result,
        timeout=10,
        results_poll_interval=0,
    )

    assert outcome is not None
    assert outcome.success
    assert outcome.api_results == {
        "status": "complete",
        "results": {"request_count": {"avg": 4}},
    }
    assert kubectl.get_json.await_count == 2
