# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.config.benchmark import BenchmarkRun
from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import SweepVariation
from aiperf.sweep_controller.k8s_executor import (
    ChildNameConflictError,
    K8sChildJobExecutor,
)


def _sweep_cr() -> dict:
    return {
        "metadata": {"name": "s", "namespace": "ns", "uid": "uid"},
        "spec": {
            "template": {
                "spec": {
                    "image": "x:latest",
                    "benchmark": {
                        "models": ["m"],
                        "endpoint": {"urls": ["http://x"], "type": "chat"},
                        "datasets": [{"name": "main", "type": "synthetic"}],
                        "phases": [
                            {
                                "name": "profiling",
                                "type": "concurrency",
                                "duration": 1,
                                "concurrency": 1,
                            }
                        ],
                    },
                }
            }
        },
    }


def _benchmark_run(var_idx: int = 7, trial: int = 2) -> BenchmarkRun:
    cfg = BenchmarkConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {"urls": ["http://x"], "type": "chat"},
            "datasets": [{"name": "main", "type": "synthetic"}],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "duration": 1,
                    "concurrency": 64,
                }
            ],
        }
    )
    return BenchmarkRun(
        benchmark_id=f"s-v{var_idx:02d}-t{trial:01d}",
        cfg=cfg,
        variation=SweepVariation(
            index=var_idx,
            label="c=64",
            values={"phases.profiling.concurrency": 64},
        ),
        trial=trial,
        label=f"run_{trial:04d}",
        artifact_dir=Path("/results"),
    )


class _NotFoundException(Exception):
    """Mimics kubernetes_asyncio.client.ApiException(status=404)."""

    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__(f"ApiException({status})")


@pytest.mark.asyncio
async def test_execute_creates_child_when_not_exists(monkeypatch):
    """When no child exists, executor creates one and waits for terminal phase."""
    api = MagicMock()
    custom = MagicMock()
    # First read returns 404, second read (after watch) returns Succeeded child.
    custom.get_namespaced_custom_object = AsyncMock(
        side_effect=[
            _NotFoundException(404),
            {
                "metadata": {
                    "name": "s-v07-t2",
                    "ownerReferences": [{"uid": "uid"}],
                    "labels": {"aiperf.nvidia.com/sweep": "s"},
                },
                "status": {
                    "phase": "Succeeded",
                    "runEpoch": "1714000000",
                    "runtimeRef": {"controllerHost": "h"},
                },
            },
        ]
    )
    custom.create_namespaced_custom_object = AsyncMock(
        return_value={
            "metadata": {
                "name": "s-v07-t2",
                "uid": "child-uid",
                "ownerReferences": [{"uid": "uid"}],
                "labels": {"aiperf.nvidia.com/sweep": "s"},
            },
        }
    )
    monkeypatch.setattr(
        "aiperf.sweep_controller.k8s_executor.CustomObjectsApi", lambda _api: custom
    )
    # The executor wraps ApiException-like errors; treat our fake as such.
    monkeypatch.setattr(
        "aiperf.sweep_controller.k8s_executor.ApiException",
        _NotFoundException,
    )

    executor = K8sChildJobExecutor(api=api, sweep=_sweep_cr(), with_trial_suffix=True)
    executor._wait_until_terminal = AsyncMock(return_value=None)
    executor._pull_summary_metrics = AsyncMock(return_value={})

    result = await executor.execute(_benchmark_run())

    custom.create_namespaced_custom_object.assert_awaited_once()
    # Terminal phase=Succeeded with empty summary is now classified as success
    # (the operator may not yet have written status.summary; failing here would
    # trip failure_policy on a write race). See _collect_run_result.
    assert result.success is True
    assert result.summary_metrics == {}
    assert result.label == "run_0002"


@pytest.mark.asyncio
async def test_execute_resumes_existing_owned_child(monkeypatch):
    """When the child already exists and is owned, executor does NOT create."""
    existing = {
        "metadata": {
            "name": "s-v07-t2",
            "ownerReferences": [{"uid": "uid"}],
            "labels": {"aiperf.nvidia.com/sweep": "s"},
        },
        "status": {
            "phase": "Succeeded",
            "runEpoch": "1714000000",
            "runtimeRef": {"controllerHost": "h"},
        },
    }
    api = MagicMock()
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(return_value=existing)
    custom.create_namespaced_custom_object = AsyncMock()
    monkeypatch.setattr(
        "aiperf.sweep_controller.k8s_executor.CustomObjectsApi", lambda _api: custom
    )
    monkeypatch.setattr(
        "aiperf.sweep_controller.k8s_executor.ApiException",
        _NotFoundException,
    )

    executor = K8sChildJobExecutor(api=api, sweep=_sweep_cr(), with_trial_suffix=True)
    executor._wait_until_terminal = AsyncMock(return_value=None)
    executor._pull_summary_metrics = AsyncMock(return_value={})

    await executor.execute(_benchmark_run())
    custom.create_namespaced_custom_object.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_raises_on_name_conflict_with_unowned_child(monkeypatch):
    """If a child name slot is occupied by an UNOWNED AIPerfJob, raise."""
    foreign = {
        "metadata": {
            "name": "s-v07-t2",
            "ownerReferences": [{"uid": "different-uid"}],
            "labels": {},
        },
    }
    api = MagicMock()
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(return_value=foreign)
    custom.create_namespaced_custom_object = AsyncMock()
    monkeypatch.setattr(
        "aiperf.sweep_controller.k8s_executor.CustomObjectsApi", lambda _api: custom
    )
    monkeypatch.setattr(
        "aiperf.sweep_controller.k8s_executor.ApiException",
        _NotFoundException,
    )

    executor = K8sChildJobExecutor(api=api, sweep=_sweep_cr(), with_trial_suffix=True)

    with pytest.raises(ChildNameConflictError, match="not owned by this sweep"):
        await executor.execute(_benchmark_run())
    custom.create_namespaced_custom_object.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_waits_through_stale_child_deletion(monkeypatch):
    """A same-named child mid-cascade-delete is waited out, then the new one is created.

    Reproduces the delete-then-recreate-with-same-name race: the new
    sweep-controller polls until the apiserver has removed the foreign
    deleting child, then creates ours.
    """
    foreign_deleting = {
        "metadata": {
            "name": "s-v07-t2",
            "ownerReferences": [{"uid": "old-sweep-uid"}],
            "labels": {"aiperf.nvidia.com/sweep": "s"},
            "deletionTimestamp": "2026-04-27T20:00:00Z",
        },
    }
    api = MagicMock()
    custom = MagicMock()
    # Reads: foreign-deleting twice (poll), then 404 (gone), then Succeeded post-watch.
    custom.get_namespaced_custom_object = AsyncMock(
        side_effect=[
            foreign_deleting,
            foreign_deleting,
            _NotFoundException(404),
            {
                "metadata": {
                    "name": "s-v07-t2",
                    "ownerReferences": [{"uid": "uid"}],
                    "labels": {"aiperf.nvidia.com/sweep": "s"},
                },
                "status": {
                    "phase": "Succeeded",
                    "runEpoch": "1714000000",
                    "runtimeRef": {"controllerHost": "h"},
                },
            },
        ]
    )
    custom.create_namespaced_custom_object = AsyncMock(
        return_value={
            "metadata": {
                "name": "s-v07-t2",
                "ownerReferences": [{"uid": "uid"}],
                "labels": {"aiperf.nvidia.com/sweep": "s"},
            },
        }
    )
    monkeypatch.setattr(
        "aiperf.sweep_controller.k8s_executor.CustomObjectsApi", lambda _api: custom
    )
    monkeypatch.setattr(
        "aiperf.sweep_controller.k8s_executor.ApiException",
        _NotFoundException,
    )

    executor = K8sChildJobExecutor(api=api, sweep=_sweep_cr(), with_trial_suffix=True)
    executor._wait_until_terminal = AsyncMock(return_value=None)
    executor._pull_summary_metrics = AsyncMock(return_value={})

    result = await executor.execute(_benchmark_run())

    custom.create_namespaced_custom_object.assert_awaited_once()
    assert result.success is True
    # 4 reads = 2 polls (foreign_deleting) + 1 (404, free slot) + 1 (post-terminal).
    assert custom.get_namespaced_custom_object.await_count == 4


@pytest.mark.asyncio
async def test_execute_raises_when_stale_child_deletion_exceeds_deadline(monkeypatch):
    """A foreign child with deletionTimestamp that never disappears trips the
    deadline-exceeded conflict error (stuck-finalizer signal).
    """
    from aiperf.operator.environment import OperatorEnvironment

    foreign_deleting = {
        "metadata": {
            "name": "s-v07-t2",
            "ownerReferences": [{"uid": "old-sweep-uid"}],
            "labels": {"aiperf.nvidia.com/sweep": "s"},
            "deletionTimestamp": "2026-04-27T20:00:00Z",
        },
    }
    api = MagicMock()
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(return_value=foreign_deleting)
    custom.create_namespaced_custom_object = AsyncMock()
    monkeypatch.setattr(
        "aiperf.sweep_controller.k8s_executor.CustomObjectsApi", lambda _api: custom
    )
    monkeypatch.setattr(
        "aiperf.sweep_controller.k8s_executor.ApiException",
        _NotFoundException,
    )
    # Tighten the deadline so the test runs fast under looptime.
    monkeypatch.setattr(
        OperatorEnvironment.SWEEP_CONTROLLER,
        "STALE_CHILD_DELETION_TIMEOUT_SECONDS",
        0.001,
    )
    monkeypatch.setattr(
        OperatorEnvironment.SWEEP_CONTROLLER,
        "STALE_CHILD_POLL_INTERVAL_SECONDS",
        0.001,
    )

    executor = K8sChildJobExecutor(api=api, sweep=_sweep_cr(), with_trial_suffix=True)

    with pytest.raises(ChildNameConflictError, match="still mid-deletion"):
        await executor.execute(_benchmark_run())
    custom.create_namespaced_custom_object.assert_not_awaited()


@pytest.mark.asyncio
async def test_collect_run_result_from_failed_child():
    """Collect path: failed phase -> success=False with reason."""
    executor = K8sChildJobExecutor(api=None, sweep=_sweep_cr(), with_trial_suffix=True)
    failed_child = {
        "metadata": {"name": "s-v07-t2"},
        "status": {"phase": "Failed", "message": "endpoint timeout"},
    }
    result = await executor._collect_run_result(failed_child, _benchmark_run())
    assert result.success is False
    assert "endpoint timeout" in result.error


# ===========================================================================
# Adversarial regression-lock for second-pass fix (commit 793260d7b):
# `_collect_run_result` no longer classifies a Succeeded child with empty
# `status.summary` as a failure. The summary write may race the phase
# transition; the operator wrote terminal-success, that's what matters.
# ===========================================================================


@pytest.mark.asyncio
async def test_collect_run_result_succeeded_empty_summary_is_success():
    """Succeeded child + empty status.summary → RunResult(success=True, metrics={}).

    Previously this was misclassified as failure, which tripped failure_policy
    on a write race between status.phase and status.summary.
    """
    executor = K8sChildJobExecutor(api=None, sweep=_sweep_cr(), with_trial_suffix=True)
    succeeded_empty_summary = {
        "metadata": {"name": "s-v0007-t02"},
        "status": {"phase": "Succeeded", "summary": {}},
    }
    result = await executor._collect_run_result(
        succeeded_empty_summary, _benchmark_run()
    )
    assert result.success is True
    assert result.summary_metrics == {}


@pytest.mark.asyncio
async def test_collect_run_result_completed_empty_summary_is_success():
    """Completed (alias for Succeeded) + empty summary is also success."""
    executor = K8sChildJobExecutor(api=None, sweep=_sweep_cr(), with_trial_suffix=True)
    completed_empty = {
        "metadata": {"name": "s-v0007-t02"},
        "status": {"phase": "Completed"},  # no summary key at all
    }
    result = await executor._collect_run_result(completed_empty, _benchmark_run())
    assert result.success is True
    assert result.summary_metrics == {}
