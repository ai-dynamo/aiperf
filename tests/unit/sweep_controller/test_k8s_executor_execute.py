# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from pytest import param

from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config import BenchmarkConfig, BenchmarkRun, SweepVariation
from aiperf.orchestrator.models import RunResult
from aiperf.sweep_controller.k8s_executor import (
    RUN_IDENTITY_ANNOTATION,
    ChildNameConflictError,
    K8sChildJobExecutor,
)


def _sweep_cr() -> dict:
    return {
        "metadata": {"name": "s", "namespace": "ns", "uid": "uid"},
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


def _successful_summary() -> dict[str, JsonMetricResult]:
    return {"request_count": JsonMetricResult(unit="requests", avg=1.0)}


class _NotFoundException(Exception):
    """Mimics kubernetes_asyncio.client.ApiException(status=404)."""

    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__(f"ApiException({status})")


def _owned_metadata(
    name: str,
    *,
    child_uid: str = "child-uid",
    sweep_run_epoch: str | None = None,
) -> dict[str, object]:
    labels = {
        "aiperf.nvidia.com/sweep": "s",
        "aiperf.nvidia.com/sweep-uid": "uid",
    }
    if sweep_run_epoch is not None:
        labels["aiperf.nvidia.com/sweep-run-epoch"] = sweep_run_epoch
    return {
        "name": name,
        "uid": child_uid,
        "ownerReferences": [
            {
                "apiVersion": "aiperf.nvidia.com/v1alpha1",
                "kind": "AIPerfSweep",
                "name": "s",
                "uid": "uid",
                "controller": True,
            }
        ],
        "labels": labels,
    }


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
                "metadata": _owned_metadata("s-v07-t2"),
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
            "metadata": _owned_metadata("s-v07-t2"),
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
    executor._pull_summary_metrics = AsyncMock(return_value=_successful_summary())

    result = await executor.execute(_benchmark_run())

    custom.create_namespaced_custom_object.assert_awaited_once()
    created_body = custom.create_namespaced_custom_object.await_args.kwargs["body"]
    expected_identity = executor._run_identity(created_body["spec"])
    assert (
        created_body["metadata"]["annotations"][RUN_IDENTITY_ANNOTATION]
        == expected_identity
    )
    assert result.success is True
    assert result.summary_metrics == _successful_summary()
    assert result.label == "run_0002"


@pytest.mark.asyncio
async def test_execute_resumes_existing_owned_child_and_rebuilds_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A fresh emptyDir rebuilds lineage when an existing child is resumed."""
    existing = {
        "metadata": _owned_metadata("s-v07-t2", sweep_run_epoch="1713999999"),
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

    executor = K8sChildJobExecutor(
        api=api,
        sweep=_sweep_cr(),
        with_trial_suffix=True,
        base_dir=tmp_path,
        sweep_run_epoch="1713999999",
    )
    run = _benchmark_run()
    existing["metadata"]["annotations"] = {
        RUN_IDENTITY_ANNOTATION: executor._run_identity(executor._build_child_spec(run))
    }
    executor._wait_until_terminal = AsyncMock(return_value=None)
    executor._pull_summary_metrics = AsyncMock(return_value=_successful_summary())

    await executor.execute(run)
    custom.create_namespaced_custom_object.assert_not_awaited()
    marker = tmp_path / "ns" / "s-v07-t2" / "sweep.json"
    assert orjson.loads(marker.read_bytes()) == {
        "sweep_name": "s",
        "variation_index": 7,
        "variation_label": "c=64",
        "trial_index": 2,
        "sweep_run_epoch": "1713999999",
        "child_run_epoch": "1714000000",
    }


@pytest.mark.asyncio
async def test_recover_terminal_results_reads_only_exact_sweep_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sweep = _sweep_cr()
    sweep["status"] = {
        "aggregate": {
            "children": {
                "children": [
                    {
                        "name": "s-v07-t2",
                        "variation_label": "c=64",
                        "variation_values": '{"phases.profiling.concurrency":64}',
                        "label": "run_0003",
                    }
                ]
            }
        }
    }
    owned_terminal = {
        "metadata": {
            "name": "s-v07-t2",
            "uid": "child-uid-07",
            "namespace": "ns",
            "ownerReferences": [
                {
                    "apiVersion": "aiperf.nvidia.com/v1alpha1",
                    "kind": "AIPerfSweep",
                    "name": "s",
                    "uid": "uid",
                    "controller": True,
                }
            ],
            "labels": {
                "aiperf.nvidia.com/sweep": "s",
                "aiperf.nvidia.com/sweep-uid": "uid",
                "aiperf.nvidia.com/sweep-run-epoch": "1713999999",
                "aiperf.nvidia.com/variation-index": "07",
                "aiperf.nvidia.com/variation-label": "c-64",
                "aiperf.nvidia.com/trial-index": "2",
            },
            "annotations": {
                "aiperf.nvidia.com/variation-values": (
                    '{"phases.profiling.concurrency":64}'
                )
            },
        },
        "status": {
            "phase": "Completed",
            "runEpoch": "1714000000",
            "summary": {"request_count": {"unit": "requests", "avg": 1.0}},
        },
    }
    foreign_terminal = {
        **owned_terminal,
        "metadata": {
            **owned_terminal["metadata"],
            "name": "s-v08-t2",
            "ownerReferences": [
                {
                    "apiVersion": "aiperf.nvidia.com/v1alpha1",
                    "kind": "AIPerfSweep",
                    "name": "s",
                    "uid": "other-uid",
                    "controller": True,
                }
            ],
        },
    }
    owned_running = {
        **owned_terminal,
        "metadata": {**owned_terminal["metadata"], "name": "s-v09-t2"},
        "status": {"phase": "Running"},
    }
    owned_failed = {
        **owned_terminal,
        "metadata": {
            **owned_terminal["metadata"],
            "name": "s-v08-t2",
            "labels": {
                **owned_terminal["metadata"]["labels"],
                "aiperf.nvidia.com/variation-index": "08",
                "aiperf.nvidia.com/variation-label": "c-128",
            },
        },
        "status": {"phase": "Failed", "error": "worker failed"},
    }
    owned_cancelled = {
        **owned_terminal,
        "metadata": {
            **owned_terminal["metadata"],
            "name": "s-v09-t2",
            "labels": {
                **owned_terminal["metadata"]["labels"],
                "aiperf.nvidia.com/variation-index": "09",
                "aiperf.nvidia.com/variation-label": "c-256",
            },
        },
        "status": {"phase": "Cancelled", "error": "user cancelled"},
    }
    wrong_epoch_terminal = {
        **owned_terminal,
        "metadata": {
            **owned_terminal["metadata"],
            "name": "s-v10-t2",
            "labels": {
                **owned_terminal["metadata"]["labels"],
                "aiperf.nvidia.com/sweep-run-epoch": "1713999998",
            },
        },
    }
    non_controller_terminal = {
        **owned_terminal,
        "metadata": {
            **owned_terminal["metadata"],
            "name": "s-v11-t2",
            "ownerReferences": [
                "malformed-owner",
                {
                    "apiVersion": "aiperf.nvidia.com/v1alpha1",
                    "kind": "AIPerfSweep",
                    "name": "s",
                    "uid": "uid",
                    "controller": False,
                },
            ],
        },
    }
    custom = MagicMock()
    custom.list_namespaced_custom_object = AsyncMock(
        return_value={
            "items": [
                foreign_terminal,
                owned_running,
                wrong_epoch_terminal,
                non_controller_terminal,
                owned_cancelled,
                owned_failed,
                owned_terminal,
            ]
        }
    )
    custom.create_namespaced_custom_object = AsyncMock()
    custom.patch_namespaced_custom_object = AsyncMock()
    monkeypatch.setattr(
        "aiperf.sweep_controller.k8s_executor.CustomObjectsApi", lambda _api: custom
    )

    executor = K8sChildJobExecutor(
        api=MagicMock(),
        sweep=sweep,
        with_trial_suffix=True,
        sweep_run_epoch="1713999999",
    )
    plan = SimpleNamespace(
        variations=[
            SimpleNamespace(
                index=7,
                label="c=64",
                values={"phases.profiling.concurrency": 64},
            )
        ]
    )

    recovered = await executor.recover_terminal_results(plan)
    recovered_again = await executor.recover_terminal_results(plan)

    assert recovered[0] == RunResult(
        label="run_0003",
        success=True,
        summary_metrics=_successful_summary(),
        variation_label="c=64",
        variation_values={"phases.profiling.concurrency": 64},
        variation_index=7,
        trial_index=2,
    )
    assert [
        (result.success, result.was_cancelled, result.error) for result in recovered[1:]
    ] == [
        (False, False, "worker failed"),
        (False, True, "user cancelled"),
    ]
    assert recovered_again == []
    assert [ref.name for ref in executor.terminal_children] == [
        "s-v07-t2",
        "s-v08-t2",
        "s-v09-t2",
    ]
    assert [ref.status for ref in executor.terminal_children] == [
        "Succeeded",
        "Failed",
        "Cancelled",
    ]
    assert executor.terminal_children[0].child_run_epoch == "1714000000"
    custom.create_namespaced_custom_object.assert_not_awaited()
    custom.patch_namespaced_custom_object.assert_not_awaited()
    assert custom.list_namespaced_custom_object.await_args.kwargs["label_selector"] == (
        "aiperf.nvidia.com/sweep=s,"
        "aiperf.nvidia.com/sweep-uid=uid,"
        "aiperf.nvidia.com/sweep-run-epoch=1713999999"
    )


@pytest.mark.asyncio
async def test_recover_terminal_results_skips_live_summary_settle_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = {
        "metadata": {
            "name": "s-v07-t2",
            "uid": "child-uid-07",
            "namespace": "ns",
            "ownerReferences": [
                {
                    "apiVersion": "aiperf.nvidia.com/v1alpha1",
                    "kind": "AIPerfSweep",
                    "name": "s",
                    "uid": "uid",
                    "controller": True,
                }
            ],
            "labels": {
                "aiperf.nvidia.com/sweep": "s",
                "aiperf.nvidia.com/sweep-uid": "uid",
                "aiperf.nvidia.com/sweep-run-epoch": "1713999999",
                "aiperf.nvidia.com/variation-index": "07",
                "aiperf.nvidia.com/variation-label": "c-64",
                "aiperf.nvidia.com/trial-index": "2",
            },
        },
        "status": {"phase": "Completed"},
    }
    custom = MagicMock()
    custom.list_namespaced_custom_object = AsyncMock(return_value={"items": [child]})
    sleep = AsyncMock()
    monkeypatch.setattr(
        "aiperf.sweep_controller.k8s_executor.CustomObjectsApi", lambda _api: custom
    )
    monkeypatch.setattr("aiperf.sweep_controller.k8s_executor.asyncio.sleep", sleep)
    executor = K8sChildJobExecutor(
        api=MagicMock(),
        sweep=_sweep_cr(),
        with_trial_suffix=True,
        sweep_run_epoch="1713999999",
    )
    fetch_summary = AsyncMock(return_value={})
    executor._fetch_summary_from_operator = fetch_summary  # type: ignore[method-assign]
    plan = SimpleNamespace(variations=[])

    recovered = await executor.recover_terminal_results(plan)

    sleep.assert_not_awaited()
    fetch_summary.assert_awaited_once_with(child, retry=False)
    assert len(recovered) == 1
    assert recovered[0].success is False
    assert recovered[0].error == (
        "No metrics found in child status/artifacts - run may have failed to complete"
    )


@pytest.mark.asyncio
async def test_collect_recovery_results_caps_summary_concurrency() -> None:
    from aiperf.sweep_controller import k8s_executor as executor_module

    limit = executor_module._RECOVERY_SUMMARY_CONCURRENCY
    candidates = [
        SimpleNamespace(child={}, label=f"run_{index:04d}")
        for index in range(limit + 3)
    ]
    active = 0
    max_active = 0
    started = 0
    release = asyncio.Event()

    async def collect(
        _child: dict[str, object],
        *,
        label: str,
        artifacts_path: Path | None,
        settle_summary: bool,
    ) -> RunResult:
        nonlocal active, max_active, started
        assert artifacts_path is None
        assert settle_summary is False
        active += 1
        started += 1
        max_active = max(max_active, active)
        if started == limit:
            release.set()
        await release.wait()
        active -= 1
        return RunResult(label=label, success=False)

    executor = K8sChildJobExecutor(
        api=MagicMock(), sweep=_sweep_cr(), with_trial_suffix=True
    )
    executor._collect_child_result = collect  # type: ignore[method-assign]

    recovered = await executor._collect_recovery_results(candidates)  # type: ignore[arg-type]

    assert len(recovered) == limit + 3
    assert max_active == limit


@pytest.mark.asyncio
@pytest.mark.parametrize("existing_identity", [None, "different-contract"])
async def test_execute_rejects_owned_child_with_unverified_contract(
    monkeypatch: pytest.MonkeyPatch,
    existing_identity: str | None,
) -> None:
    """Ownership alone cannot make a deterministic child name safe to reuse."""
    metadata = _owned_metadata("s-v07-t2")
    if existing_identity is not None:
        metadata["annotations"] = {RUN_IDENTITY_ANNOTATION: existing_identity}
    existing = {"metadata": metadata, "status": {"phase": "Succeeded"}}
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(return_value=existing)
    custom.create_namespaced_custom_object = AsyncMock()
    monkeypatch.setattr(
        "aiperf.sweep_controller.k8s_executor.CustomObjectsApi", lambda _api: custom
    )

    executor = K8sChildJobExecutor(
        api=MagicMock(), sweep=_sweep_cr(), with_trial_suffix=True
    )

    with pytest.raises(ChildNameConflictError, match="execution contract") as exc_info:
        await executor.execute(_benchmark_run())
    message = str(exc_info.value)
    assert f"persisted identity={existing_identity or '<missing>'!r}" in message
    assert "planned identity=" in message
    assert "delete the stale child before retrying" in message
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
                "metadata": _owned_metadata("s-v07-t2"),
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
            "metadata": _owned_metadata("s-v07-t2"),
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
    executor._pull_summary_metrics = AsyncMock(return_value=_successful_summary())

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
async def test_execute_records_terminal_child_run_epoch() -> None:
    """Terminal child manifest entries use AIPerfJob status.runEpoch."""
    executor = K8sChildJobExecutor(
        api=None,
        sweep=_sweep_cr(),
        with_trial_suffix=True,
        sweep_run_epoch="1714000000",
    )
    terminal_child = {
        "metadata": _owned_metadata("s-v07-t2", sweep_run_epoch="1714000000"),
        "status": {"phase": "Succeeded", "runEpoch": "1714000042"},
    }
    status_writer = MagicMock()
    status_writer.current_cell = AsyncMock()
    status_writer.partial_children = AsyncMock()
    executor._status_writer = status_writer
    executor._get_or_create = AsyncMock(  # type: ignore[method-assign]
        return_value={"metadata": terminal_child["metadata"]}
    )
    executor._wait_until_terminal = AsyncMock(return_value=None)  # type: ignore[method-assign]
    executor._try_read_child = AsyncMock(return_value=terminal_child)  # type: ignore[method-assign]
    executor._collect_run_result = AsyncMock(  # type: ignore[method-assign]
        return_value=MagicMock(success=True)
    )

    await executor.execute(_benchmark_run())

    children = status_writer.partial_children.await_args.kwargs["children"]
    assert children[0]["child_run_epoch"] == "1714000042"
    assert executor.terminal_children[0].child_run_epoch == "1714000042"


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
    assert result.was_cancelled is False
    assert "endpoint timeout" in result.error


@pytest.mark.asyncio
async def test_collect_run_result_succeeded_empty_summary_is_failure() -> None:
    """Exhausted status/PVC recovery cannot produce a successful sweep cell."""
    executor = K8sChildJobExecutor(api=None, sweep=_sweep_cr(), with_trial_suffix=True)
    succeeded_empty_summary = {
        "metadata": {"name": "s-v0007-t02"},
        "status": {"phase": "Succeeded", "summary": {}},
    }
    executor._pull_summary_metrics = AsyncMock(return_value={})  # type: ignore[method-assign]
    result = await executor._collect_run_result(
        succeeded_empty_summary, _benchmark_run()
    )
    assert result.success is False
    assert result.summary_metrics == {}
    assert result.was_cancelled is False
    assert result.error == (
        "No metrics found in child status/artifacts - run may have failed to complete"
    )


@pytest.mark.asyncio
async def test_collect_run_result_completed_empty_summary_is_failure() -> None:
    """Completed is subject to the same canonical metrics contract as Succeeded."""
    executor = K8sChildJobExecutor(api=None, sweep=_sweep_cr(), with_trial_suffix=True)
    completed_empty = {
        "metadata": {"name": "s-v0007-t02"},
        "status": {"phase": "Completed"},  # no summary key at all
    }
    executor._pull_summary_metrics = AsyncMock(return_value={})  # type: ignore[method-assign]
    result = await executor._collect_run_result(completed_empty, _benchmark_run())
    assert result.success is False
    assert result.summary_metrics == {}
    assert result.error == (
        "No metrics found in child status/artifacts - run may have failed to complete"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metrics", "expected_error"),
    [
        param(
            {"request_latency": JsonMetricResult(unit="ms", avg=10.0)},
            "No requests completed",
            id="missing-request-count",
        ),
        param(
            {"request_count": JsonMetricResult(unit="requests", avg=0.0)},
            "No requests completed",
            id="zero-request-count",
        ),
        param(
            {
                "request_count": JsonMetricResult(unit="requests", avg=0.0),
                "error_request_count": JsonMetricResult(unit="requests", avg=17.0),
            },
            "All 17 requests failed",
            id="all-requests-failed",
        ),
        param(
            {
                "request_count": JsonMetricResult(unit="requests", avg=0.0),
                "error_request_count": JsonMetricResult(unit="requests"),
            },
            "No requests completed",
            id="missing-error-count-average",
        ),
    ],
)  # fmt: skip
async def test_collect_run_result_rejects_missing_or_zero_completed_requests(
    metrics: dict[str, JsonMetricResult], expected_error: str
) -> None:
    """Kubernetes children use the canonical local run-success contract."""
    executor = K8sChildJobExecutor(api=None, sweep=_sweep_cr(), with_trial_suffix=True)
    executor._pull_summary_metrics = AsyncMock(return_value=metrics)  # type: ignore[method-assign]

    result = await executor._collect_run_result(
        {"status": {"phase": "Completed"}}, _benchmark_run()
    )

    assert result.success is False
    assert result.was_cancelled is False
    assert result.error == expected_error


@pytest.mark.asyncio
async def test_collect_run_result_positive_request_count_is_success() -> None:
    """A recovered summary with completed requests remains successful."""
    metrics = _successful_summary()
    executor = K8sChildJobExecutor(api=None, sweep=_sweep_cr(), with_trial_suffix=True)
    executor._pull_summary_metrics = AsyncMock(return_value=metrics)  # type: ignore[method-assign]

    result = await executor._collect_run_result(
        {"status": {"phase": "Completed"}}, _benchmark_run()
    )

    assert result.success is True
    assert result.summary_metrics == metrics
    assert result.was_cancelled is False


@pytest.mark.asyncio
async def test_collect_run_result_cancelled_preserves_cancellation_semantics() -> None:
    """A stale child error cannot turn a Cancelled terminal phase into failure."""
    executor = K8sChildJobExecutor(api=None, sweep=_sweep_cr(), with_trial_suffix=True)
    child = {
        "status": {
            "phase": "Cancelled",
            "error": "stale monitor failure before cancellation",
        }
    }

    result = await executor._collect_run_result(child, _benchmark_run())

    assert result.success is False
    assert result.was_cancelled is True
    assert result.error == "stale monitor failure before cancellation"
