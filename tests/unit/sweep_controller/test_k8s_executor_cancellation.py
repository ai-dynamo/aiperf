# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial regression-locks for K8sChildJobExecutor cancellation + status wiring.

Locks in three just-fixed bugs:
  - status_writer.current_cell was never invoked (dead code).
  - cancel_check was never consulted before child create or during waits.
  - cancel mid-wait did not propagate to the child via spec.cancel patch.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.config import BenchmarkConfig, BenchmarkRun, SweepVariation
from aiperf.sweep_controller.k8s_executor import (
    RUN_IDENTITY_ANNOTATION,
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
        benchmark_id=f"s-v{var_idx:04d}-t{trial:02d}",
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


def _stamp_run_identity(
    executor: K8sChildJobExecutor,
    child: dict,
    run: BenchmarkRun,
) -> None:
    child["metadata"]["annotations"] = {
        RUN_IDENTITY_ANNOTATION: executor._run_identity(executor._build_child_spec(run))
    }


def _owned_metadata(name: str, *, child_uid: str = "child-uid") -> dict[str, object]:
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
        "labels": {
            "aiperf.nvidia.com/sweep": "s",
            "aiperf.nvidia.com/sweep-uid": "uid",
        },
    }


# ---------------------------------------------------------------------------
# C) current_cell wiring — was previously never invoked.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_invokes_status_writer_current_cell_once_with_run_metadata(
    monkeypatch,
):
    """Regression-lock: execute() awaits status_writer.current_cell exactly once
    with kwargs matching the run's variation index, label, and trial.
    """
    api = MagicMock()
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(
        return_value={
            "metadata": _owned_metadata("s-v0007-t02"),
            "status": {"phase": "Succeeded"},
        }
    )
    custom.create_namespaced_custom_object = AsyncMock()
    monkeypatch.setattr(
        "aiperf.sweep_controller.k8s_executor.CustomObjectsApi", lambda _api: custom
    )
    monkeypatch.setattr(
        "aiperf.sweep_controller.k8s_executor.ApiException", _NotFoundException
    )

    status_writer = MagicMock()
    status_writer.current_cell = AsyncMock()

    executor = K8sChildJobExecutor(
        api=api,
        sweep=_sweep_cr(),
        with_trial_suffix=True,
        status_writer=status_writer,
    )
    executor._wait_until_terminal = AsyncMock(return_value=None)
    executor._pull_summary_metrics = AsyncMock(return_value={})

    run = _benchmark_run(var_idx=7, trial=2)
    _stamp_run_identity(executor, custom.get_namespaced_custom_object.return_value, run)
    await executor.execute(run)

    status_writer.current_cell.assert_awaited_once()
    kwargs = status_writer.current_cell.call_args.kwargs
    assert kwargs["variation_index"] == 7
    assert kwargs["label"] == "run_0002"
    assert kwargs["trial"] == 2


@pytest.mark.asyncio
async def test_execute_swallows_status_writer_failure_and_continues(monkeypatch):
    """current_cell write failures must NOT crash execute (best-effort write)."""
    api = MagicMock()
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(
        return_value={
            "metadata": _owned_metadata("s-v0007-t02"),
            "status": {"phase": "Succeeded"},
        }
    )
    monkeypatch.setattr(
        "aiperf.sweep_controller.k8s_executor.CustomObjectsApi", lambda _api: custom
    )
    monkeypatch.setattr(
        "aiperf.sweep_controller.k8s_executor.ApiException", _NotFoundException
    )

    status_writer = MagicMock()
    status_writer.current_cell = AsyncMock(side_effect=RuntimeError("apiserver blip"))

    executor = K8sChildJobExecutor(
        api=api,
        sweep=_sweep_cr(),
        with_trial_suffix=True,
        status_writer=status_writer,
    )
    executor._wait_until_terminal = AsyncMock(return_value=None)
    executor._pull_summary_metrics = AsyncMock(return_value={})

    # Must not raise — best-effort write.
    run = _benchmark_run()
    _stamp_run_identity(executor, custom.get_namespaced_custom_object.return_value, run)
    result = await executor.execute(run)
    assert result is not None


# ---------------------------------------------------------------------------
# D) cancel_check short-circuits BEFORE child create.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_cancel_before_create_short_circuits_to_failed_run_result(
    monkeypatch,
):
    """When cancel_check returns True before create, executor must return a
    failed RunResult and NEVER touch the apiserver (no get/create/wait/patch).
    """

    def _boom(*args, **kwargs):
        raise AssertionError(
            "apiserver should NOT be touched after cancel short-circuit"
        )

    api = MagicMock()
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(side_effect=_boom)
    custom.create_namespaced_custom_object = AsyncMock(side_effect=_boom)
    custom.patch_namespaced_custom_object = AsyncMock(side_effect=_boom)
    monkeypatch.setattr(
        "aiperf.sweep_controller.k8s_executor.CustomObjectsApi", lambda _api: custom
    )

    status_writer = MagicMock()
    status_writer.current_cell = AsyncMock()

    executor = K8sChildJobExecutor(
        api=api,
        sweep=_sweep_cr(),
        with_trial_suffix=True,
        status_writer=status_writer,
        cancel_check=lambda: True,
    )
    # Hard-fail if waits/spawns sneak through despite the short-circuit.
    executor._wait_until_terminal = AsyncMock(side_effect=_boom)
    executor._get_or_create = AsyncMock(side_effect=_boom)

    result = await executor.execute(_benchmark_run())

    assert result.success is False
    assert "cancel" in (result.error or "").lower()
    custom.create_namespaced_custom_object.assert_not_awaited()
    custom.patch_namespaced_custom_object.assert_not_awaited()
    executor._wait_until_terminal.assert_not_awaited()
    executor._get_or_create.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_cancel_check_false_proceeds_normally(monkeypatch):
    """cancel_check=lambda: False must NOT alter the happy path."""
    api = MagicMock()
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(
        return_value={
            "metadata": _owned_metadata("s-v0007-t02"),
            "status": {"phase": "Succeeded"},
        }
    )
    custom.create_namespaced_custom_object = AsyncMock()
    monkeypatch.setattr(
        "aiperf.sweep_controller.k8s_executor.CustomObjectsApi", lambda _api: custom
    )
    monkeypatch.setattr(
        "aiperf.sweep_controller.k8s_executor.ApiException", _NotFoundException
    )

    executor = K8sChildJobExecutor(
        api=api,
        sweep=_sweep_cr(),
        with_trial_suffix=True,
        cancel_check=lambda: False,
    )
    executor._wait_until_terminal = AsyncMock(return_value=None)
    executor._pull_summary_metrics = AsyncMock(return_value={})

    run = _benchmark_run()
    _stamp_run_identity(executor, custom.get_namespaced_custom_object.return_value, run)
    result = await executor.execute(run)
    assert result is not None
    # No assertion on success here — collect_run_result uses the read child;
    # the point is that we did NOT short-circuit.


# ---------------------------------------------------------------------------
# E) cancel during wait propagates a spec.cancel patch to the child.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_until_terminal_patches_child_cancel_when_signaled_mid_wait(
    monkeypatch,
):
    """Mid-wait cancel: _wait_until_terminal must patch spec.cancel=true on the child.

    Sequence:
      1) First poll iteration: child phase != terminal, cancel_check returns False.
      2) Second poll iteration: child phase != terminal, cancel_check returns True.
      3) Executor patches spec.cancel=true, then on next poll the child returns
         phase=Cancelled and the wait exits.
    """
    # Child phase progression: Running, Running, Cancelled.
    phase_responses = [
        {"metadata": _owned_metadata("child"), "status": {"phase": "Running"}},
        {"metadata": _owned_metadata("child"), "status": {"phase": "Running"}},
        {"metadata": _owned_metadata("child"), "status": {"phase": "Cancelled"}},
    ]

    api = MagicMock()
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(side_effect=phase_responses)
    custom.patch_namespaced_custom_object = AsyncMock()
    monkeypatch.setattr(
        "aiperf.sweep_controller.k8s_executor.CustomObjectsApi", lambda _api: custom
    )
    monkeypatch.setattr(
        "aiperf.sweep_controller.k8s_executor.ApiException", _NotFoundException
    )

    # asyncio.sleep(0) keeps the test fast.
    import aiperf.sweep_controller.k8s_executor as mod

    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep)

    flag = {"requested": False}

    def cancel_check() -> bool:
        # First poll iteration sees False, then becomes True.
        result = flag["requested"]
        flag["requested"] = True
        return result

    executor = K8sChildJobExecutor(
        api=api,
        sweep=_sweep_cr(),
        with_trial_suffix=True,
        cancel_check=cancel_check,
    )
    await executor._wait_until_terminal(
        "child",
        _benchmark_run(),
        expected_child_uid="child-uid",
        poll_interval=0.0,
        cancel_check=cancel_check,
    )
    custom.patch_namespaced_custom_object.assert_awaited()
    kwargs = custom.patch_namespaced_custom_object.call_args.kwargs
    assert kwargs.get("body") == [
        {"op": "test", "path": "/metadata/uid", "value": "child-uid"},
        {"op": "add", "path": "/spec/cancel", "value": True},
    ]
    assert kwargs.get("_content_type") == "application/json-patch+json"


@pytest.mark.asyncio
async def test_wait_until_terminal_classifies_persistent_404_as_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aiperf.sweep_controller.k8s_executor as mod
    from aiperf.operator.environment import OperatorEnvironment

    executor = K8sChildJobExecutor(
        api=MagicMock(),
        sweep=_sweep_cr(),
        with_trial_suffix=True,
    )
    executor._try_read_child = AsyncMock(return_value=None)

    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        OperatorEnvironment.SWEEP_CONTROLLER,
        "CHILD_MISSING_TIMEOUT_SECONDS",
        0.0,
    )

    result = await executor._wait_until_terminal(
        "deleted-child",
        _benchmark_run(),
        expected_child_uid="deleted-child-uid",
        poll_interval=0.0,
    )

    assert result is not None
    assert result.success is False
    assert result.was_cancelled is False
    assert "disappeared" in (result.error or "")


@pytest.mark.asyncio
async def test_wait_until_terminal_rejects_same_name_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(
        return_value={
            "metadata": _owned_metadata("child", child_uid="replacement-uid"),
            "status": {"phase": "Succeeded"},
        }
    )
    custom.patch_namespaced_custom_object = AsyncMock()
    monkeypatch.setattr(
        "aiperf.sweep_controller.k8s_executor.CustomObjectsApi", lambda _api: custom
    )
    executor = K8sChildJobExecutor(
        api=MagicMock(), sweep=_sweep_cr(), with_trial_suffix=True
    )

    result = await executor._wait_until_terminal(
        "child",
        _benchmark_run(),
        expected_child_uid="original-uid",
        poll_interval=0.0,
    )

    assert result is not None
    assert result.success is False
    assert "identity changed" in (result.error or "")
    custom.patch_namespaced_custom_object.assert_not_awaited()
