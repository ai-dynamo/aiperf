# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial regression-locks for sweep_controller.main._poll_cancel_flag.

The poller is a best-effort background task: it must
  - set flag['requested']=True when the parent CR's spec.cancel transitions True,
  - swallow apiserver hiccups (ApiException) without crashing the controller,
  - exit immediately once the flag is set.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.sweep_controller.main import _poll_cancel_flag


def _cr_with_cancel(value: bool) -> dict:
    return {"spec": {"cancel": value}}


@pytest.mark.asyncio
async def test_poll_cancel_flag_sets_requested_when_spec_cancel_true():
    """Polling sets flag['requested']=True the first tick where spec.cancel=True."""
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(
        side_effect=[
            _cr_with_cancel(False),
            _cr_with_cancel(False),
            _cr_with_cancel(True),
        ]
    )

    flag: dict[str, bool] = {"requested": False}
    await asyncio.wait_for(
        _poll_cancel_flag(custom, namespace="ns", name="s", flag=flag, interval=0.0),
        timeout=2.0,
    )
    assert flag["requested"] is True
    assert custom.get_namespaced_custom_object.await_count == 3


@pytest.mark.asyncio
async def test_poll_cancel_flag_survives_apiserver_exception():
    """ApiException must NOT crash the poller; flag stays False, loop continues."""

    class _ApiException(Exception):
        def __init__(self, status: int = 500) -> None:
            self.status = status
            super().__init__(f"ApiException({status})")

    # Raise an exception, then return False, then return True so we exit cleanly.
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock(
        side_effect=[
            _ApiException(500),
            _cr_with_cancel(False),
            _cr_with_cancel(True),
        ]
    )

    flag: dict[str, bool] = {"requested": False}
    await asyncio.wait_for(
        _poll_cancel_flag(custom, namespace="ns", name="s", flag=flag, interval=0.0),
        timeout=2.0,
    )
    # Eventually exits because the third tick returns cancel=True.
    assert flag["requested"] is True
    # Three calls: hiccup, False, True.
    assert custom.get_namespaced_custom_object.await_count == 3


@pytest.mark.asyncio
async def test_poll_cancel_flag_exits_immediately_if_already_requested():
    """Pre-set flag means the poller exits before the first apiserver hit."""
    custom = MagicMock()
    custom.get_namespaced_custom_object = AsyncMock()

    flag: dict[str, bool] = {"requested": True}
    await asyncio.wait_for(
        _poll_cancel_flag(custom, namespace="ns", name="s", flag=flag, interval=0.0),
        timeout=1.0,
    )
    custom.get_namespaced_custom_object.assert_not_awaited()


# ---------------------------------------------------------------------------
# Light end-to-end smoke for main(): verifies the cancel-poll background task
# is torn down in the finally block.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_main_cancels_poll_task_in_finally(monkeypatch, tmp_path):
    """main()'s try/finally must cancel the background cancel-poll task.

    Heavy deps are patched: k8s_client (yields a fake api), CustomObjectsApi
    (returns a minimal sweep CR), MultiRunOrchestrator.execute (returns []),
    SweepStatusWriter (no-op), K8sChildJobExecutor (no-op), build_strategy +
    aggregate_and_export + build_plan_from_sweep (no-ops). The aggregate-ready
    marker is pre-written so main() takes the marker-present branch.
    """
    import contextlib as _cl
    import importlib

    monkeypatch.setenv("AIPERF_SWEEP_NAME", "s")
    monkeypatch.setenv("AIPERF_SWEEP_NAMESPACE", "ns")
    monkeypatch.setenv("AIPERF_SWEEP_EPOCH", "1714069323")

    main_mod = importlib.import_module("aiperf.sweep_controller.main")

    # Pre-write the marker so the aggregation branch is skipped.
    monkeypatch.setattr(main_mod, "RESULTS_DIR", tmp_path)
    main_mod.write_aggregate_marker(tmp_path)

    # Capture the cancel-poll task so we can verify it gets cancelled.
    captured: dict[str, asyncio.Task] = {}
    real_create_task = asyncio.create_task

    def _capturing_create_task(coro, *args, **kwargs):
        task = real_create_task(coro, *args, **kwargs)
        # Only record the *first* task (the cancel-poll task is created first).
        captured.setdefault("task", task)
        return task

    # Make _poll_cancel_flag a hot loop we can quickly cancel.
    async def _fast_poll(*args, **kwargs) -> None:
        flag = kwargs["flag"]
        while not flag["requested"]:
            await asyncio.sleep(0)

    monkeypatch.setattr(main_mod, "_poll_cancel_flag", _fast_poll)
    monkeypatch.setattr(main_mod.asyncio, "create_task", _capturing_create_task)

    # Skip the idle-forever sleep at the end.
    async def _no_idle() -> None:
        return None

    # `_idle_until_terminated` was removed when the controller stopped
    # idling forever (now exits 0 on success / 1 on patch failure so the
    # JobSet completes and the parent CR's TTL reaper can fire). Nothing
    # to monkeypatch — the test's "skip the idle" intent is satisfied by
    # the new clean-exit semantics.
    _ = _no_idle  # intentionally unused — see the comment above

    # Patch the lazy imports inside main(): the function imports them itself
    # via from-imports, so patch the attributes on the source modules.
    @_cl.asynccontextmanager
    async def _fake_k8s_client():
        yield MagicMock()

    monkeypatch.setattr(
        "aiperf.kubernetes.client.k8s_client", _fake_k8s_client, raising=True
    )

    sweep_cr = {
        "metadata": {"name": "s", "namespace": "ns", "uid": "uid"},
        "spec": {
            "image": "x:latest",
            "sweep": {
                "type": "grid",
                "variables": {"benchmark.phases.profiling.concurrency": [1]},
            },
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
            "multiRun": {"numRuns": 1},
        },
    }

    fake_custom = MagicMock()
    fake_custom.get_namespaced_custom_object = AsyncMock(return_value=sweep_cr)
    monkeypatch.setattr(
        "kubernetes_asyncio.client.CustomObjectsApi", lambda _api: fake_custom
    )

    # Stub the heavy pieces.
    fake_plan = MagicMock()
    fake_plan.configs = []
    fake_plan.is_adaptive_search = False
    monkeypatch.setattr(
        "aiperf.sweep_controller.plan_builder.build_plan_from_sweep",
        lambda cr: fake_plan,
    )

    class _Strategy:
        def get_aggregate_path(self, base):
            return base / "aggregate"

    monkeypatch.setattr(
        "aiperf.cli_runner._strategy.build_strategy",
        lambda plan, logger: _Strategy(),
    )
    monkeypatch.setattr(
        "aiperf.cli_runner._aggregate.aggregate_and_export",
        lambda *a, **kw: None,
    )

    class _Orch:
        def __init__(self, base_dir):
            self.base_dir = base_dir

        async def execute(
            self, plan, executor, *, cancel_check=None, search_planner=None
        ):
            return []

    monkeypatch.setattr("aiperf.orchestrator.orchestrator.MultiRunOrchestrator", _Orch)

    class _Writer:
        def __init__(self, *args, **kwargs):
            pass

        async def aggregation_running(self):
            pass

        async def aggregation_complete(self, **kwargs):
            pass

        async def aggregation_failed(self, **kwargs):
            pass

        async def parent_running(self):
            pass

    monkeypatch.setattr(
        "aiperf.sweep_controller.status_writer.SweepStatusWriter", _Writer
    )

    class _Exec:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(
        "aiperf.sweep_controller.k8s_executor.K8sChildJobExecutor", _Exec
    )
    monkeypatch.setattr(
        "aiperf.sweep_controller.k8s_executor.needs_trial_suffix",
        lambda **kwargs: False,
    )

    rc = await main_mod.main()
    assert rc == 0
    # The cancel-poll task should have been cancelled in the finally block.
    poll_task = captured.get("task")
    assert poll_task is not None
    assert poll_task.done(), "cancel-poll task was not torn down"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("results", "case"),
    [
        pytest.param([], "before-child-results", id="before-child-results"),
        pytest.param(
            [
                SimpleNamespace(
                    label="cell-0",
                    success=True,
                    error=None,
                    variation_values={"index": 0},
                    variation_label="v0",
                    trial_index=0,
                    child_run_epoch="1714069324",
                )
            ],
            "after-partial-child-results",
            id="after-partial-child-results",
        ),
    ],
)  # fmt: skip
async def test_main_marks_cancelled_when_cancel_requested(
    monkeypatch,
    tmp_path,
    results,
    case: str,
):
    """Cancellation terminal phase wins while partial aggregation still runs."""
    import contextlib as _cl
    import importlib

    async def _run() -> tuple[int, dict]:
        monkeypatch.setenv("AIPERF_SWEEP_NAME", "s")
        monkeypatch.setenv("AIPERF_SWEEP_NAMESPACE", "ns")
        monkeypatch.setenv("AIPERF_SWEEP_EPOCH", "1714069323")

        main_mod = importlib.import_module("aiperf.sweep_controller.main")
        monkeypatch.setattr(main_mod, "RESULTS_DIR", tmp_path)

        async def _request_cancel(*args, **kwargs) -> None:
            kwargs["flag"]["requested"] = True

        monkeypatch.setattr(main_mod, "_poll_cancel_flag", _request_cancel)

        @_cl.asynccontextmanager
        async def _fake_k8s_client():
            yield MagicMock()

        monkeypatch.setattr(
            "aiperf.kubernetes.client.k8s_client", _fake_k8s_client, raising=True
        )

        sweep_cr = {
            "metadata": {"name": "s", "namespace": "ns", "uid": "uid"},
            "spec": {
                "image": "x:latest",
                "sweep": {
                    "type": "grid",
                    "variables": {"benchmark.phases.profiling.concurrency": [1]},
                },
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
                "multiRun": {"numRuns": 1},
            },
        }
        fake_custom = MagicMock()
        fake_custom.get_namespaced_custom_object = AsyncMock(return_value=sweep_cr)
        monkeypatch.setattr(
            "kubernetes_asyncio.client.CustomObjectsApi", lambda _api: fake_custom
        )

        fake_plan = MagicMock()
        fake_plan.configs = [object()]
        fake_plan.is_adaptive_search = False
        fake_plan.confidence_level = 0.95
        fake_plan.cooldown_seconds = 0.0
        fake_plan.use_adaptive = False
        fake_plan.export_jsonl_file = None
        monkeypatch.setattr(
            "aiperf.sweep_controller.plan_builder.build_plan_from_sweep",
            lambda cr: fake_plan,
        )
        monkeypatch.setattr(
            "aiperf.cli_runner._strategy._build_search_planner", lambda plan: None
        )

        class _Strategy:
            def get_aggregate_path(self, base):
                return base / "aggregate"

        monkeypatch.setattr(
            "aiperf.cli_runner._strategy.build_strategy",
            lambda plan, logger: _Strategy(),
        )
        calls: dict[str, object] = {"case": case}

        async def _aggregate_and_export(all_results, *args, **kwargs) -> None:
            calls["aggregated_results"] = list(all_results)

        if results:
            monkeypatch.setattr(
                "aiperf.cli_runner._aggregate.aggregate_and_export",
                _aggregate_and_export,
            )
        monkeypatch.setattr(main_mod, "_write_aggregate_manifest", lambda *a: None)
        monkeypatch.setattr(
            main_mod, "_mirror_strategy_aggregate_to_sweep_dir", lambda **kw: None
        )
        monkeypatch.setattr(
            main_mod, "_write_sweep_parent_aggregate", lambda **kw: None
        )
        monkeypatch.setattr(
            main_mod, "_load_aggregate_for_cr", lambda *a: {"parent": {}}
        )

        class _Orch:
            def __init__(self, base_dir):
                self.base_dir = base_dir

            async def execute(
                self, plan, executor, *, cancel_check=None, search_planner=None
            ):
                await asyncio.sleep(0)
                assert cancel_check is not None
                assert cancel_check() is True
                return list(results)

        monkeypatch.setattr(
            "aiperf.orchestrator.orchestrator.MultiRunOrchestrator", _Orch
        )

        class _Writer:
            def __init__(self, *args, **kwargs):
                pass

            async def aggregation_running(self):
                calls["aggregation_running"] = True

            async def aggregation_complete(self, **kwargs):
                calls["terminal_phase"] = kwargs["terminal_phase"]

            async def aggregation_failed(self, **kwargs):
                calls["aggregation_failed"] = kwargs

            async def parent_running(self):
                calls["parent_running"] = True

        monkeypatch.setattr(
            "aiperf.sweep_controller.status_writer.SweepStatusWriter", _Writer
        )

        class _Exec:
            def __init__(self, *args, **kwargs):
                pass

        monkeypatch.setattr(
            "aiperf.sweep_controller.k8s_executor.K8sChildJobExecutor", _Exec
        )
        monkeypatch.setattr(
            "aiperf.sweep_controller.k8s_executor.needs_trial_suffix",
            lambda **kwargs: False,
        )

        rc = await main_mod.main()
        return rc, calls

    rc, calls = await _run()

    assert rc == 0
    assert calls["terminal_phase"] == "Cancelled"
    if results:
        assert calls["aggregated_results"] == results
    else:
        assert "aggregated_results" not in calls
