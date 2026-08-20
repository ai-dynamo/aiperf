# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial regression-locks for Kubernetes sweep cancellation paths.

Focuses on:
- parent AIPerfSweep deletion propagating ``spec.cancel=true`` to child AIPerfJobs;
- cancelled children accounting in the dedicated ``runStates.cancelled`` bucket;
- JSON-patch conflict behavior that must retry via controller restart or skip safely;
- adaptive sweep-controller cancellation before and after partial child results;
- no aggregate-sidecar fetch when a cancelled parent lacks a run epoch.

Out of scope: full sweep creation and router rendering; see sibling files
``test_sweep_adversarial.py``, ``test_sweep_lifecycle_handlers.py``, and
``tests/unit/sweep_controller/test_*`` for those contracts.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import kopf
import pytest
from kubernetes_asyncio.client import ApiException

from aiperf.common.enums import OptimizationDirection
from aiperf.config.sweep import AdaptiveSearchSweep, Objective
from aiperf.config.sweep.adaptive import SearchSpaceDimension
from aiperf.operator.handlers.sweep import child_rollup, lifecycle
from aiperf.plugin.enums import SearchPlannerType
from aiperf.sweep_controller.main import SWEEP_CONTROLLER_RESULTS_SIDECAR_PORT

# ============================================================
# Helpers
# ============================================================


@contextlib.asynccontextmanager
async def _fake_k8s_client() -> AsyncGenerator[MagicMock, None]:
    yield MagicMock(name="k8s-api-client")


def _child(name: str, phase: str, index: str) -> dict[str, object]:
    return {
        "metadata": {
            "name": name,
            "uid": f"{name}-uid",
            "ownerReferences": [
                {
                    "apiVersion": "aiperf.nvidia.com/v1alpha1",
                    "kind": "AIPerfSweep",
                    "name": "latency-sweep",
                    "uid": "sweep-uid-7f2a",
                    "controller": True,
                }
            ],
            "labels": {
                "aiperf.nvidia.com/sweep": "latency-sweep",
                "aiperf.nvidia.com/sweep-uid": "sweep-uid-7f2a",
                "aiperf.nvidia.com/sweep-run-epoch": "1778027124",
                "aiperf.nvidia.com/variation-index": index,
                "aiperf.nvidia.com/variation-label": f"concurrency-{index}",
            },
        },
        "status": {"phase": phase},
    }


def _trigger_child_body() -> dict[str, object]:
    return {
        "metadata": {
            "name": "latency-sweep-v02-t0",
            "namespace": "aiperf-benchmarks",
            "uid": "latency-sweep-v02-t0-uid",
            "ownerReferences": [
                {
                    "apiVersion": "aiperf.nvidia.com/v1alpha1",
                    "kind": "AIPerfSweep",
                    "name": "latency-sweep",
                    "uid": "sweep-uid-7f2a",
                    "controller": True,
                }
            ],
            "labels": {
                "aiperf.nvidia.com/sweep": "latency-sweep",
                "aiperf.nvidia.com/sweep-uid": "sweep-uid-7f2a",
                "aiperf.nvidia.com/sweep-run-epoch": "1778027124",
            },
        }
    }


def _install_lifecycle_k8s(
    monkeypatch: pytest.MonkeyPatch,
    *,
    children: list[dict[str, object]],
    patch_side_effect: list[object] | BaseException | None = None,
) -> MagicMock:
    custom = MagicMock(name="custom-objects-api")
    custom.list_namespaced_custom_object = AsyncMock(return_value={"items": children})
    custom.patch_namespaced_custom_object = AsyncMock(side_effect=patch_side_effect)
    monkeypatch.setattr("aiperf.kubernetes.client.k8s_client", _fake_k8s_client)
    monkeypatch.setattr(
        "kubernetes_asyncio.client.CustomObjectsApi", lambda _api: custom
    )
    return custom


def _adaptive_sweep_cr() -> dict[str, object]:
    return {
        "metadata": {
            "name": "latency-sweep",
            "namespace": "aiperf-benchmarks",
            "uid": "sweep-uid-7f2a",
        },
        "status": {"runEpoch": "1778027124"},
        "spec": {
            "image": "nvcr.io/nvidia/aiperf:branch-may18",
            "sweep": {
                "type": "adaptive_search",
                "planner": "bayesian",
                "searchSpace": [
                    {
                        "path": "phases.profiling.concurrency",
                        "lo": 1,
                        "hi": 128,
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
                "maxIterations": 3,
                "nInitialPoints": 1,
            },
            "benchmark": {
                "models": ["meta-llama/Llama-3-8B"],
                "endpoint": {"urls": ["http://localhost:8000"], "type": "chat"},
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
            "failurePolicy": {"maxFailures": 1},
        },
    }


def _adaptive_plan() -> SimpleNamespace:
    return SimpleNamespace(
        configs=[object()],
        sweep=AdaptiveSearchSweep(
            planner=SearchPlannerType.BAYESIAN,
            search_space=[
                SearchSpaceDimension(
                    path="phases.profiling.concurrency",
                    lo=1,
                    hi=128,
                    kind="int",
                )
            ],
            objectives=[
                Objective(
                    metric="output_token_throughput",
                    stat="avg",
                    direction=OptimizationDirection.MAXIMIZE,
                )
            ],
            max_iterations=3,
            n_initial_points=1,
        ),
        confidence_level=0.95,
        cooldown_seconds=0.0,
        use_adaptive=True,
        is_sweep=False,
        is_adaptive_search=True,
        export_jsonl_file=None,
    )


def _run_result(*, label: str, success: bool) -> SimpleNamespace:
    return SimpleNamespace(
        label=label,
        success=success,
        error=None if success else "child cancelled",
        was_cancelled=not success,
        variation_values={"index": 0},
        variation_label="search_iter_0000",
        variation_index=0,
        trial_index=0,
    )


def _install_adaptive_main_harness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    results: list[SimpleNamespace],
    recovered_results: list[SimpleNamespace] | None = None,
) -> dict[str, object]:
    import importlib

    main_mod = importlib.import_module("aiperf.sweep_controller.main")
    monkeypatch.setenv("AIPERF_SWEEP_NAME", "latency-sweep")
    monkeypatch.setenv("AIPERF_SWEEP_NAMESPACE", "aiperf-benchmarks")
    monkeypatch.setenv("AIPERF_SWEEP_EPOCH", "1778027124")
    monkeypatch.setenv("HOSTNAME", "sweep-controller-0")
    monkeypatch.setattr(main_mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr("aiperf.kubernetes.client.k8s_client", _fake_k8s_client)

    captured: dict[str, object] = {"module": main_mod}

    async def _request_cancel(*_args: object, **kwargs: object) -> None:
        kwargs["flag"]["requested"] = True

    monkeypatch.setattr(main_mod, "_poll_cancel_flag", _request_cancel)

    custom = MagicMock(name="custom-objects-api")
    custom.get_namespaced_custom_object = AsyncMock(return_value=_adaptive_sweep_cr())
    monkeypatch.setattr(
        "kubernetes_asyncio.client.CustomObjectsApi", lambda _api: custom
    )
    monkeypatch.setattr(
        "aiperf.sweep_controller.plan_builder.build_plan_from_sweep",
        lambda _cr: _adaptive_plan(),
    )
    planner = SimpleNamespace(name="bayesian-planner-7f2a")
    monkeypatch.setattr(
        "aiperf.orchestrator.search_planner.build_search_planner",
        lambda _plan: planner,
    )

    class _Strategy:
        def get_aggregate_path(self, base_dir: Path) -> Path:
            return base_dir / "aggregate"

    monkeypatch.setattr(
        "aiperf.cli_runner._strategy.build_strategy", lambda _plan, _logger: _Strategy()
    )
    aggregate_plan_results = AsyncMock(return_value=tmp_path / "sweep_aggregate")
    monkeypatch.setattr(
        "aiperf.cli_runner._aggregation_dispatch.aggregate_plan_results",
        aggregate_plan_results,
    )
    monkeypatch.setattr(
        main_mod, "_load_aggregate_for_cr", lambda *args: {"parent": {}}
    )

    class _Orchestrator:
        def __init__(self, base_dir: Path) -> None:
            self.base_dir = base_dir

        async def execute(
            self,
            plan: object,
            executor: object,
            *,
            cancel_check: object | None = None,
            search_planner: object | None = None,
        ) -> list[SimpleNamespace]:
            await asyncio.sleep(0)
            captured["orchestrator_cancel_requested"] = (
                cancel_check is not None and cancel_check()
            )
            captured["search_planner"] = search_planner
            captured["executor"] = executor
            if search_planner is not None:
                await asyncio.to_thread(
                    (self.base_dir / "search_history.json").write_bytes,
                    b'{"iterations":[]}',
                )
            return list(results)

    monkeypatch.setattr(
        "aiperf.orchestrator.orchestrator.MultiRunOrchestrator", _Orchestrator
    )

    class _StatusWriter:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def parent_running(self) -> None:
            captured["parent_running"] = True

        async def aggregation_running(self) -> None:
            captured["aggregation_running"] = True

        async def aggregation_complete(self, **kwargs: object) -> None:
            captured["aggregation_complete"] = kwargs

        async def aggregation_failed(self, **kwargs: object) -> None:
            captured["aggregation_failed"] = kwargs

    monkeypatch.setattr(
        "aiperf.sweep_controller.status_writer.SweepStatusWriter", _StatusWriter
    )

    class _Executor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["executor_kwargs"] = kwargs
            self.terminal_children: tuple[object, ...] = ()

        async def recover_terminal_results(
            self, _plan: object
        ) -> list[SimpleNamespace]:
            captured["recovery_called"] = True
            return list(recovered_results or [])

    monkeypatch.setattr(
        "aiperf.sweep_controller.k8s_executor.K8sChildJobExecutor", _Executor
    )
    monkeypatch.setattr(
        "aiperf.sweep_controller.k8s_executor.needs_trial_suffix",
        lambda **_kwargs: False,
    )
    captured["aggregate_plan_results"] = aggregate_plan_results
    return captured


# ============================================================
# Operator parent-cancel propagation
# ============================================================


class TestOperatorParentCancelPropagation:
    """AIPerfSweep deletion should cooperatively cancel child AIPerfJobs."""

    @pytest.mark.asyncio
    async def test_on_delete_patch_conflict_continues_to_remaining_children(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        children = [
            {
                "metadata": {
                    "name": child_name,
                    "uid": f"{child_name}-uid",
                    "ownerReferences": [
                        {
                            "apiVersion": "aiperf.nvidia.com/v1alpha1",
                            "kind": "AIPerfSweep",
                            "name": "latency-sweep",
                            "uid": "sweep-uid-7f2a",
                            "controller": True,
                        }
                    ],
                    "labels": {
                        "aiperf.nvidia.com/sweep": "latency-sweep",
                        "aiperf.nvidia.com/sweep-uid": "sweep-uid-7f2a",
                        "aiperf.nvidia.com/sweep-run-epoch": "1778027124",
                    },
                }
            }
            for child_name in (
                "latency-sweep-v00-t0",
                "latency-sweep-v01-t0",
            )
        ]
        custom = _install_lifecycle_k8s(
            monkeypatch,
            children=children,
            patch_side_effect=[ApiException(status=409, reason="Conflict"), None],
        )

        await lifecycle.on_delete(
            body=_adaptive_sweep_cr(),
            uid="sweep-uid-7f2a",
            name="latency-sweep",
            namespace="aiperf-benchmarks",
        )

        assert custom.patch_namespaced_custom_object.await_count == 2
        patched_names = [
            call.kwargs["name"]
            for call in custom.patch_namespaced_custom_object.await_args_list
        ]
        assert patched_names == ["latency-sweep-v00-t0", "latency-sweep-v01-t0"]
        for call in custom.patch_namespaced_custom_object.await_args_list:
            child_name = call.kwargs["name"]
            assert call.kwargs["body"] == [
                {
                    "op": "test",
                    "path": "/metadata/uid",
                    "value": f"{child_name}-uid",
                },
                {"op": "add", "path": "/spec/cancel", "value": True},
            ]
            assert call.kwargs["_content_type"] == "application/json-patch+json"

    @pytest.mark.asyncio
    async def test_on_delete_patch_503_raises_temporary_error_for_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_lifecycle_k8s(
            monkeypatch,
            children=[
                {
                    "metadata": {
                        "name": "latency-sweep-v00-t0",
                        "uid": "latency-sweep-v00-t0-uid",
                        "ownerReferences": [
                            {
                                "apiVersion": "aiperf.nvidia.com/v1alpha1",
                                "kind": "AIPerfSweep",
                                "name": "latency-sweep",
                                "uid": "sweep-uid-7f2a",
                                "controller": True,
                            }
                        ],
                        "labels": {
                            "aiperf.nvidia.com/sweep": "latency-sweep",
                            "aiperf.nvidia.com/sweep-uid": "sweep-uid-7f2a",
                            "aiperf.nvidia.com/sweep-run-epoch": "1778027124",
                        },
                    }
                }
            ],
            patch_side_effect=ApiException(status=503, reason="Unavailable"),
        )

        with pytest.raises(
            kopf.TemporaryError,
            match=r"cooperative-cancel patch.*aiperf-benchmarks/latency-sweep-v00-t0.*503",
        ):
            await lifecycle.on_delete(
                body=_adaptive_sweep_cr(),
                uid="sweep-uid-7f2a",
                name="latency-sweep",
                namespace="aiperf-benchmarks",
            )


# ============================================================
# Operator rollup cancellation accounting and conflict guards
# ============================================================


class TestOperatorCancelledBucketAccounting:
    """Cancelled children are terminal but not failed for parent rollup counts."""

    @pytest.mark.asyncio
    async def test_child_rollup_mixed_failed_and_cancelled_children_preserves_buckets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        parent_patches: list[dict[str, object]] = []
        phase_calls: list[dict[str, str]] = []

        async def _patch_parent(*, body: dict[str, object], **_kwargs: object) -> None:
            parent_patches.append(body)

        async def _phase_set(
            *,
            namespace: str,
            name: str,
            expect_phase: str,
            new_phase: str,
            expected_uid: str | None = None,
            api: object | None = None,
        ) -> None:
            phase_calls.append(
                {
                    "namespace": namespace,
                    "name": name,
                    "expect": expect_phase,
                    "new": new_phase,
                    "expected_uid": expected_uid,
                }
            )

        monkeypatch.setattr(child_rollup, "_patch_parent_status", _patch_parent)
        monkeypatch.setattr(child_rollup, "_append_run_entry", AsyncMock())
        monkeypatch.setattr(child_rollup, "_conditional_phase_set", _phase_set)
        monkeypatch.setattr(
            child_rollup,
            "_read_parent_status",
            AsyncMock(return_value={"phase": "Running", "maxTotalRuns": 3}),
        )
        monkeypatch.setattr(
            child_rollup,
            "_count_owned_children",
            AsyncMock(
                return_value={
                    "pending": 0,
                    "running": 0,
                    "completed": 1,
                    "failed": 1,
                    "cancelled": 1,
                    "in_flight": 0,
                    "total_terminal_phase": "Aggregating",
                    "owned_children": [
                        _child("latency-sweep-v00-t0", "Completed", "00"),
                        _child("latency-sweep-v01-t0", "Failed", "01"),
                        _child("latency-sweep-v02-t0", "Cancelled", "02"),
                    ],
                }
            ),
        )
        monkeypatch.setattr(
            child_rollup,
            "_read_current_child",
            AsyncMock(return_value=_trigger_child_body()),
        )
        monkeypatch.setattr("aiperf.kubernetes.client.k8s_client", _fake_k8s_client)

        await child_rollup.on_child_phase_transition(
            body=_trigger_child_body(),
            status={"phase": "Cancelled"},
            name="latency-sweep-v02-t0",
            namespace="aiperf-benchmarks",
        )

        assert parent_patches
        status_patch = parent_patches[0]["status"]
        assert status_patch["runStates"] == {
            "pending": 0,
            "running": 0,
            "completed": 1,
            "failed": 1,
            "cancelled": 1,
        }
        assert status_patch["completedRuns"] == 1
        assert status_patch["failedRuns"] == 1
        assert phase_calls == [
            {
                "namespace": "aiperf-benchmarks",
                "name": "latency-sweep",
                "expect": "Running",
                "new": "Aggregating",
                "expected_uid": "sweep-uid-7f2a",
            }
        ]

    @pytest.mark.asyncio
    async def test_conditional_phase_set_patch_conflict_drops_aggregating_write(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        custom = MagicMock(name="custom-objects-api")
        custom.patch_namespaced_custom_object_status = AsyncMock(
            side_effect=ApiException(status=422, reason="Unprocessable Entity")
        )
        monkeypatch.setattr(
            "kubernetes_asyncio.client.CustomObjectsApi", lambda _api: custom
        )

        await child_rollup._conditional_phase_set(
            namespace="aiperf-benchmarks",
            name="latency-sweep",
            expect_phase="Running",
            new_phase="Aggregating",
            api=MagicMock(),
        )

        kwargs = custom.patch_namespaced_custom_object_status.await_args.kwargs
        assert kwargs["body"] == [
            {"op": "test", "path": "/status/phase", "value": "Running"},
            {"op": "replace", "path": "/status/phase", "value": "Aggregating"},
        ]
        assert kwargs["_content_type"] == "application/json-patch+json"


# ============================================================
# Sweep-controller adaptive cancellation
# ============================================================


class TestAdaptiveSweepControllerCancellation:
    """Adaptive sweeps must observe parent cancellation through the orchestrator seam."""

    @pytest.mark.asyncio
    async def test_main_adaptive_cancel_before_success_skips_confidence_aggregate(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured = _install_adaptive_main_harness(monkeypatch, tmp_path, results=[])
        main_mod = captured["module"]

        rc = await main_mod.main()

        assert rc == 0
        assert captured["orchestrator_cancel_requested"] is True
        assert captured["aggregation_complete"]["terminal_phase"] == "Cancelled"
        assert (
            captured["aggregation_complete"]["port"]
            == SWEEP_CONTROLLER_RESULTS_SIDECAR_PORT
        )
        assert captured["aggregate_plan_results"].await_count == 0

    @pytest.mark.asyncio
    async def test_main_adaptive_cancel_after_partial_success_aggregates_partial_results(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        results = [
            _run_result(label="search-iter-0000", success=True),
            _run_result(label="search-iter-0001", success=False),
        ]
        captured = _install_adaptive_main_harness(
            monkeypatch, tmp_path, results=results
        )
        main_mod = captured["module"]

        rc = await main_mod.main()

        assert rc == 0
        assert captured["aggregation_complete"]["terminal_phase"] == "Cancelled"
        aggregate_plan_results = captured["aggregate_plan_results"]
        aggregate_plan_results.assert_awaited_once()
        assert aggregate_plan_results.await_args.args[0] == results

    @pytest.mark.asyncio
    async def test_main_aggregate_patch_conflict_returns_nonzero_for_restart_retry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured = _install_adaptive_main_harness(
            monkeypatch,
            tmp_path,
            results=[_run_result(label="search-iter-0000", success=True)],
        )
        main_mod = captured["module"]

        async def _conflict_on_terminal_patch(**_kwargs: object) -> None:
            raise ApiException(status=409, reason="Conflict")

        writer_calls: dict[str, object] = {}

        class _StatusWriter:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            async def parent_running(self) -> None:
                return None

            async def aggregation_running(self) -> None:
                return None

            async def aggregation_complete(self, **kwargs: object) -> None:
                await _conflict_on_terminal_patch(**kwargs)

            async def aggregation_failed(self, **kwargs: object) -> None:
                writer_calls["aggregation_failed_kwargs"] = kwargs

        monkeypatch.setattr(
            "aiperf.sweep_controller.status_writer.SweepStatusWriter", _StatusWriter
        )

        rc = await main_mod.main()

        assert rc == 1
        assert "aggregation_failed_kwargs" not in writer_calls


# ============================================================
# Operator aggregate fetch cancellation guard
# ============================================================


class TestOperatorAggregateFetchCancellationGuard:
    """Cancelled parents without a controller epoch should not trigger sidecar fetch."""

    @pytest.mark.asyncio
    async def test_aggregation_complete_cancelled_without_run_epoch_skips_fetch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.handlers.sweep import _aggregate_fetch
        from aiperf.operator.main import on_aiperfsweep_aggregation_complete

        fetch = AsyncMock()
        monkeypatch.setattr(_aggregate_fetch, "fetch_sweep_aggregate_to_disk", fetch)

        await on_aiperfsweep_aggregation_complete(
            body={"metadata": {"name": "latency-sweep"}},
            status={"phase": "Cancelled", "aggregation": {"phase": "Complete"}},
            name="latency-sweep",
            namespace="aiperf-benchmarks",
        )

        fetch.assert_not_awaited()
