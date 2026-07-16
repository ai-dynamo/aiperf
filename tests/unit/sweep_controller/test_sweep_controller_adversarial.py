# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial sweep-controller tests.

Focuses on:
- adaptive planner wiring and failure cleanup in ``sweep_controller.main``;
- result-stream driven child manifest derivation for adaptive and trial naming;
- malformed aggregate bundle members that must not poison the CR status patch;
- inline aggregate truncation when no-PVC parent mirrors exceed the CR budget.

Out of scope: child AIPerfJob CRUD and watch polling, covered by sibling
``test_k8s_executor_*.py`` files.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from pytest import param

from aiperf.common.enums import OptimizationDirection
from aiperf.config.sweep import AdaptiveSearchSweep, Objective
from aiperf.config.sweep.adaptive import SearchSpaceDimension
from aiperf.plugin.enums import SearchPlannerType
from aiperf.sweep_controller.main import (
    SWEEP_CONTROLLER_RESULTS_SIDECAR_PORT,
    _load_aggregate_for_cr,
    _write_sweep_parent_aggregate,
    resolve_terminal_phase,
    write_aggregate_marker,
)

# ===========================================================================
# Helpers
# ===========================================================================


class _SearchPlanner(Protocol):
    marker: str


def _write_json(path: Path, doc: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(doc, option=orjson.OPT_INDENT_2))


def _result(
    *,
    label: str,
    variation_label: str,
    trial_index: int,
    variation_index: int = 0,
    success: bool = True,
    variation_values: dict[str, object] | None = None,
    child_run_epoch: str = "1778027130",
) -> SimpleNamespace:
    return SimpleNamespace(
        label=label,
        success=success,
        error=None if success else "worker pod cancelled",
        variation_values=variation_values or {},
        variation_label=variation_label,
        variation_index=variation_index,
        trial_index=trial_index,
        child_run_epoch=child_run_epoch,
    )


def _valid_sweep_cr() -> dict[str, object]:
    return {
        "metadata": {
            "name": "sweep-conc-demo",
            "namespace": "aiperf-benchmarks",
            "uid": "sweep-uid-7f2a",
        },
        "spec": {
            "image": "nvcr.io/nvidia/aiperf:branch-may18",
            "sweep": {
                "type": "grid",
                "variables": {"benchmark.phases.profiling.concurrency": [1]},
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
        },
    }


def _adaptive_sweep() -> AdaptiveSearchSweep:
    return AdaptiveSearchSweep(
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
    )


@contextlib.asynccontextmanager
async def _fake_k8s_client() -> AsyncGenerator[MagicMock, None]:
    yield MagicMock(name="k8s-api-client")


def _install_main_harness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    search_planner: _SearchPlanner | None = None,
    planner_error: Exception | None = None,
) -> dict[str, object]:
    import importlib

    main_mod = importlib.import_module("aiperf.sweep_controller.main")
    monkeypatch.setenv("AIPERF_SWEEP_NAME", "sweep-conc-demo")
    monkeypatch.setenv("AIPERF_SWEEP_NAMESPACE", "aiperf-benchmarks")
    monkeypatch.setenv("AIPERF_SWEEP_EPOCH", "1778027124")
    monkeypatch.setenv("HOSTNAME", "sweep-controller-0")
    monkeypatch.setattr(main_mod, "RESULTS_DIR", tmp_path)
    write_aggregate_marker(tmp_path)

    captured: dict[str, object] = {"module": main_mod}
    real_create_task = asyncio.create_task

    def _capturing_create_task(
        coro: object, *args: object, **kwargs: object
    ) -> asyncio.Task:
        task = real_create_task(coro, *args, **kwargs)
        captured.setdefault("cancel_task", task)
        return task

    async def _fast_cancel_poll(*args: object, **kwargs: object) -> None:
        flag = kwargs["flag"]
        while not flag["requested"]:
            await asyncio.sleep(0)

    monkeypatch.setattr(main_mod, "_poll_cancel_flag", _fast_cancel_poll)
    monkeypatch.setattr(main_mod.asyncio, "create_task", _capturing_create_task)
    monkeypatch.setattr("aiperf.kubernetes.client.k8s_client", _fake_k8s_client)

    fake_custom = MagicMock(name="custom-objects-api")
    fake_custom.get_namespaced_custom_object = AsyncMock(return_value=_valid_sweep_cr())
    monkeypatch.setattr(
        "kubernetes_asyncio.client.CustomObjectsApi", lambda _api: fake_custom
    )

    fake_plan = SimpleNamespace(
        configs=[object()],
        sweep=_adaptive_sweep(),
        confidence_level=0.95,
        cooldown_seconds=0.0,
        use_adaptive=True,
        export_jsonl_file=None,
    )
    monkeypatch.setattr(
        "aiperf.sweep_controller.plan_builder.build_plan_from_sweep",
        lambda _cr: fake_plan,
    )

    def _build_search_planner(_plan: object) -> _SearchPlanner | None:
        if planner_error is not None:
            raise planner_error
        return search_planner

    monkeypatch.setattr(
        "aiperf.cli_runner._strategy._build_search_planner", _build_search_planner
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
            captured["orchestrator_plan"] = plan
            captured["executor"] = executor
            captured["search_planner"] = search_planner
            captured["cancel_check"] = cancel_check
            return []

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

    monkeypatch.setattr(
        "aiperf.sweep_controller.k8s_executor.K8sChildJobExecutor", _Executor
    )
    monkeypatch.setattr(
        "aiperf.sweep_controller.k8s_executor.needs_trial_suffix",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        main_mod, "_load_aggregate_for_cr", lambda *args: {"parent": {}}
    )
    return captured


# ===========================================================================
# Terminal phase boundaries
# ===========================================================================


@pytest.mark.parametrize(
    "completed,failed,max_failures,cancel_requested,expected",
    [
        (6, 0, 0, False, "Succeeded"),
        (5, 1, 0, False, "PartiallyFailed"),
        (0, 6, 0, False, "Failed"),
        param(4, 2, 2, False, "Failed", id="failure-budget-boundary"),
        param(0, 0, 0, True, "Cancelled", id="cancel-wins-before-results"),
        param(5, 1, 1, True, "Cancelled", id="cancel-wins-over-failure-budget"),
    ],
)  # fmt: skip
def test_resolve_terminal_phase_boundary_matrix_returns_parent_phase(
    completed: int,
    failed: int,
    max_failures: int,
    cancel_requested: bool,
    expected: str,
) -> None:
    assert (
        resolve_terminal_phase(
            completed=completed,
            failed=failed,
            max_failures=max_failures,
            cancel_requested=cancel_requested,
        )
        == expected
    )


# ===========================================================================
# Adaptive planner and cleanup adversaries
# ===========================================================================


@pytest.mark.asyncio
async def test_main_adaptive_search_planner_passes_planner_and_sidecar_port(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    planner = SimpleNamespace(marker="bayesian-planner-7f2a")
    captured = _install_main_harness(monkeypatch, tmp_path, search_planner=planner)
    main_mod = captured["module"]

    rc = await main_mod.main()

    assert rc == 0
    assert captured["search_planner"] is planner
    assert captured["parent_running"] is True
    aggregate_complete = captured["aggregation_complete"]
    assert aggregate_complete["port"] == SWEEP_CONTROLLER_RESULTS_SIDECAR_PORT
    assert aggregate_complete["aggregate_path"] == (
        "/api/v1/results/aiperf-benchmarks/sweep-conc-demo/aggregate"
    )
    assert aggregate_complete["controller_host"] == "sweep-controller-0"


@pytest.mark.asyncio
async def test_main_adaptive_planner_failure_cancels_poll_task_before_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = _install_main_harness(
        monkeypatch,
        tmp_path,
        planner_error=RuntimeError("bayesian planner cold-start failed"),
    )
    main_mod = captured["module"]

    try:
        with pytest.raises(RuntimeError, match="bayesian planner cold-start failed"):
            await main_mod.main()
        cancel_task = captured.get("cancel_task")
        assert isinstance(cancel_task, asyncio.Task)
        assert cancel_task.done(), "planner failure leaked the cancel-poll task"
    finally:
        cancel_task = captured.get("cancel_task")
        if isinstance(cancel_task, asyncio.Task):
            cancel_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cancel_task


# ===========================================================================
# Child manifest derivation from result stream
# ===========================================================================


def test_write_sweep_parent_aggregate_adaptive_results_use_true_variation_index(
    tmp_path: Path,
) -> None:
    # The child manifest must reflect each result's real variation_index
    # (stamped from BenchmarkVariation.index), not a label-appearance counter.
    # A BO planner can re-propose the same config under a fresh index, so two
    # results sharing a variation_label can still belong to distinct cells.
    results = [
        _result(
            label="trial-a0",
            variation_label="search_iter_0000",
            variation_index=0,
            trial_index=0,
        ),
        _result(
            label="trial-b0",
            variation_label="search_iter_0001",
            variation_index=1,
            trial_index=0,
        ),
        _result(
            label="trial-a1",
            variation_label="search_iter_0000",
            variation_index=0,
            trial_index=1,
        ),
    ]

    _write_sweep_parent_aggregate(
        base_dir=tmp_path,
        sweep_cr={"metadata": {"namespace": "aiperf-benchmarks", "name": "demo-bo"}},
        spec=SimpleNamespace(model_dump=lambda mode: {"benchmark": {"models": ["m"]}}),
        results=results,
        plan=SimpleNamespace(configs=[object(), object(), object()]),
        sweep_run_epoch="1778027124",
        with_trial_suffix=True,
    )

    children_doc = orjson.loads(
        (
            tmp_path
            / "aiperf-benchmarks"
            / "sweeps"
            / "demo-bo"
            / "1778027124"
            / "children.json"
        ).read_bytes()
    )
    children = children_doc["children"]
    assert [child["name"] for child in children] == [
        "demo-bo-v00-t0",
        "demo-bo-v00-t1",
        "demo-bo-v01-t0",
    ]
    assert [child["variation_index"] for child in children] == [0, 0, 1]
    assert [child["trial_index"] for child in children] == [0, 1, 0]


def test_write_sweep_parent_aggregate_single_trial_omits_trial_suffix_and_field(
    tmp_path: Path,
) -> None:
    results = [
        _result(
            label="grid-c128",
            variation_label="concurrency=128",
            trial_index=0,
            variation_index=7,
            variation_values={"phases.profiling.concurrency": 128},
        )
    ]

    _write_sweep_parent_aggregate(
        base_dir=tmp_path,
        sweep_cr={"metadata": {"namespace": "aiperf-benchmarks", "name": "demo-grid"}},
        spec=SimpleNamespace(model_dump=lambda mode: {"benchmark": {"models": ["m"]}}),
        results=results,
        plan=SimpleNamespace(configs=[object()]),
        sweep_run_epoch="1778027124",
        with_trial_suffix=False,
    )

    children_doc = orjson.loads(
        (
            tmp_path
            / "aiperf-benchmarks"
            / "sweeps"
            / "demo-grid"
            / "1778027124"
            / "children.json"
        ).read_bytes()
    )
    child = children_doc["children"][0]
    assert child["name"] == "demo-grid-v07"
    assert child["variation_index"] == 7
    assert child["trial_index"] is None


# ===========================================================================
# Aggregate bundle malformed-file resilience and CR budget boundaries
# ===========================================================================


def test_load_aggregate_for_cr_malformed_parent_and_confidence_keeps_children(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    sweep_dir = tmp_path / "aiperf-benchmarks" / "sweeps" / "sweep-x" / "1778027124"
    sweep_dir.mkdir(parents=True)
    aggregate_dir = tmp_path / "aggregate"
    aggregate_dir.mkdir()
    (sweep_dir / "aggregate.json").write_bytes(b'{"phase":')
    _write_json(
        sweep_dir / "children.json",
        {"children": [{"name": "sweep-x-v00", "status": "Succeeded"}]},
    )
    (aggregate_dir / "profile_export_aiperf_aggregate.json").write_bytes(b"not-json")

    with caplog.at_level(logging.WARNING, logger="aiperf.sweep_controller.main"):
        bundle = _load_aggregate_for_cr(
            tmp_path,
            "aiperf-benchmarks",
            "sweep-x",
            "1778027124",
        )

    assert bundle == {
        "children": {"children": [{"name": "sweep-x-v00", "status": "Succeeded"}]}
    }
    assert "skipping parent" in caplog.text
    assert "skipping confidence" in caplog.text
    assert "sweep-x" in caplog.text


def test_load_aggregate_for_cr_parent_only_over_budget_uses_truncation_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sweep_dir = tmp_path / "aiperf-benchmarks" / "sweeps" / "sweep-huge" / "1778027124"
    _write_json(sweep_dir / "aggregate.json", {"payload": "x" * 10_000})
    monkeypatch.setattr(
        "aiperf.sweep_controller.main._AGGREGATE_INLINE_MAX_BYTES",
        256,
    )

    bundle = _load_aggregate_for_cr(
        tmp_path,
        "aiperf-benchmarks",
        "sweep-huge",
        "1778027124",
    )

    assert set(bundle) == {"aggregateTruncated"}
    marker = bundle["aggregateTruncated"]
    assert marker["reason"] == "inline_status_budget_exceeded"
    assert marker["omittedKeys"] == ["parent"]
    assert marker["maxBytes"] == 256
    assert len(orjson.dumps(bundle)) <= 256
