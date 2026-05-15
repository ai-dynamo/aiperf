# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the orchestrator's per-cell callback hook.

The hook fires once per variation in the grid (independent) and adaptive
paths so a live-Pareto tracker can aggregate cells as they complete. Tests
drive ``_run_independent_cell`` directly with a mocked executor and a
mocked plan/strategy so they stay fast and free of real subprocess work.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config import BenchmarkConfig
from aiperf.config.sweep import SweepVariation
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.orchestrator import MultiRunOrchestrator
from aiperf.search_recipes._pareto_axes import ParetoAxesSpec

_MINIMAL_CONFIG_KWARGS = {
    "models": ["test-model"],
    "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
    "datasets": [
        {
            "name": "default",
            "type": "synthetic",
            "entries": 100,
            "prompts": {"isl": 128, "osl": 64},
        }
    ],
    "phases": [
        {
            "name": "profiling",
            "type": "concurrency",
            "requests": 100,
            "concurrency": 1,
        },
    ],
}


def _make_cfg() -> BenchmarkConfig:
    return BenchmarkConfig(**_MINIMAL_CONFIG_KWARGS)


def _make_pareto_axes() -> ParetoAxesSpec:
    return ParetoAxesSpec(
        x_metric="request_latency",
        x_stat="p95",
        y_metric="output_token_throughput",
        y_stat="avg",
        series_keys=(),
    )


def _make_variation(values: dict, label: str = "v0", index: int = 0) -> SweepVariation:
    """Real SweepVariation: BenchmarkRun's validator rejects MagicMocks."""
    return SweepVariation(index=index, label=label, values=dict(values))


def _make_plan(*, trials: int = 1) -> MagicMock:
    """Mocked BenchmarkPlan covering every attribute _run_independent_cell touches.

    Pareto axes are injected by patching
    ``aiperf.cli_runner._pareto._resolve_pareto_axes`` per test, not by
    stubbing fields on the plan.
    """
    plan = MagicMock()
    plan.is_adaptive_search = False
    plan.is_sweep = True
    plan.trials = trials
    plan.confidence_level = 0.95
    plan.variation_seeds = []
    plan.variables = {}
    plan.failure_policy = None
    plan.sweep = None  # _plan_iteration_order falls back to REPEATED
    plan.sweep_id = "test-sweep-id"
    return plan


def _patch_axes(axes):
    """Context manager: make ``_resolve_pareto_axes(plan)`` return ``axes``."""
    return patch("aiperf.cli_runner._pareto._resolve_pareto_axes", return_value=axes)


def _make_strategy(num_trials: int = 1) -> MagicMock:
    """Mocked ExecutionStrategy returning fixed-trials semantics."""
    strategy = MagicMock()
    strategy.should_continue.side_effect = lambda results: len(results) < num_trials
    strategy.get_next_config.side_effect = lambda cfg, _r: cfg
    strategy.get_run_label.side_effect = lambda i: f"run_{i + 1:04d}"
    strategy.get_cooldown_seconds.return_value = 0.0
    return strategy


def _make_executor_returning(results: list[RunResult]) -> MagicMock:
    """Mocked RunExecutor whose execute() pops successive results."""
    iterator = iter(results)
    executor = MagicMock()
    executor.derive_id.side_effect = (
        lambda plan, var_idx, trial: f"id-{var_idx}-{trial}"
    )
    executor.execute = AsyncMock(side_effect=lambda _run: next(iterator))
    return executor


def _run_result_with_axes(
    *,
    label: str,
    latency_p95: float,
    throughput_avg: float,
) -> RunResult:
    """Build a successful RunResult with the two metrics our axes spec consumes."""
    return RunResult(
        label=label,
        success=True,
        summary_metrics={
            "request_latency": JsonMetricResult(
                unit="ms", avg=latency_p95, p95=latency_p95
            ),
            "output_token_throughput": JsonMetricResult(
                unit="tokens/s", avg=throughput_avg
            ),
        },
    )


@pytest.mark.asyncio
async def test_cell_callback_fires_after_each_variation_in_grid_path(tmp_path: Path):
    """One callback per variation, with the aggregated cell dict."""
    received: list[tuple] = []

    def hook(variation_key, cell):
        received.append((variation_key, cell))

    axes = _make_pareto_axes()
    plan = _make_plan(trials=1)

    orch = MultiRunOrchestrator(tmp_path, cell_callback=hook)

    variations = [
        _make_variation({"concurrency": 1}, label="v0", index=0),
        _make_variation({"concurrency": 4}, label="v1", index=1),
    ]
    results_per_variation = [
        _run_result_with_axes(label="run_0001", latency_p95=10.0, throughput_avg=50.0),
        _run_result_with_axes(label="run_0001", latency_p95=20.0, throughput_avg=80.0),
    ]

    with _patch_axes(axes):
        for var, result in zip(variations, results_per_variation, strict=True):
            executor = _make_executor_returning([result])
            strategy = _make_strategy(num_trials=1)
            cell_results, aborted = await orch._run_independent_cell(
                plan,
                executor,
                strategy=strategy,
                cfg=_make_cfg(),
                variation=var,
                var_idx=var.index,
                prior_all_results=[],
                cancel_check=None,
            )
            assert not aborted
            assert len(cell_results) == 1

    assert len(received) == 2
    # Variation key is (label, sorted-values-tuple).
    key0, cell0 = received[0]
    assert key0 == ("v0", (("concurrency", 1),))
    assert cell0["x"] == 10.0
    assert cell0["y"] == 50.0
    assert cell0["params"] == {"concurrency": 1}
    assert cell0["pareto_optimal"] is False

    key1, cell1 = received[1]
    assert key1 == ("v1", (("concurrency", 4),))
    assert cell1["x"] == 20.0
    assert cell1["y"] == 80.0


@pytest.mark.asyncio
async def test_cell_callback_not_called_when_none(tmp_path: Path):
    """No callback registered: the cell loop completes without raising."""
    axes = _make_pareto_axes()
    plan = _make_plan(trials=1)

    orch = MultiRunOrchestrator(tmp_path)  # no cell_callback

    var = _make_variation({"concurrency": 1})
    executor = _make_executor_returning(
        [_run_result_with_axes(label="run_0001", latency_p95=10.0, throughput_avg=50.0)]
    )
    with _patch_axes(axes):
        cell_results, aborted = await orch._run_independent_cell(
            plan,
            executor,
            strategy=_make_strategy(num_trials=1),
            cfg=_make_cfg(),
            variation=var,
            var_idx=0,
            prior_all_results=[],
            cancel_check=None,
        )

    assert len(cell_results) == 1
    assert not aborted


@pytest.mark.asyncio
async def test_cell_callback_fires_with_minimal_cell_when_recipe_has_no_pareto_axes(
    tmp_path: Path,
):
    """Recipe without pareto_axes: hook still fires with a minimal cell.

    The sweep-mode-agnostic contract (``SweepTableLogger`` and similar
    consumers) requires the callback to fire for every variation, even
    when the recipe declares no ``pareto_axes``. ``x``/``y`` are ``None``
    in that case but ``params`` is populated.
    """
    received: list[tuple] = []

    def hook(variation_key, cell):
        received.append((variation_key, cell))

    plan = _make_plan(trials=1)

    orch = MultiRunOrchestrator(tmp_path, cell_callback=hook)

    var = _make_variation({"concurrency": 1})
    executor = _make_executor_returning(
        [_run_result_with_axes(label="run_0001", latency_p95=10.0, throughput_avg=50.0)]
    )
    with _patch_axes(None):
        cell_results, aborted = await orch._run_independent_cell(
            plan,
            executor,
            strategy=_make_strategy(num_trials=1),
            cfg=_make_cfg(),
            variation=var,
            var_idx=0,
            prior_all_results=[],
            cancel_check=None,
        )

    assert len(cell_results) == 1
    assert not aborted
    assert len(received) == 1
    key, cell = received[0]
    assert key == ("v0", (("concurrency", 1),))
    assert cell["x"] is None
    assert cell["y"] is None
    assert cell["params"] == {"concurrency": 1}
    assert cell["pareto_optimal"] is False
    assert cell["_cell_results"] == cell_results


@pytest.mark.asyncio
async def test_cell_callback_exception_does_not_break_sweep(tmp_path: Path):
    """A buggy callback must not surface its exception into the sweep loop."""

    def bad_hook(variation_key, cell):
        raise RuntimeError("kaboom")

    axes = _make_pareto_axes()
    plan = _make_plan(trials=1)

    orch = MultiRunOrchestrator(tmp_path, cell_callback=bad_hook)

    var = _make_variation({"concurrency": 1})
    executor = _make_executor_returning(
        [_run_result_with_axes(label="run_0001", latency_p95=10.0, throughput_avg=50.0)]
    )

    # Must NOT raise.
    with _patch_axes(axes):
        cell_results, aborted = await orch._run_independent_cell(
            plan,
            executor,
            strategy=_make_strategy(num_trials=1),
            cfg=_make_cfg(),
            variation=var,
            var_idx=0,
            prior_all_results=[],
            cancel_check=None,
        )

    assert len(cell_results) == 1
    assert not aborted
