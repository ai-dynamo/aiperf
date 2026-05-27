# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for aiperf.cli_runner._sweep_aggregate.aggregate_sweep_and_export."""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from aiperf.cli_runner._sweep_aggregate import (
    _per_variation_aggregate_dir,
    aggregate_per_variation_and_export,
    aggregate_sweep_and_export,
)
from aiperf.cli_runner._sweep_table import SweepTableLogger
from aiperf.common.enums import SweepMode
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config.resolution.plan import BenchmarkConfig, BenchmarkPlan
from aiperf.orchestrator.models import RunResult

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
        }
    ],
}


def _make_plan(confidence_level: float = 0.95) -> BenchmarkPlan:
    cfg = BenchmarkConfig(**_MINIMAL_CONFIG_KWARGS)
    return BenchmarkPlan(
        configs=[cfg], confidence_level=confidence_level, random_seed=42
    )


def _result(
    label: str, concurrency: int, throughput: float, ttft_p99: float, *, success=True
) -> RunResult:
    """Build a RunResult tagged with a single-param sweep variation."""
    return RunResult(
        label=label,
        success=success,
        summary_metrics={
            "request_throughput": JsonMetricResult(
                unit="requests/sec",
                avg=throughput,
                min=throughput,
                max=throughput,
                count=10,
                sum=throughput * 10,
            ),
            "time_to_first_token": JsonMetricResult(
                unit="ms", avg=ttft_p99 - 5, p99=ttft_p99, min=ttft_p99, max=ttft_p99
            ),
        },
        variation_label=f"concurrency={concurrency}",
        variation_values={"concurrency": concurrency},
        trial_index=0,
    )


@pytest.fixture
def logger() -> logging.Logger:
    return logging.getLogger("test.sweep_helpers")


@pytest.mark.asyncio
async def test_aggregate_sweep_and_export_two_variations_one_trial(tmp_path, logger):
    """Two variations × 1 trial: writes JSON+CSV with 2 per-combination rows."""
    plan = _make_plan()
    results = [
        _result("c10", concurrency=10, throughput=100.0, ttft_p99=50.0),
        _result("c20", concurrency=20, throughput=180.0, ttft_p99=80.0),
    ]

    out_dir = await aggregate_sweep_and_export(results, plan, tmp_path, logger)

    assert out_dir == tmp_path / "sweep_aggregate"
    json_path = out_dir / "profile_export_aiperf_sweep.json"
    csv_path = out_dir / "profile_export_aiperf_sweep.csv"
    assert json_path.exists()
    assert csv_path.exists()

    data = json.loads(json_path.read_text())
    assert len(data["per_combination_metrics"]) == 2
    # Single-trial collapse: std == 0 across cells
    for entry in data["per_combination_metrics"]:
        for metric in entry["metrics"].values():
            assert metric["std"] == 0.0
    # Schema 1.1 fields propagate through the single-trial projection.
    rt_stats = next(
        e["metrics"]["request_throughput"] for e in data["per_combination_metrics"]
    )
    assert rt_stats["count"] == 10
    assert rt_stats["sum"] > 0


@pytest.mark.asyncio
async def test_aggregate_sweep_and_export_two_variations_three_trials(tmp_path, logger):
    """Two variations × 3 trials: ConfidenceAggregation runs inside each cell."""
    plan = _make_plan()
    results = []
    # concurrency=10: throughput jitters around 100
    for i, tput in enumerate([100.0, 105.0, 95.0]):
        r = _result(f"c10_t{i}", concurrency=10, throughput=tput, ttft_p99=50.0 + i)
        r.trial_index = i
        results.append(r)
    # concurrency=20: throughput jitters around 180
    for i, tput in enumerate([180.0, 175.0, 185.0]):
        r = _result(f"c20_t{i}", concurrency=20, throughput=tput, ttft_p99=80.0 + i)
        r.trial_index = i
        results.append(r)

    out_dir = await aggregate_sweep_and_export(results, plan, tmp_path, logger)
    assert out_dir is not None

    data = json.loads((out_dir / "profile_export_aiperf_sweep.json").read_text())
    assert len(data["per_combination_metrics"]) == 2

    # Multi-trial path: at least one metric has non-zero std.
    saw_nonzero_std = False
    for entry in data["per_combination_metrics"]:
        for stats in entry["metrics"].values():
            if stats.get("std", 0.0) > 0.0:
                saw_nonzero_std = True
    assert saw_nonzero_std, "expected aggregation across trials to produce non-zero std"


@pytest.mark.asyncio
async def test_aggregate_sweep_and_export_empty_results_no_crash(tmp_path, logger):
    """Empty results list: helper logs and returns None without writing files."""
    plan = _make_plan()

    out = await aggregate_sweep_and_export([], plan, tmp_path, logger)

    assert out is None
    assert not (tmp_path / "sweep_aggregate").exists()


def _make_plan_mode(mode: SweepMode) -> BenchmarkPlan:
    """Plan with explicit sweep iteration_order for per-variation path tests."""
    from aiperf.config.sweep import GridSweep

    cfg = BenchmarkConfig(**_MINIMAL_CONFIG_KWARGS)
    return BenchmarkPlan(
        configs=[cfg],
        sweep=GridSweep(
            parameters={"phases.profiling.concurrency": [1]},
            iteration_order=mode,
        ),
    )


def test_per_variation_aggregate_dir_independent_mode():
    """Independent mode -> ``<base>/<variation_label>/aggregate/``."""
    from pathlib import Path

    out = _per_variation_aggregate_dir(
        Path("/tmp/x"),
        "phases.profiling.concurrency=10",
        SweepMode.INDEPENDENT,
    )
    assert out == Path("/tmp/x/phases.profiling.concurrency=10/aggregate")


def test_per_variation_aggregate_dir_repeated_mode():
    """Repeated mode -> ``<base>/aggregate/<variation_label>/``."""
    from pathlib import Path

    out = _per_variation_aggregate_dir(
        Path("/tmp/x"),
        "phases.profiling.concurrency=10",
        SweepMode.REPEATED,
    )
    assert out == Path("/tmp/x/aggregate/phases.profiling.concurrency=10")


@pytest.mark.asyncio
async def test_aggregate_per_variation_writes_aggregate_per_cell_independent(
    tmp_path, logger
):
    """Independent mode: 2 variations × 3 trials -> 2 aggregate dirs under cells."""
    plan = _make_plan_mode(SweepMode.INDEPENDENT)
    results = []
    for i, tput in enumerate([100.0, 105.0, 95.0]):
        r = _result(f"c10_t{i}", concurrency=10, throughput=tput, ttft_p99=50.0 + i)
        r.variation_label = "phases.profiling.concurrency=10"
        r.variation_values = {"phases.profiling.concurrency": 10}
        r.trial_index = i
        results.append(r)
    for i, tput in enumerate([180.0, 175.0, 185.0]):
        r = _result(f"c20_t{i}", concurrency=20, throughput=tput, ttft_p99=80.0 + i)
        r.variation_label = "phases.profiling.concurrency=20"
        r.variation_values = {"phases.profiling.concurrency": 20}
        r.trial_index = i
        results.append(r)

    written = await aggregate_per_variation_and_export(results, plan, tmp_path, logger)
    assert len(written) == 2

    for concurrency in (10, 20):
        agg_dir = tmp_path / f"concurrency_{concurrency}" / "aggregate"
        agg_json = agg_dir / "profile_export_aiperf_aggregate.json"
        agg_csv = agg_dir / "profile_export_aiperf_aggregate.csv"
        assert agg_json.exists(), f"missing per-variation aggregate JSON: {agg_json}"
        assert agg_csv.exists(), f"missing per-variation aggregate CSV: {agg_csv}"

        data = json.loads(agg_json.read_text())
        # AggregateConfidenceJsonExporter flattens our AggregateResult
        # metadata into a single ``metadata`` block, with the run counts
        # bumped up under ``num_profile_runs`` / ``num_successful_runs``.
        meta = data["metadata"]
        assert meta["aggregation_type"] == "confidence"
        assert meta["num_profile_runs"] == 3
        assert meta["num_successful_runs"] == 3
        assert meta["variation_label"] == (
            f"phases.profiling.concurrency={concurrency}"
        )
        assert str(meta["sweep_mode"]).lower() == "independent"
        # The model is stamped so the plot loader can recover it for
        # aggregate-only runs (no input_config in the aggregate JSON).
        assert meta["model"] == "test-model"


@pytest.mark.asyncio
async def test_aggregate_per_variation_writes_aggregate_per_cell_repeated(
    tmp_path, logger
):
    """Repeated mode: per-variation dirs land under ``<base>/aggregate/<label>/``."""
    plan = _make_plan_mode(SweepMode.REPEATED)
    results = []
    for i, tput in enumerate([100.0, 105.0]):
        r = _result(f"c10_t{i}", concurrency=10, throughput=tput, ttft_p99=50.0 + i)
        r.variation_label = "phases.profiling.concurrency=10"
        r.variation_values = {"phases.profiling.concurrency": 10}
        r.trial_index = i
        results.append(r)
    for i, tput in enumerate([180.0, 175.0]):
        r = _result(f"c20_t{i}", concurrency=20, throughput=tput, ttft_p99=80.0 + i)
        r.variation_label = "phases.profiling.concurrency=20"
        r.variation_values = {"phases.profiling.concurrency": 20}
        r.trial_index = i
        results.append(r)

    written = await aggregate_per_variation_and_export(results, plan, tmp_path, logger)
    assert len(written) == 2

    for concurrency in (10, 20):
        agg_dir = tmp_path / "aggregate" / f"concurrency_{concurrency}"
        assert (agg_dir / "profile_export_aiperf_aggregate.json").exists()
        # Independent-mode layout must not be written.
        wrong = tmp_path / f"concurrency_{concurrency}" / "aggregate"
        assert not (wrong / "profile_export_aiperf_aggregate.json").exists()


@pytest.mark.asyncio
async def test_aggregate_per_variation_writes_single_run_in_degraded_mode(
    tmp_path, logger
):
    """Single-trial cells get a degraded-mode aggregate, mirroring sweep aggregate.

    Per round-2 R2-L10: per-variation and sweep aggregation paths must
    use the same gating. ``ConfidenceAggregation`` has a documented
    single-run mode (std=0, CI collapsed to mean, ``single_run: True``);
    skipping it here used to produce dangling references when the sweep
    summary still listed the cell.
    """
    plan = _make_plan_mode(SweepMode.INDEPENDENT)
    r = _result("c10", concurrency=10, throughput=100.0, ttft_p99=50.0)
    r.variation_label = "phases.profiling.concurrency=10"
    r.variation_values = {"phases.profiling.concurrency": 10}

    written = await aggregate_per_variation_and_export([r], plan, tmp_path, logger)
    expected = tmp_path / "concurrency_10" / "aggregate"
    assert written == [expected]
    assert (expected / "profile_export_aiperf_aggregate.json").exists()


@pytest.mark.asyncio
async def test_aggregate_per_variation_skips_when_zero_successful(tmp_path, logger):
    """Cells with 0 successful runs are skipped without crashing."""
    plan = _make_plan_mode(SweepMode.INDEPENDENT)
    r = RunResult(
        label="c10",
        success=False,
        error="timeout",
        variation_label="phases.profiling.concurrency=10",
        variation_values={"phases.profiling.concurrency": 10},
    )

    written = await aggregate_per_variation_and_export([r], plan, tmp_path, logger)
    assert written == []
    assert not (tmp_path / "phases.profiling.concurrency=10" / "aggregate").exists()


@pytest.mark.asyncio
async def test_aggregate_per_variation_handles_partial_failures_per_cell(
    tmp_path, logger
):
    """One variation fully fails; the OTHER variation still produces an aggregate."""
    plan = _make_plan_mode(SweepMode.INDEPENDENT)
    results = []
    # concurrency=10: 2 successful trials.
    for i, tput in enumerate([100.0, 105.0]):
        r = _result(f"c10_t{i}", concurrency=10, throughput=tput, ttft_p99=50.0 + i)
        r.variation_label = "phases.profiling.concurrency=10"
        r.variation_values = {"phases.profiling.concurrency": 10}
        r.trial_index = i
        results.append(r)
    # concurrency=20: both failed -> no aggregate.
    for i in range(2):
        r = _result(
            f"c20_t{i}", concurrency=20, throughput=0.0, ttft_p99=0.0, success=False
        )
        r.error = "synthetic failure"
        r.variation_label = "phases.profiling.concurrency=20"
        r.variation_values = {"phases.profiling.concurrency": 20}
        r.trial_index = i
        results.append(r)

    written = await aggregate_per_variation_and_export(results, plan, tmp_path, logger)
    assert len(written) == 1

    success_dir = tmp_path / "concurrency_10" / "aggregate"
    fail_dir = tmp_path / "concurrency_20" / "aggregate"
    assert (success_dir / "profile_export_aiperf_aggregate.json").exists()
    assert not fail_dir.exists()


# ---------------------------------------------------------------------------
# Best Configurations + Pareto stdout summary: aggregate_sweep_and_export
# echoes the structured "Best Configurations:" / "Pareto optimal points:"
# block after the JSON+CSV files are written.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aggregate_sweep_logs_best_configurations(tmp_path, caplog):
    """aggregate_sweep_and_export echoes Best Configurations after writing exporters."""
    plan = _make_plan()

    # SweepAnalyzer keys best/pareto off ``request_throughput_avg`` (maximize)
    # and ``time_to_first_token_p99`` (minimize). Build results with those exact
    # metric names so the analyzer populates best_configurations + pareto.
    def _bp_result(label: str, conc: int, tput: float, ttft_p99: float) -> RunResult:
        return RunResult(
            label=label,
            success=True,
            summary_metrics={
                "request_throughput_avg": JsonMetricResult(
                    unit="requests/sec", avg=tput, min=tput, max=tput
                ),
                "time_to_first_token_p99": JsonMetricResult(
                    unit="ms", avg=ttft_p99, p99=ttft_p99, min=ttft_p99, max=ttft_p99
                ),
            },
            variation_label=f"concurrency={conc}",
            variation_values={"concurrency": conc},
            trial_index=0,
        )

    results = [
        _bp_result("c10", 10, tput=100.0, ttft_p99=50.0),
        _bp_result("c20", 20, tput=180.0, ttft_p99=80.0),
    ]
    test_logger = logging.getLogger("aiperf.cli_runner._sweep_aggregate.best_test")
    with caplog.at_level(
        logging.INFO, logger="aiperf.cli_runner._sweep_aggregate.best_test"
    ):
        await aggregate_sweep_and_export(results, plan, tmp_path, test_logger)

    msgs = [r.message for r in caplog.records]
    assert any(m == "Best Configurations:" for m in msgs)
    # Best throughput: 180.0 reqs/s wins -> concurrency=20.
    assert any(
        m.startswith("  Best throughput: concurrency=20") and "180.00" in m
        for m in msgs
    )
    # Best latency (p99): 50ms wins -> concurrency=10.
    assert any(
        m.startswith("  Best latency (p99): concurrency=10") and "50.00" in m
        for m in msgs
    )


@pytest.mark.asyncio
async def test_aggregate_sweep_logs_pareto_optimal(tmp_path, caplog):
    """aggregate_sweep_and_export echoes the Pareto optimal points line."""
    plan = _make_plan()

    def _bp_result(label: str, conc: int, tput: float, ttft_p99: float) -> RunResult:
        return RunResult(
            label=label,
            success=True,
            summary_metrics={
                "request_throughput_avg": JsonMetricResult(
                    unit="requests/sec", avg=tput, min=tput, max=tput
                ),
                "time_to_first_token_p99": JsonMetricResult(
                    unit="ms", avg=ttft_p99, p99=ttft_p99, min=ttft_p99, max=ttft_p99
                ),
            },
            variation_label=f"concurrency={conc}",
            variation_values={"concurrency": conc},
            trial_index=0,
        )

    results = [
        _bp_result("c10", 10, tput=100.0, ttft_p99=50.0),
        _bp_result("c20", 20, tput=180.0, ttft_p99=80.0),
    ]
    test_logger = logging.getLogger("aiperf.cli_runner._sweep_aggregate.pareto_test")
    with caplog.at_level(
        logging.INFO, logger="aiperf.cli_runner._sweep_aggregate.pareto_test"
    ):
        await aggregate_sweep_and_export(results, plan, tmp_path, test_logger)

    msgs = [r.message for r in caplog.records]
    assert any(m.startswith("  Pareto optimal points:") for m in msgs)


# ---------------------------------------------------------------------------
# Wiring: _execute_multi_benchmark passes a SweepTableLogger as cell_callback
# when the suppress predicate clears, and None when it fires.
# ---------------------------------------------------------------------------


def test_execute_multi_benchmark_wires_sweep_table_logger() -> None:
    """When the suppress predicate clears, MultiRunOrchestrator gets a SweepTableLogger."""
    from aiperf.cli_runner._multi_run import _execute_multi_benchmark
    from aiperf.common.aiperf_logger import AIPerfLogger

    plan = MagicMock()
    plan.configs = [MagicMock()]
    plan.configs[0].sweeping = MagicMock(no_sweep_table=False)

    with (
        patch(
            "aiperf.cli_runner._sweep_table._should_emit_sweep_table",
            return_value=True,
        ),
        patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator") as mock_orch_cls,
        patch("aiperf.orchestrator.local_executor.LocalSubprocessExecutor"),
        patch("aiperf.cli_runner._multi_run._build_search_planner", return_value=None),
        patch("aiperf.cli_runner._multi_run._log_search_planner_active"),
        patch("asyncio.run", return_value=[]),
    ):
        _execute_multi_benchmark(plan, base_dir=MagicMock(), logger=AIPerfLogger("t"))

    kwargs = mock_orch_cls.call_args.kwargs
    assert "cell_callback" in kwargs
    assert isinstance(kwargs["cell_callback"], SweepTableLogger)


def test_execute_multi_benchmark_skips_table_when_suppressed() -> None:
    """When the suppress predicate fires, no callback is passed."""
    from aiperf.cli_runner._multi_run import _execute_multi_benchmark
    from aiperf.common.aiperf_logger import AIPerfLogger

    plan = MagicMock()
    plan.configs = [MagicMock()]
    plan.configs[0].sweeping = MagicMock(no_sweep_table=True)

    with (
        patch(
            "aiperf.cli_runner._sweep_table._should_emit_sweep_table",
            return_value=False,
        ),
        patch("aiperf.orchestrator.orchestrator.MultiRunOrchestrator") as mock_orch_cls,
        patch("aiperf.orchestrator.local_executor.LocalSubprocessExecutor"),
        patch("aiperf.cli_runner._multi_run._build_search_planner", return_value=None),
        patch("aiperf.cli_runner._multi_run._log_search_planner_active"),
        patch("asyncio.run", return_value=[]),
    ):
        _execute_multi_benchmark(plan, base_dir=MagicMock(), logger=AIPerfLogger("t"))

    kwargs = mock_orch_cls.call_args.kwargs
    assert kwargs.get("cell_callback") is None


class TestResolveModelNameForVariation:
    """Unit tests for ``_resolve_model_name_for_variation``."""

    def test__resolve_model_name_for_variation_single_config_no_variations_returns_first_model(
        self,
    ):
        from aiperf.cli_runner._sweep_aggregate import (
            _resolve_model_name_for_variation,
        )

        plan = _make_plan()  # one config with models=["test-model"], no variations
        key = ("any-label", ())

        assert _resolve_model_name_for_variation(plan, key) == "test-model"

    def test__resolve_model_name_for_variation_multi_config_matches_variation_index_returns_expected(
        self,
    ):
        from aiperf.cli_runner._sweep_aggregate import (
            _resolve_model_name_for_variation,
        )
        from aiperf.config.sweep.config import SweepVariation

        cfg_a = BenchmarkConfig(**{**_MINIMAL_CONFIG_KWARGS, "models": ["model-a"]})
        cfg_b = BenchmarkConfig(**{**_MINIMAL_CONFIG_KWARGS, "models": ["model-b"]})
        plan = BenchmarkPlan(
            configs=[cfg_a, cfg_b],
            variations=[
                SweepVariation(index=0, label="cell_a", values={}),
                SweepVariation(index=1, label="cell_b", values={}),
            ],
        )

        assert _resolve_model_name_for_variation(plan, ("cell_a", ())) == "model-a"
        assert _resolve_model_name_for_variation(plan, ("cell_b", ())) == "model-b"

    def test__resolve_model_name_for_variation_unmatched_label_falls_back_to_first_config(
        self,
    ):
        from aiperf.cli_runner._sweep_aggregate import (
            _resolve_model_name_for_variation,
        )
        from aiperf.config.sweep.config import SweepVariation

        cfg_a = BenchmarkConfig(**{**_MINIMAL_CONFIG_KWARGS, "models": ["model-a"]})
        cfg_b = BenchmarkConfig(**{**_MINIMAL_CONFIG_KWARGS, "models": ["model-b"]})
        plan = BenchmarkPlan(
            configs=[cfg_a, cfg_b],
            variations=[
                SweepVariation(index=0, label="cell_a", values={}),
                SweepVariation(index=1, label="cell_b", values={}),
            ],
        )

        # Unmatched label -> fall back to configs[0].
        assert _resolve_model_name_for_variation(plan, ("ghost_label", ())) == "model-a"
