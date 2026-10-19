# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for plan-driven CLI runner aggregation helpers."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
from pytest import param

from aiperf.cli_runner._multi_run import _run_multi_benchmark, _summarize_and_export
from aiperf.cli_runner._sweep_aggregate import (
    _aggregate_group_to_stats,
    _build_per_combination_stats,
    _build_sweep_aggregate_result,
    _compute_sweep_parameters,
    _group_results_by_variation,
    _per_variation_aggregate_dir,
)
from aiperf.common.enums import SweepMode
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.config import BenchmarkConfig, BenchmarkPlan
from aiperf.config.sweep import AdaptiveSearchSweep, GridSweep, Objective
from aiperf.config.sweep.adaptive import SearchSpaceDimension, SLAFilter
from aiperf.orchestrator.models import RunResult
from aiperf.orchestrator.strategies import FixedTrialsStrategy
from aiperf.plugin.enums import UIType

_MINIMAL_CONFIG = {
    "models": ["test-model"],
    "endpoint": {
        "urls": ["http://localhost:8000/v1/chat/completions"],
        "wait_for_model_timeout": 0,
    },
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


def _make_plan(*, iteration_order: SweepMode = SweepMode.INDEPENDENT) -> BenchmarkPlan:
    cfg = BenchmarkConfig.model_validate(_MINIMAL_CONFIG)
    return BenchmarkPlan(
        configs=[cfg, cfg.model_copy(deep=True)],
        trials=2,
        confidence_level=0.95,
        cooldown_seconds=10,
        sweep=GridSweep(
            parameters={"phases.profiling.concurrency": [10, 20]},
            iteration_order=iteration_order,
        ),
    )


def _successful_result(
    *,
    label: str,
    concurrency: int,
    throughput: float,
    trial_index: int = 0,
) -> RunResult:
    return RunResult(
        label=label,
        success=True,
        summary_metrics={
            "request_throughput": JsonMetricResult(
                unit="requests/sec",
                avg=throughput,
                min=throughput,
                max=throughput,
            ),
            "time_to_first_token": JsonMetricResult(unit="ms", avg=50.0 + concurrency),
        },
        variation_label=f"phases.profiling.concurrency={concurrency}",
        variation_values={"concurrency": concurrency},
        trial_index=trial_index,
    )


@pytest.fixture
def sweep_results() -> list[RunResult]:
    return [
        _successful_result(
            label="c10_t0", concurrency=10, throughput=100.0, trial_index=0
        ),
        _successful_result(
            label="c10_t1", concurrency=10, throughput=102.0, trial_index=1
        ),
        _successful_result(
            label="c20_t0", concurrency=20, throughput=200.0, trial_index=0
        ),
        RunResult(
            label="c20_t1",
            success=False,
            error="timeout",
            variation_label="phases.profiling.concurrency=20",
            variation_values={"concurrency": 20},
            trial_index=1,
        ),
    ]


class TestSweepGrouping:
    """Test grouping and parameter extraction for sweep aggregation."""

    def test_group_results_by_variation_preserves_first_seen_order(
        self, sweep_results: list[RunResult]
    ):
        groups = _group_results_by_variation(sweep_results)

        # Keys are (variation_label, sorted_values_tuple) so QMC-collision
        # cells stay distinct even when their ``values`` happen to match.
        c10_key = ("phases.profiling.concurrency=10", (("concurrency", 10),))
        c20_key = ("phases.profiling.concurrency=20", (("concurrency", 20),))
        assert list(groups) == [c10_key, c20_key]
        assert [r.label for r in groups[c10_key]] == ["c10_t0", "c10_t1"]

    def test_compute_sweep_parameters_returns_values_in_result_order(
        self, sweep_results: list[RunResult]
    ):
        groups = _group_results_by_variation(sweep_results)

        assert _compute_sweep_parameters(groups) == [
            {"name": "concurrency", "values": [10, 20]}
        ]

    def test_colliding_leaf_names_preserve_dotted_parameters(self):
        results = [
            RunResult(
                label="warmup1_profile3",
                success=True,
                summary_metrics={
                    "request_throughput": JsonMetricResult(
                        unit="requests/sec",
                        avg=100.0,
                        min=100.0,
                        max=100.0,
                    )
                },
                variation_label="phases.profiling.concurrency=3, phases.warmup.concurrency=1",
                variation_values={
                    "phases.profiling.concurrency": 3,
                    "phases.warmup.concurrency": 1,
                },
            ),
            RunResult(
                label="warmup2_profile4",
                success=True,
                summary_metrics={
                    "request_throughput": JsonMetricResult(
                        unit="requests/sec",
                        avg=200.0,
                        min=200.0,
                        max=200.0,
                    )
                },
                variation_label="phases.profiling.concurrency=4, phases.warmup.concurrency=2",
                variation_values={
                    "phases.profiling.concurrency": 4,
                    "phases.warmup.concurrency": 2,
                },
            ),
        ]
        groups = _group_results_by_variation(results)

        assert _compute_sweep_parameters(groups) == [
            {"name": "phases.profiling.concurrency", "values": [3, 4]},
            {"name": "phases.warmup.concurrency", "values": [1, 2]},
        ]
        combos = [
            combo.to_dict() for combo in _build_per_combination_stats(groups, 0.95)
        ]
        assert combos == [
            {"phases.profiling.concurrency": 3, "phases.warmup.concurrency": 1},
            {"phases.profiling.concurrency": 4, "phases.warmup.concurrency": 2},
        ]


class TestSweepStats:
    """Test per-variation stat reduction used by sweep aggregation."""

    def test_single_successful_run_uses_summary_metrics(self):
        result = _successful_result(label="c10_t0", concurrency=10, throughput=100.0)

        stats = _aggregate_group_to_stats([result], confidence_level=0.95)

        assert stats is not None
        assert stats["request_throughput"]["mean"] == 100.0
        assert stats["request_throughput"]["std"] == 0.0

    def test_failed_single_run_returns_none(self):
        result = RunResult(
            label="failed",
            success=False,
            error="timeout",
            variation_values={"concurrency": 10},
        )

        assert _aggregate_group_to_stats([result], confidence_level=0.95) is None

    def test_multi_trial_group_uses_confidence_aggregation(
        self, sweep_results: list[RunResult]
    ):
        c10_key = ("phases.profiling.concurrency=10", (("concurrency", 10),))
        group = _group_results_by_variation(sweep_results)[c10_key]

        stats = _aggregate_group_to_stats(group, confidence_level=0.95)

        assert stats is not None
        assert "request_throughput_avg" in stats
        assert stats["request_throughput_avg"]["mean"] == pytest.approx(101.0)


class TestSweepExportHelpers:
    """Test export-oriented sweep aggregation helpers."""

    @pytest.mark.parametrize(
        "mode,expected",
        [
            (SweepMode.REPEATED, Path("/tmp/base/aggregate/concurrency=10")),
            (SweepMode.INDEPENDENT, Path("/tmp/base/concurrency=10/aggregate")),
        ],
    )
    def test_per_variation_aggregate_dir_depends_on_iteration_order(
        self, mode: SweepMode, expected: Path
    ):
        assert (
            _per_variation_aggregate_dir(Path("/tmp/base"), "concurrency=10", mode)
            == expected
        )

    def test_build_sweep_aggregate_result_includes_failure_metadata(
        self, sweep_results: list[RunResult]
    ):
        sweep_dict = {
            "metadata": {"parameters": [{"name": "concurrency", "values": [10, 20]}]},
            "best_configurations": {
                "best_throughput": {"parameters": {"concurrency": 20}}
            },
            "pareto_optimal": [{"concurrency": 20}],
            "per_combination_metrics": [{"parameters": {"concurrency": 10}}],
        }

        aggregate = _build_sweep_aggregate_result(sweep_results, sweep_dict)

        assert aggregate.aggregation_type == "sweep"
        assert aggregate.num_runs == 4
        assert aggregate.num_successful_runs == 3
        assert aggregate.failed_runs == [{"label": "c20_t1", "error": "timeout"}]
        assert "best_configurations" in aggregate.metadata
        assert "pareto_optimal" in aggregate.metadata

    @patch(
        "aiperf.cli_runner._post_process.export_sweep_aggregate", new_callable=AsyncMock
    )
    @patch("aiperf.orchestrator.aggregation.sweep.SweepAnalyzer.compute")
    @pytest.mark.asyncio
    async def test_aggregate_sweep_and_export_writes_sweep_aggregate(
        self,
        mock_compute: MagicMock,
        mock_export: AsyncMock,
        sweep_results: list[RunResult],
        tmp_path: Path,
    ):
        from aiperf.cli_runner._sweep_aggregate import aggregate_sweep_and_export

        mock_compute.return_value = {
            "metadata": {},
            "best_configurations": {},
            "pareto_optimal": [],
            "per_combination_metrics": [],
        }
        logger = MagicMock()

        aggregate_dir = await aggregate_sweep_and_export(
            sweep_results,
            _make_plan(),
            tmp_path,
            logger,
        )

        assert aggregate_dir == tmp_path / "sweep_aggregate"
        mock_compute.assert_called_once()
        mock_export.assert_awaited_once()


class TestSummarizeAndExport:
    @pytest.mark.parametrize(
        "mode,through_cli",
        [
            param("single", False, id="single-config"),
            param("grid", False, id="single-point-grid"),
            param("search", True, id="one-point-search-cli-with-no-feasible-point"),
        ],
    )  # fmt: skip
    def test_single_success_exports_and_returns_zero(
        self, mode: str, through_cli: bool, tmp_path: Path
    ) -> None:
        """SLA infeasibility is a search outcome, not a benchmark execution failure."""
        sweep = None
        if mode == "grid":
            sweep = GridSweep(parameters={"phases.profiling.concurrency": [10]})
        elif mode == "search":
            sweep = AdaptiveSearchSweep(
                search_space=[
                    SearchSpaceDimension(
                        path="phases.profiling.concurrency", lo=1, hi=100, kind="int"
                    )
                ],
                objectives=[
                    Objective(
                        metric="request_throughput", stat="avg", direction="maximize"
                    )
                ],
                max_iterations=30,
                sla_filters=[
                    SLAFilter(
                        metric_tag="time_to_first_token",
                        stat="avg",
                        op="lt",
                        threshold=1,
                    )
                ],
            )
        plan = BenchmarkPlan(
            configs=[
                BenchmarkConfig.model_validate(
                    {**_MINIMAL_CONFIG, "runtime": {"ui": UIType.SIMPLE}}
                )
            ],
            trials=1,
            sweep=sweep,
        )
        result = _successful_result(label="c10_t0", concurrency=10, throughput=100.0)

        if through_cli:
            with (
                patch("aiperf.common.logging.setup_rich_logging"),
                patch(
                    "aiperf.cli_runner._multi_run._estimate_and_log_duration",
                    return_value=tmp_path,
                ),
                patch(
                    "aiperf.cli_runner._multi_run._execute_multi_benchmark",
                    return_value=[result],
                ) as execute,
                patch("aiperf.cli_runner._multi_run.os._exit") as exit_process,
            ):
                _run_multi_benchmark(plan)

            execute.assert_called_once()
            assert execute.call_args.args[:2] == (plan, tmp_path)
            exit_process.assert_called_once_with(0)
        else:
            exit_code = _summarize_and_export(
                plan,
                [result],
                total_runs=1,
                strategy=FixedTrialsStrategy(num_trials=1),
                base_dir=tmp_path,
                logger=MagicMock(),
            )
            assert exit_code == 0
        aggregates = list(tmp_path.rglob("profile_export_aiperf_aggregate.json"))
        assert len(aggregates) == 1
        aggregate = orjson.loads(aggregates[0].read_bytes())
        assert aggregate["metadata"]["num_successful_runs"] == 1
        assert aggregate["metadata"]["single_run"] is True
        assert aggregates[0].with_suffix(".csv").is_file()
        if mode == "search":
            sweep_path = (
                tmp_path / "sweep_aggregate" / "profile_export_aiperf_sweep.json"
            )
            summary = orjson.loads(sweep_path.read_bytes())
            assert summary["num_successful_runs"] == 1
            assert len(summary["per_combination_metrics"]) == 1
            assert summary["metadata"]["sla_constraints"]["feasible_count"] == 0
            assert summary["metadata"]["sla_constraints"]["infeasible_count"] == 1
            assert sweep_path.with_suffix(".csv").is_file()

    @pytest.mark.parametrize(
        "successes,expected_exit,expected_aggregations",
        [
            param([False], 1, 0, id="all-failed"),
            param([True, False], 1, 0, id="one-success-with-failure"),
            param([True, True], 0, 1, id="two-successes"),
            param([True, True, False], 0, 1, id="two-successes-with-failure"),
        ],
    )  # fmt: skip
    def test_existing_failure_and_multi_run_behavior_is_preserved(
        self,
        successes: list[bool],
        expected_exit: int,
        expected_aggregations: int,
        tmp_path: Path,
    ) -> None:
        """One success with a failed trial must not enter singleton aggregation."""
        results = [
            RunResult(label=f"run_{i}", success=success)
            for i, success in enumerate(successes)
        ]
        with patch(
            "aiperf.cli_runner._multi_run.aggregate_plan_results",
            new_callable=AsyncMock,
        ) as aggregate:
            exit_code = _summarize_and_export(
                _make_plan(),
                results,
                total_runs=len(results),
                strategy=FixedTrialsStrategy(num_trials=1),
                base_dir=tmp_path,
                logger=MagicMock(),
            )

        assert exit_code == expected_exit
        assert aggregate.await_count == expected_aggregations

    def test_single_success_propagates_aggregation_errors(self, tmp_path: Path) -> None:
        """Successful benchmark execution does not imply aggregation completed."""
        result = _successful_result(label="c10_t0", concurrency=10, throughput=100.0)
        with (
            patch(
                "aiperf.cli_runner._multi_run.aggregate_plan_results",
                new_callable=AsyncMock,
                side_effect=ValueError("aggregation failed"),
            ),
            pytest.raises(ValueError, match="aggregation failed"),
        ):
            _summarize_and_export(
                _make_plan(),
                [result],
                total_runs=1,
                strategy=FixedTrialsStrategy(num_trials=1),
                base_dir=tmp_path,
                logger=MagicMock(),
            )
