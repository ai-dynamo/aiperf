# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for plan-driven CLI runner aggregation helpers."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
from aiperf.config.sweep import GridSweep
from aiperf.orchestrator.models import RunResult

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
