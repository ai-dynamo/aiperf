# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component-integration: adaptive-search end-to-end with stub executor."""

from __future__ import annotations

from pathlib import Path

import pytest

skopt = pytest.importorskip("skopt")

# Imports below depend on skopt being importable. pytest.importorskip must
# precede them so the whole module is skipped when the `bo` extra is absent.
import orjson  # noqa: E402

from aiperf.common.models.export_models import JsonMetricResult  # noqa: E402
from aiperf.config.benchmark import BenchmarkRun  # noqa: E402
from aiperf.config.config import AIPerfConfig  # noqa: E402
from aiperf.config.loader.plan import build_benchmark_plan  # noqa: E402
from aiperf.orchestrator.executor import RunExecutor  # noqa: E402
from aiperf.orchestrator.models import RunResult  # noqa: E402
from aiperf.orchestrator.orchestrator import MultiRunOrchestrator  # noqa: E402
from aiperf.orchestrator.search_planner.bayesian import (  # noqa: E402
    BayesianSearchPlanner,
)

pytestmark = pytest.mark.component_integration


class _StubExecutor(RunExecutor):
    def derive_id(self, plan, var_idx, trial):
        return f"stub-v{var_idx}-t{trial}"

    async def execute(self, run: BenchmarkRun) -> RunResult:
        c = run.variation.values.get("phases.profiling.concurrency", 1)
        return RunResult(
            label=run.label,
            success=True,
            summary_metrics={
                "output_token_throughput": JsonMetricResult(
                    unit="tok/s", avg=float(c) * 5.0
                ),
            },
            artifacts_path=run.artifact_dir,
        )


@pytest.mark.asyncio
async def test_search_e2e_via_build_benchmark_plan(tmp_path: Path):
    cfg = AIPerfConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {"urls": ["http://x"], "type": "chat"},
            "datasets": [{"name": "default", "type": "synthetic"}],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "concurrency": 1,
                    "requests": 1,
                }
            ],
            "multi_run": {
                "num_runs": 1,
                "adaptive_search": {
                    "algorithm": "bayes",
                    "search_space": [
                        {
                            "path": "phases.profiling.concurrency",
                            "lo": 1,
                            "hi": 50,
                            "kind": "int",
                        },
                    ],
                    "objective_metric": "output_token_throughput",
                    "objective_stat": "avg",
                    "objective_direction": "maximize",
                    "max_iterations": 5,
                    "n_initial_points": 2,
                    "random_seed": 42,
                },
            },
        }
    )
    plan = build_benchmark_plan(cfg)
    assert plan.is_adaptive_search
    assert plan.adaptive_search.max_iterations == 5

    orch = MultiRunOrchestrator(base_dir=tmp_path)
    planner = BayesianSearchPlanner(plan.configs[0], plan.adaptive_search)
    results = await orch.execute(plan, _StubExecutor(), search_planner=planner)

    assert len(results) == 5
    assert (tmp_path / "search_history.json").exists()
    history = orjson.loads((tmp_path / "search_history.json").read_bytes())
    assert history["best"] is not None
    # With reward = concurrency * 5, the search should find positive objectives.
    # Exact convergence is not asserted to avoid skopt-version flake.
    assert history["best"]["objective_value"] > 0
