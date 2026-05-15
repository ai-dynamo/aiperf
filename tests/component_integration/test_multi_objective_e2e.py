# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end multi-objective qNEHVI search through the orchestrator.

Mirrors the in-process stub-executor pattern used by ``test_search_e2e.py``
so the smoke is hermetic: no live mock server, no network. The point is to
wire the multi-objective Optuna+BoTorch path end-to-end -- schema ->
``build_benchmark_plan`` -> ``OptunaSearchPlanner`` with
``optuna_sampler="botorch"`` + ``optuna_acquisition="qlognehvi"`` ->
``MultiRunOrchestrator.execute`` -> ``search_history.json`` exporter -- and
check that the Pareto-front shape lands on disk as documented.

The synthetic surface is a deliberate trade-off in concurrency: throughput
saturates while TTFT grows linearly. With both objectives active, no single
``c`` dominates -- the optimizer should accumulate a non-trivial Pareto
front rather than collapsing to one point.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

pytest.importorskip("botorch")
pytest.importorskip("optuna")

import orjson  # noqa: E402

from aiperf.common.models.export_models import JsonMetricResult  # noqa: E402
from aiperf.config.config import AIPerfConfig  # noqa: E402
from aiperf.config.loader.plan import build_benchmark_plan  # noqa: E402
from aiperf.config.resolution.plan import BenchmarkRun  # noqa: E402
from aiperf.config.sweep import AdaptiveSearchSweep  # noqa: E402
from aiperf.orchestrator.executor import RunExecutor  # noqa: E402
from aiperf.orchestrator.models import RunResult  # noqa: E402
from aiperf.orchestrator.orchestrator import MultiRunOrchestrator  # noqa: E402
from aiperf.orchestrator.search_planner.optuna_planner import (  # noqa: E402
    OptunaSearchPlanner,
)

pytestmark = pytest.mark.component_integration


class _ParetoSurfaceStubExecutor(RunExecutor):
    """Two-objective synthetic surface with a real throughput/TTFT trade-off.

    - ``output_token_throughput`` (MAXIMIZE) saturates: ``100 * (1 - exp(-c/10))``.
    - ``time_to_first_token`` (MINIMIZE) grows linearly: ``10 + 2*c``.

    Low ``c`` wins on TTFT; high ``c`` wins on throughput. No single ``c``
    dominates, so the Pareto front contains multiple distinct points.
    """

    def derive_id(self, plan, var_idx, trial):
        return f"stub-v{var_idx}-t{trial}"

    async def execute(self, run: BenchmarkRun) -> RunResult:
        c = float(run.variation.values.get("phases.profiling.concurrency", 1))
        run.artifact_dir.mkdir(parents=True, exist_ok=True)
        throughput = 100.0 * (1.0 - math.exp(-c / 10.0))
        ttft = 10.0 + 2.0 * c
        return RunResult(
            label=run.label,
            success=True,
            summary_metrics={
                "output_token_throughput": JsonMetricResult(
                    unit="tok/s", avg=throughput
                ),
                "time_to_first_token": JsonMetricResult(unit="ms", avg=ttft),
            },
            artifacts_path=run.artifact_dir,
        )


@pytest.mark.asyncio
async def test_qlognehvi_two_objective_search(tmp_path: Path):
    """qNEHVI two-objective sweep emits a multi-point Pareto front to disk."""
    cfg = AIPerfConfig.model_validate(
        {
            "benchmark": {
                "models": ["m"],
                "endpoint": {"urls": ["http://x"], "type": "chat"},
                "datasets": [{"name": "profiling", "type": "synthetic"}],
                "phases": [
                    {
                        "name": "profiling",
                        "type": "concurrency",
                        "concurrency": 1,
                        "requests": 1,
                    }
                ],
            },
            "multi_run": {"num_runs": 1},
            "sweep": {
                "type": "adaptive_search",
                "search_space": [
                    {
                        "path": "phases.profiling.concurrency",
                        "lo": 1,
                        "hi": 20,
                        "kind": "int",
                    },
                ],
                "objectives": [
                    {
                        "metric": "output_token_throughput",
                        "stat": "avg",
                        "direction": "maximize",
                    },
                    {
                        "metric": "time_to_first_token",
                        "stat": "avg",
                        "direction": "minimize",
                    },
                ],
                "max_iterations": 8,
                "n_initial_points": 3,
                "improvement_patience": 4,
                "optuna_sampler": "botorch",
                "optuna_acquisition": "qlognehvi",
                "random_seed": 7,
            },
            "random_seed": 7,
        }
    )
    plan = build_benchmark_plan(cfg)
    assert plan.is_adaptive_search
    assert isinstance(plan.sweep, AdaptiveSearchSweep)

    orch = MultiRunOrchestrator(base_dir=tmp_path)
    planner = OptunaSearchPlanner(plan.configs[0], plan.sweep)
    await orch.execute(plan, _ParetoSurfaceStubExecutor(), search_planner=planner)

    history_path = tmp_path / "search_history.json"
    assert history_path.exists()
    payload = orjson.loads(history_path.read_bytes())

    assert payload["config"]["objectives"][0]["metric"] == "output_token_throughput"
    assert payload["config"]["objectives"][1]["metric"] == "time_to_first_token"

    front = payload["best_trials"]
    assert len(front) >= 1
    assert all(p["pareto_rank"] == 0 for p in front)
    assert all(
        "objective_values" in p and len(p["objective_values"]) == 2 for p in front
    )

    # Regression: assert qLogNEHVI (not Optuna's default qEHVI) actually got
    # installed onto the BoTorchSampler. Optuna stores the callable as the
    # private `_candidates_func`; writing the public `candidates_func`
    # is a silent no-op that lets `qExpectedHypervolumeImprovement` run
    # instead -- the warning at botorch monte_carlo.py:110 is the smoke.
    sampler = planner._study.sampler
    installed = sampler._candidates_func
    assert installed is not None, "BoTorchSampler._candidates_func not installed"
    assert "qnehvi_candidates_func" in getattr(installed, "__qualname__", ""), (
        f"expected qLogNEHVI candidates_func, got {installed!r}"
    )
