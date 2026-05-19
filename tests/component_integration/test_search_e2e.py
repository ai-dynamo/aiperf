# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component-integration: adaptive-search end-to-end with stub executor."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("optuna")
pytest.importorskip("botorch")

# Imports below depend on Optuna+BoTorch being importable. pytest.importorskip
# must precede them so the whole module is skipped when the optional BoTorch
# stack is absent.
import orjson  # noqa: E402

from aiperf.common.models.export_models import JsonMetricResult  # noqa: E402
from aiperf.config.config import AIPerfConfig  # noqa: E402
from aiperf.config.loader.plan import build_benchmark_plan  # noqa: E402
from aiperf.config.resolution.plan import BenchmarkRun  # noqa: E402
from aiperf.config.sweep import AdaptiveSearchSweep  # noqa: E402
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


@pytest.mark.slow
@pytest.mark.asyncio
async def test_search_e2e_via_build_benchmark_plan(tmp_path: Path):
    # Schema-2.0 shape: adaptive search lives under the envelope `sweep:` block
    # with `type: adaptive_search`; the body lives under `benchmark:`.
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
                        "hi": 50,
                        "kind": "int",
                    },
                ],
                "objectives": [
                    {
                        "metric": "output_token_throughput",
                        "stat": "avg",
                        "direction": "maximize",
                    },
                ],
                "max_iterations": 5,
                "n_initial_points": 2,
            },
            "random_seed": 42,
        }
    )
    plan = build_benchmark_plan(cfg)
    assert plan.is_adaptive_search
    assert isinstance(plan.sweep, AdaptiveSearchSweep)
    assert plan.sweep.max_iterations == 5

    orch = MultiRunOrchestrator(base_dir=tmp_path)
    planner = BayesianSearchPlanner(plan.configs[0], plan.sweep)
    results = await orch.execute(plan, _StubExecutor(), search_planner=planner)

    assert len(results) == 5
    assert (tmp_path / "search_history.json").exists()
    history = orjson.loads((tmp_path / "search_history.json").read_bytes())
    assert history["best_trials"] is not None
    # With reward = concurrency * 5, the search should find positive objectives.
    # Exact convergence is not asserted to avoid sampler-version flake.
    assert history["best_trials"][0]["objective_values"][0] > 0
