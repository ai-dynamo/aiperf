# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component-integration: cluster-side adaptive search end-to-end.

Drives the same flow `sweep_controller.main` runs (build_plan_from_sweep ->
BayesianSearchPlanner -> MultiRunOrchestrator.execute) but with a stub
`RunExecutor` instead of `K8sChildJobExecutor`, so we don't need a live
apiserver. Proves the controller-pod's BO wiring is correct end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("optuna")

# Imports below depend on optuna being importable (BayesianSearchPlanner
# subclasses OptunaSearchPlanner); pytest.importorskip must precede them so
# the whole module is skipped when the `optuna`/`botorch` extra is absent.
import orjson  # noqa: E402

from aiperf.common.models.export_models import JsonMetricResult  # noqa: E402
from aiperf.config.resolution.plan import BenchmarkRun  # noqa: E402
from aiperf.config.sweep import AdaptiveSearchSweep  # noqa: E402
from aiperf.operator.models import AIPerfSweepSpec  # noqa: E402
from aiperf.orchestrator.executor import RunExecutor  # noqa: E402
from aiperf.orchestrator.models import RunResult  # noqa: E402
from aiperf.orchestrator.orchestrator import MultiRunOrchestrator  # noqa: E402
from aiperf.orchestrator.search_planner.bayesian import (  # noqa: E402
    BayesianSearchPlanner,
)
from aiperf.sweep_controller.main import (  # noqa: E402
    _write_sweep_parent_aggregate,
)
from aiperf.sweep_controller.plan_builder import build_plan_from_sweep  # noqa: E402

pytestmark = pytest.mark.component_integration


class _StubK8sExecutor(RunExecutor):
    """Stand-in for K8sChildJobExecutor that returns synthetic results.

    Mirrors the in-process `_StubExecutor` from `test_search_e2e.py`. Returns
    a deterministic objective so the BO loop has a meaningful gradient.
    """

    def derive_id(self, plan, var_idx, trial):
        return f"stub-cluster-v{var_idx}-t{trial}"

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


def _benchmark() -> dict:
    return {
        "models": "mock",
        "endpoint": {"urls": ["http://x:8000/v1/chat/completions"]},
        "datasets": [{"name": "main", "type": "synthetic"}],
        "phases": [
            {
                "name": "profiling",
                "type": "concurrency",
                "requests": 1,
                "concurrency": 1,
            }
        ],
    }


def _sweep_cr_with_adaptive_search() -> dict:
    """Schema-2.0 flat-envelope shape: spec.{benchmark, sweep, multi_run, ...}.

    The legacy shape used `template.spec.benchmark` and `multi_run.adaptive_search`;
    those wrappers are gone. Adaptive search now lives on the envelope `sweep:`
    block as `type: adaptive_search`.
    """
    return {
        "metadata": {
            "name": "test-bo-sweep",
            "namespace": "default",
            "uid": "abc-123",
        },
        "spec": {
            "benchmark": _benchmark(),
            "multiRun": {"numRuns": 1},
            "sweep": {
                "type": "adaptive_search",
                "planner": "bayesian",
                "searchSpace": [
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
                "maxIterations": 5,
                "nInitialPoints": 2,
            },
            "randomSeed": 42,
            "image": "aiperf:test",
        },
    }


@pytest.mark.asyncio
async def test_cluster_search_e2e_via_build_plan_from_sweep(tmp_path: Path):
    """Build a plan from an AIPerfSweep CR with adaptive_search, run it.

    This is the cluster-path equivalent of test_search_e2e.py's in-process
    test. We drive the orchestrator the same way `sweep_controller.main`
    does, just with a stub executor.
    """
    cr = _sweep_cr_with_adaptive_search()
    plan = build_plan_from_sweep(cr)

    # Plan must carry adaptive search through from spec.sweep so the
    # orchestrator's adaptive-dispatch path fires.
    assert plan.is_adaptive_search, (
        "build_plan_from_sweep must propagate spec.sweep (adaptive_search) "
        "to plan.sweep; otherwise the orchestrator runs grid mode."
    )
    assert isinstance(plan.sweep, AdaptiveSearchSweep)
    assert plan.sweep.max_iterations == 5
    assert plan.sweep.objectives[0].metric == "output_token_throughput"

    orch = MultiRunOrchestrator(base_dir=tmp_path)
    planner = BayesianSearchPlanner(plan.configs[0], plan.sweep)
    results = await orch.execute(plan, _StubK8sExecutor(), search_planner=planner)

    assert len(results) == 5, "BO should run exactly max_iterations iterations"
    history_path = tmp_path / "search_history.json"
    assert history_path.exists(), "execute_adaptive_search must persist history"
    history = orjson.loads(history_path.read_bytes())
    assert history["best"] is not None
    # reward = concurrency * 5 — should produce positive objectives.
    assert history["best"]["objective_value"] > 0

    # children.json must enumerate exactly max_iterations x trials entries —
    # results-driven, not derived from plan.variations (which is a length-1
    # placeholder for adaptive search).
    spec = AIPerfSweepSpec.model_validate(cr["spec"])
    # write_sweep_latest validates the epoch against EPOCH_RE (9-20 decimal
    # digits) before advancing latest.txt, so the run epoch must be a realistic
    # epoch-seconds value, not a bare "1".
    sweep_run_epoch = "1714069323"
    _write_sweep_parent_aggregate(
        base_dir=tmp_path,
        sweep_cr=cr,
        spec=spec,
        results=results,
        plan=plan,
        sweep_run_epoch=sweep_run_epoch,
        with_trial_suffix=False,
    )
    children_path = (
        tmp_path
        / cr["metadata"]["namespace"]
        / "sweeps"
        / cr["metadata"]["name"]
        / sweep_run_epoch
        / "children.json"
    )
    assert children_path.exists(), "children.json must be written"
    children_doc = orjson.loads(children_path.read_bytes())
    children = children_doc["children"]
    # max_iterations=5 x trials=1 = 5 entries.
    assert len(children) == 5, (
        f"children.json must list exactly max_iterations x trials entries; "
        f"got {len(children)}: {children}"
    )
    # Each child must carry a distinct variation_index drawn from the actual
    # results stream, not from the placeholder plan.variations[0].
    variation_indices = [c["variation_index"] for c in children]
    assert len(set(variation_indices)) == 5, (
        f"each adaptive iteration must produce a distinct variation_index; "
        f"got {variation_indices}"
    )
