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

skopt = pytest.importorskip("skopt")

# Imports below depend on skopt being importable; pytest.importorskip must
# precede them so the whole module is skipped when the `bo` extra is absent.
import orjson  # noqa: E402

from aiperf.common.models.export_models import JsonMetricResult  # noqa: E402
from aiperf.config.benchmark import BenchmarkRun  # noqa: E402
from aiperf.kubernetes.sweep_models import AIPerfSweepSpec  # noqa: E402
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
    return {
        "metadata": {
            "name": "test-bo-sweep",
            "namespace": "default",
            "uid": "abc-123",
        },
        "spec": {
            "multiRun": {
                "trials": 1,
                "adaptiveSearch": {
                    "algorithm": "bayes",
                    "searchSpace": [
                        {
                            "path": "phases.profiling.concurrency",
                            "lo": 1,
                            "hi": 50,
                            "kind": "int",
                        },
                    ],
                    "objectiveMetric": "output_token_throughput",
                    "objectiveStat": "avg",
                    "objectiveDirection": "maximize",
                    "maxIterations": 5,
                    "nInitialPoints": 2,
                    "randomSeed": 42,
                },
            },
            "template": {"spec": {"benchmark": _benchmark()}},
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

    # Plan must carry adaptive_search through from spec.multiRun.adaptiveSearch
    # so the orchestrator's adaptive-dispatch path fires.
    assert plan.is_adaptive_search, (
        "build_plan_from_sweep must propagate multi_run.adaptive_search to "
        "plan.adaptive_search; otherwise the orchestrator runs grid mode."
    )
    assert plan.adaptive_search.max_iterations == 5
    assert plan.adaptive_search.objective_metric == "output_token_throughput"

    orch = MultiRunOrchestrator(base_dir=tmp_path)
    planner = BayesianSearchPlanner(plan.configs[0], plan.adaptive_search)
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
    _write_sweep_parent_aggregate(
        base_dir=tmp_path,
        sweep_cr=cr,
        spec=spec,
        results=results,
        plan=plan,
        sweep_run_epoch="1",
        with_trial_suffix=False,
    )
    children_path = (
        tmp_path
        / cr["metadata"]["namespace"]
        / "sweeps"
        / cr["metadata"]["name"]
        / "1"
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
