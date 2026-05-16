# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.sweep_controller.plan_builder import build_plan_from_sweep


def _sweep_cr(spec: dict) -> dict:
    return {
        "metadata": {"name": "test-sweep", "namespace": "default", "uid": "abc"},
        "spec": spec,
    }


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


def test_build_plan_grid_sweep():
    cr = _sweep_cr(
        {
            "sweep": {
                "type": "grid",
                "variables": {"benchmark.phases.profiling.concurrency": [8, 32]},
            },
            "multiRun": {"numRuns": 2},
            "benchmark": _benchmark(),
        }
    )
    plan = build_plan_from_sweep(cr)
    assert len(plan.configs) == 2
    assert len(plan.variations) == 2
    assert plan.trials == 2
    assert plan.variations[0].values == {"phases.profiling.concurrency": 8}
    assert plan.variations[1].values == {"phases.profiling.concurrency": 32}


def test_build_plan_no_sweep_just_multirun():
    cr = _sweep_cr(
        {
            "sweep": {
                "type": "grid",
                "variables": {"benchmark.phases.profiling.concurrency": [16]},
            },
            "multiRun": {"numRuns": 5, "cooldownSeconds": 10},
            "benchmark": _benchmark(),
        }
    )
    plan = build_plan_from_sweep(cr)
    assert len(plan.configs) == 1
    assert plan.trials == 5


def test_build_plan_convergence_uses_num_runs_for_trials():
    cr = _sweep_cr(
        {
            "sweep": {
                "type": "grid",
                "variables": {"benchmark.phases.profiling.concurrency": [16]},
                "iteration_order": "independent",
            },
            "multiRun": {
                "numRuns": 7,
                "cooldownSeconds": 30,
                "convergence": {"metric": "ttft_p99", "threshold": 0.05},
            },
            "benchmark": _benchmark(),
        }
    )
    plan = build_plan_from_sweep(cr)
    # Convergence early-stops within numRuns; numRuns is the worst-case cap.
    assert plan.trials == 7
    assert plan.multi_run.convergence is not None
    assert plan.multi_run.convergence.metric == "ttft_p99"
    assert plan.multi_run.convergence.threshold == 0.05
