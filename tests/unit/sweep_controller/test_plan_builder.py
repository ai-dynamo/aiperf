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
        "datasets": {"main": {"type": "synthetic"}},
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
                "variables": {"random_seed": [8, 32]},
            },
            "multiRun": {"trials": 2},
            "template": {"spec": {"benchmark": _benchmark()}},
        }
    )
    plan = build_plan_from_sweep(cr)
    assert len(plan.configs) == 2
    assert len(plan.variations) == 2
    assert plan.trials == 2
    assert plan.variations[0].values == {"random_seed": 8}
    assert plan.variations[1].values == {"random_seed": 32}


def test_build_plan_no_sweep_just_multirun():
    cr = _sweep_cr(
        {
            "multiRun": {"trials": 5, "cooldownSeconds": 10},
            "template": {"spec": {"benchmark": _benchmark()}},
        }
    )
    plan = build_plan_from_sweep(cr)
    assert len(plan.configs) == 1
    assert plan.trials == 5


def test_build_plan_convergence_uses_max_runs_for_trials():
    cr = _sweep_cr(
        {
            "multiRun": {"cooldownSeconds": 30},
            "convergence": {"metric": "ttft_p99", "minRuns": 3, "maxRuns": 7},
            "template": {"spec": {"benchmark": _benchmark()}},
        }
    )
    plan = build_plan_from_sweep(cr)
    # When convergence is set, trials acts as the worst-case max.
    assert plan.trials == 7
