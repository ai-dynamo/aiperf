# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema-shape tests for BenchmarkPlan post-Task-3 collapse."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aiperf.config.sweep import (
    AdaptiveSearchSweep,
    Objective,
)
from aiperf.config.sweep.multi_run import ConvergenceConfig, MultiRunConfig


def test_benchmark_plan_default_sub_objects():
    from aiperf.config.resolution.plan import BenchmarkPlan

    plan = BenchmarkPlan.model_construct(
        configs=[],
        variations=[],
        trials=1,
        cooldown_seconds=0.0,
        confidence_level=0.95,
        random_seed=None,
        set_consistent_seed=True,
        disable_warmup_after_first=True,
        multi_run=MultiRunConfig(),
        sweep=None,
        failure_policy=None,
        variables={},
    )
    assert plan.multi_run.num_runs == 1
    assert plan.sweep is None
    assert plan.failure_policy is None
    assert plan.use_adaptive is False
    assert plan.is_adaptive_search is False


def test_benchmark_plan_use_adaptive_when_convergence_set():
    from aiperf.config.resolution.plan import BenchmarkPlan

    plan = BenchmarkPlan.model_construct(
        configs=[],
        variations=[],
        trials=1,
        cooldown_seconds=0.0,
        confidence_level=0.95,
        random_seed=None,
        set_consistent_seed=True,
        disable_warmup_after_first=True,
        multi_run=MultiRunConfig(
            num_runs=10,
            convergence=ConvergenceConfig(metric="ttft"),
        ),
        sweep=None,
        failure_policy=None,
        variables={},
    )
    assert plan.use_adaptive is True
    assert plan.is_adaptive_search is False


def test_benchmark_plan_is_adaptive_search_when_sweep_is_adaptive():
    from aiperf.config.resolution.plan import BenchmarkPlan

    sweep = AdaptiveSearchSweep(
        search_space=[
            {"path": "phases.profiling.concurrency", "lo": 1, "hi": 1000, "kind": "int"}
        ],
        objectives=[Objective(metric="m", direction="maximize")],
        max_iterations=10,
    )
    plan = BenchmarkPlan.model_construct(
        configs=[],
        variations=[],
        trials=1,
        cooldown_seconds=0.0,
        confidence_level=0.95,
        random_seed=None,
        set_consistent_seed=True,
        disable_warmup_after_first=True,
        multi_run=MultiRunConfig(),
        sweep=sweep,
        failure_policy=None,
        variables={},
    )
    assert plan.is_adaptive_search is True
    assert plan.use_adaptive is False


def test_benchmark_plan_rejects_old_flat_fields():
    """Construction with removed fields fails (model uses extra='forbid')."""
    from aiperf.config.resolution.plan import BenchmarkPlan

    # We don't bother building a valid BenchmarkConfig; even if other
    # validation errors fire, extra='forbid' must report the removed
    # field as a ValidationError.
    with pytest.raises(ValidationError, match=r"convergence_metric") as exc_info:
        BenchmarkPlan(
            configs=[],
            variations=[],
            trials=1,
            cooldown_seconds=0.0,
            confidence_level=0.95,
            random_seed=None,
            set_consistent_seed=True,
            disable_warmup_after_first=True,
            multi_run=MultiRunConfig(),
            sweep=None,
            failure_policy=None,
            variables={},
            convergence_metric="ttft",  # removed field
        )
    assert "convergence_metric" in str(exc_info.value)


# ============================================================
# Validation-respecting siblings — these go through full
# AIPerfConfig.model_validate -> build_benchmark_plan instead of
# BenchmarkPlan.model_construct, so annotation drift on the body
# fields surfaces as a real schema error instead of being silently
# bypassed by model_construct.
# ============================================================


def _aiperf_config(**envelope_overrides):
    from aiperf.config.config import AIPerfConfig

    body = {
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
                "requests": 10,
                "concurrency": 1,
            }
        ],
    }
    return AIPerfConfig.model_validate({"benchmark": body, **envelope_overrides})


def test_benchmark_plan_default_sub_objects_via_validation():
    from aiperf.config.loader.plan import build_benchmark_plan

    plan = build_benchmark_plan(_aiperf_config())
    assert plan.multi_run.num_runs == 1
    assert plan.sweep is None
    assert plan.use_adaptive is False
    assert plan.is_adaptive_search is False


def test_benchmark_plan_use_adaptive_when_convergence_set_via_validation():
    from aiperf.config.loader.plan import build_benchmark_plan

    cfg = _aiperf_config(
        multi_run={
            "num_runs": 10,
            "convergence": {"metric": "ttft"},
        }
    )
    plan = build_benchmark_plan(cfg)
    assert plan.use_adaptive is True
    assert plan.is_adaptive_search is False


def test_benchmark_plan_is_adaptive_search_when_sweep_is_adaptive_via_validation():
    from aiperf.config.loader.plan import build_benchmark_plan

    cfg = _aiperf_config(
        sweep={
            "type": "adaptive_search",
            "search_space": [
                {
                    "path": "phases.profiling.concurrency",
                    "lo": 1,
                    "hi": 1000,
                    "kind": "int",
                }
            ],
            "objectives": [{"metric": "m", "direction": "maximize"}],
            "max_iterations": 10,
        }
    )
    plan = build_benchmark_plan(cfg)
    assert plan.is_adaptive_search is True
    assert plan.use_adaptive is False
