# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a BenchmarkPlan from an AIPerfSweep CR dict.

Reuses aiperf.config.sweep.expand_sweep to produce variations from the
template's benchmark config plus the sweep config from the CR's spec.
"""

from __future__ import annotations

from typing import Any

from aiperf.config.benchmark import BenchmarkPlan
from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import expand_sweep
from aiperf.kubernetes.sweep_models import AIPerfSweepSpec

__all__ = ["build_plan_from_sweep"]


def build_plan_from_sweep(sweep_cr: dict[str, Any]) -> BenchmarkPlan:
    """Construct a BenchmarkPlan from an AIPerfSweep CR.

    Args:
        sweep_cr: Raw AIPerfSweep dict (typically from kubernetes_asyncio read).

    Returns:
        BenchmarkPlan with one config per variation, trial count from
        spec.multiRun (or convergence.maxRuns when convergence is set).

    Raises:
        ValidationError: If the CR spec fails Pydantic validation.
    """
    spec_dict = sweep_cr["spec"]
    spec = AIPerfSweepSpec.model_validate(spec_dict)

    benchmark_body = spec.template.spec.benchmark.model_dump(
        by_alias=True, exclude_none=True, exclude_defaults=True
    )
    # Build the envelope dict that expand_sweep expects: body under
    # `benchmark`, cross-variation fields (sweep, variables, random_seed)
    # at envelope level. multi_run is handled separately via plan_kwargs.
    sweep_input: dict[str, Any] = {"benchmark": benchmark_body}
    if spec.sweep is not None:
        sweep_input["sweep"] = spec.sweep.model_dump(by_alias=True)
    if spec.variables:
        sweep_input["variables"] = spec.variables
    if spec.random_seed is not None:
        sweep_input["random_seed"] = spec.random_seed

    expanded = expand_sweep(sweep_input)
    configs: list[BenchmarkConfig] = []
    variations = []
    for variant_dict, variation in expanded:
        # expand_sweep returns envelope-shaped variants — strip everything
        # but the body before instantiating BenchmarkConfig.
        body = variant_dict.get("benchmark", {})
        configs.append(BenchmarkConfig.model_validate(body))
        variations.append(variation)

    if spec.convergence is not None:
        trials = spec.convergence.max_runs
    elif spec.multi_run is not None and spec.multi_run.trials is not None:
        trials = spec.multi_run.trials
    else:
        trials = 1

    plan_kwargs: dict[str, Any] = {
        "configs": configs,
        "variations": variations,
        "trials": trials,
    }
    if spec.multi_run is not None:
        plan_kwargs["cooldown_seconds"] = spec.multi_run.cooldown_seconds
        plan_kwargs["set_consistent_seed"] = spec.multi_run.auto_set_seed
        plan_kwargs["disable_warmup_after_first"] = (
            spec.multi_run.disable_warmup_after_first
        )
        plan_kwargs["parameter_sweep_mode"] = spec.multi_run.mode
        # Propagate Bayesian-Optimization config so the controller pod can
        # instantiate a BayesianSearchPlanner and dispatch the adaptive
        # outer loop (mirrors the in-process build_benchmark_plan flow).
        if spec.multi_run.adaptive_search is not None:
            plan_kwargs["adaptive_search"] = spec.multi_run.adaptive_search
    if spec.convergence is not None:
        plan_kwargs["convergence_metric"] = spec.convergence.metric
        plan_kwargs["convergence_threshold"] = spec.convergence.threshold

    plan = BenchmarkPlan(**plan_kwargs)
    # Attach sweep-specific config so downstream readers (orchestrator
    # failure-threshold check, build_strategy adaptive flags) can use them.
    # BenchmarkPlan is a pydantic BaseModel without frozen/validate_assignment,
    # so plain attribute assignment is allowed and won't be validated again.
    plan.failure_policy = spec.failure_policy
    plan.convergence_config = spec.convergence
    return plan
