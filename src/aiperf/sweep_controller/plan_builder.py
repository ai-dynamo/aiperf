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

    base_benchmark = spec.template.spec.get("benchmark") or {}
    if spec.sweep is not None:
        sweep_input = {
            **base_benchmark,
            "sweep": spec.sweep.model_dump(by_alias=True),
        }
    else:
        sweep_input = dict(base_benchmark)

    expanded = expand_sweep(sweep_input)
    configs: list[BenchmarkConfig] = []
    variations = []
    for variant_dict, variation in expanded:
        configs.append(BenchmarkConfig.model_validate(variant_dict))
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
    if spec.convergence is not None:
        plan_kwargs["convergence_metric"] = spec.convergence.metric
        plan_kwargs["convergence_threshold"] = spec.convergence.threshold

    plan = BenchmarkPlan(**plan_kwargs)
    # Best-effort: attach sweep-specific config so downstream readers
    # (orchestrator failure-threshold check, build_strategy adaptive flags)
    # can use them. Tolerated as no-op when BenchmarkPlan is extra="forbid".
    try:
        plan.failure_policy = spec.failure_policy
        plan.convergence_config = spec.convergence
    except (ValueError, AttributeError):
        pass
    return plan
