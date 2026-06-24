# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a BenchmarkPlan from an AIPerfSweep CR dict.

Reuses aiperf.config.sweep.expand_sweep to produce variations from the
benchmark config plus the sweep config from the CR's spec.
"""

from __future__ import annotations

from typing import Any

from aiperf.config.config import BenchmarkConfig
from aiperf.config.loader.plan import _apply_sweep_seed_derivation
from aiperf.config.resolution.plan import BenchmarkPlan
from aiperf.config.sweep import expand_sweep
from aiperf.operator.models import AIPerfSweepSpec

__all__ = ["build_plan_from_sweep"]


def build_plan_from_sweep(sweep_cr: dict[str, Any]) -> BenchmarkPlan:
    """Construct a BenchmarkPlan from an AIPerfSweep CR.

    Args:
        sweep_cr: Raw AIPerfSweep dict (typically from kubernetes_asyncio read).

    Returns:
        BenchmarkPlan with one config per variation; trial count comes from
        ``spec.multiRun.numRuns`` (default 1). When convergence is active it
        early-stops within that ``numRuns`` cap.

    Raises:
        ValidationError: If the CR spec fails Pydantic validation.
    """
    spec_dict = sweep_cr["spec"]
    spec = AIPerfSweepSpec.model_validate(spec_dict)

    benchmark_body = spec.benchmark.model_dump(
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

    multi_run = spec.multi_run
    trials = multi_run.num_runs if multi_run is not None else 1

    plan_kwargs: dict[str, Any] = {
        "configs": configs,
        "variations": variations,
        "trials": trials,
        "random_seed": spec.random_seed,
        "variables": dict(spec.variables) if spec.variables else {},
    }
    if multi_run is not None:
        # Mirror MultiRunConfig fields onto BenchmarkPlan top-level (kept for
        # consumers that read ``plan.cooldown_seconds`` directly), plus
        # propagate the full multi_run sub-object so convergence is visible
        # to the orchestrator via ``plan.multi_run.convergence``.
        plan_kwargs["cooldown_seconds"] = multi_run.cooldown_seconds
        plan_kwargs["confidence_level"] = multi_run.confidence_level
        plan_kwargs["set_consistent_seed"] = multi_run.set_consistent_seed
        plan_kwargs["disable_warmup_after_first"] = multi_run.disable_warmup_after_first
        plan_kwargs["multi_run"] = multi_run
    # Sweep block (grid/scenarios/adaptive_search) flows through unchanged —
    # the orchestrator dispatches off ``plan.sweep`` for parameter-sweep mode,
    # same_seed, and the BO planner.
    if spec.sweep is not None:
        plan_kwargs["sweep"] = spec.sweep
    if spec.failure_policy is not None:
        plan_kwargs["failure_policy"] = spec.failure_policy

    plan = BenchmarkPlan(**plan_kwargs)
    # Mirror the in-process loader: derive per-variation seeds from the
    # envelope random_seed so cluster sweeps populate plan.variation_seeds
    # the same way `aiperf profile` does.
    _apply_sweep_seed_derivation(plan, spec)
    return plan
