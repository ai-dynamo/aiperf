# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark plan construction from AIPerf configuration."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiperf.config.loader.jinja import (
    build_template_context,
    render_jinja2_templates,
)
from aiperf.config.resolution.plan import BenchmarkPlan
from aiperf.config.sweep import AdaptiveSearchSweep, _GridSweepBase

if TYPE_CHECKING:
    from aiperf.config.config import AIPerfConfig, BenchmarkConfig


def build_benchmark_plan(config: AIPerfConfig) -> BenchmarkPlan:
    """Build a BenchmarkPlan from a validated AIPerfConfig.

    Sweep + adaptive_search are mutually exclusive at the envelope level
    (the discriminator on `sweep.type` collapses both into one block),
    so by the time we get here `config.sweep` is either absent, a grid /
    scenario sweep, or an adaptive_search sweep — never both.
    """
    from aiperf.config.sweep import SweepVariation

    is_adaptive = isinstance(config.sweep, AdaptiveSearchSweep)

    if is_adaptive or config.sweep is None:
        configs = [config.benchmark.model_copy(deep=True)]
        variations = [SweepVariation(index=0, label="base", values={})]
    else:
        # Prefer the pre-Jinja envelope dict captured at load time so per-
        # variation re-rendering can resolve `{{ var }}` body fields against
        # each variation's `variables:` overrides. Direct AIPerfConfig(...)
        # callers (no loader) leave _raw_envelope=None; fall back to
        # model_dump (templates are already gone, so swept-variable body
        # templating is a no-op there — same behavior as before).
        if config._raw_envelope is not None:
            envelope_dict = copy.deepcopy(config._raw_envelope)
            sweep_dict = envelope_dict.pop("sweep", None)
        else:
            envelope_dict = config.model_dump(
                mode="json", exclude_none=True, exclude_unset=True
            )
            sweep_dict = envelope_dict.pop("sweep", None)
        configs, variations = _expand_envelope_variations(envelope_dict, sweep_dict)

    return _assemble_plan_from_aiperf_config(config, configs, variations)


def _expand_envelope_variations(
    config_dict: dict[str, Any],
    sweep_dict: dict[str, Any],
) -> tuple[list[BenchmarkConfig], list[Any]]:
    """Expand the sweep block into per-variation BenchmarkConfigs.

    Operates on the envelope dict: each variation has its own benchmark
    subtree (post-merge for scenarios, post-grid-write for grids).
    Re-renders Jinja per variation against the merged context, then
    validates the rendered benchmark subtree as a BenchmarkConfig.
    """
    from aiperf.config.config import BenchmarkConfig
    from aiperf.config.sweep import SweepVariation, expand_sweep

    config_dict = dict(config_dict)
    config_dict["sweep"] = sweep_dict
    expanded = expand_sweep(config_dict)

    configs: list[BenchmarkConfig] = []
    variations: list[SweepVariation] = []
    for variation_dict, variation_meta in expanded:
        variation_dict.pop("sweep", None)
        variation_dict.pop("multi_run", None)
        context = build_template_context(variation_dict)
        variation_dict = render_jinja2_templates(variation_dict, context)
        bench_dict = variation_dict.get("benchmark", {})
        configs.append(BenchmarkConfig.model_validate(bench_dict))
        variations.append(variation_meta)
    if not variations:
        variations = [SweepVariation(index=0, label="base", values={})]
    return configs, variations


def _assemble_plan_from_aiperf_config(
    config: AIPerfConfig,
    configs: list[BenchmarkConfig],
    variations: list[Any],
) -> BenchmarkPlan:
    """Assemble a BenchmarkPlan, handing the envelope sub-objects through.

    Trial-mechanic scalars (trials, cooldown, confidence, seed flags) are
    still copied flat onto the plan because hot-path orchestrator code
    reads them directly. The full `multi_run` and `sweep` objects ride
    along for downstream readers that need the typed structure.
    """
    plan = BenchmarkPlan(
        configs=configs,
        variations=variations,
        trials=config.multi_run.num_runs,
        cooldown_seconds=config.multi_run.cooldown_seconds,
        confidence_level=config.multi_run.confidence_level,
        random_seed=config.random_seed,
        set_consistent_seed=config.multi_run.set_consistent_seed,
        disable_warmup_after_first=config.multi_run.disable_warmup_after_first,
        no_sweep_table=config.no_sweep_table,
        multi_run=config.multi_run,
        sweep=config.sweep,
        failure_policy=None,
        variables=dict(config.variables),
        plot=config.plot,
    )
    _apply_sweep_seed_derivation(plan, config)
    return plan


def _apply_sweep_seed_derivation(plan: BenchmarkPlan, config: AIPerfConfig) -> None:
    """Populate plan.variation_seeds from the envelope random_seed.

    Variation 0 carries the base seed; variation N gets ``base + N``.
    When the sweep's ``same_seed`` flag
    is True (grid / scenario / zip), every variation reuses the base seed.
    Adaptive sweeps add variations on the fly at runtime past the length of
    this list — the orchestrator falls back to SHA derivation (see
    ``resolve_run_seed``) for those overflow indices.
    """
    base_seed = config.random_seed
    same_seed = (
        isinstance(plan.sweep, _GridSweepBase) and plan.sweep.same_seed
    ) or not plan.is_sweep
    plan.variation_seeds = []
    for variation_idx in range(len(plan.configs)):
        if base_seed is None:
            plan.variation_seeds.append(None)
        elif same_seed:
            plan.variation_seeds.append(base_seed)
        else:
            plan.variation_seeds.append(base_seed + variation_idx)


def load_benchmark_plan(
    file_path: Path | str,
    *,
    substitute_env: bool = True,
) -> BenchmarkPlan:
    """Load a YAML config file and return a BenchmarkPlan.

    This is the new primary entry point for the orchestrator.
    Parses YAML -> AIPerfConfig -> expands sweep -> BenchmarkPlan.

    Args:
        file_path: Path to the YAML configuration file.
        substitute_env: Whether to process environment variable substitution.

    Returns:
        BenchmarkPlan with expanded configs and execution preferences.
    """
    # Import here to avoid circular import at module load time
    from aiperf.config.loader.core import load_config

    config = load_config(file_path, substitute_env=substitute_env)
    return build_benchmark_plan(config)
