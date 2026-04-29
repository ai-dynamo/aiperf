# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark plan construction from AIPerf configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aiperf.config.benchmark import BenchmarkPlan
from aiperf.config.config import AIPerfConfig, BenchmarkConfig
from aiperf.config.loader.jinja import (
    build_template_context,
    render_jinja2_templates,
)


def build_benchmark_plan(config: AIPerfConfig) -> BenchmarkPlan:
    """Build a BenchmarkPlan from a validated AIPerfConfig.

    Expands sweep variations and extracts multi_run settings.
    If no sweep, returns a plan with a single config.

    Args:
        config: Validated AIPerfConfig (may contain sweep + multi_run).

    Returns:
        BenchmarkPlan with expanded configs and execution preferences.
    """
    from aiperf.config.sweep import SweepVariation, expand_sweep

    # Dump to dict, excluding sweep and multi_run (those are plan-level).
    # exclude_none/exclude_unset as safety net: annotated_types (Ge/Gt/Le/Lt)
    # handles None natively, but these flags protect against any future Field(gt=)
    # regressions that would break round-trip validation.
    config_dict = config.model_dump(mode="json", exclude_none=True, exclude_unset=True)
    sweep_dict = config_dict.pop("sweep", None)
    multi_run = config_dict.pop("multi_run", {})

    # Re-inject sweep for expand_sweep to process
    if sweep_dict is not None:
        config_dict["sweep"] = sweep_dict

    # Expand sweep variations
    expanded = expand_sweep(config_dict)

    configs = []
    variations = []
    for variation_dict, variation_meta in expanded:
        variation_dict.pop("sweep", None)
        variation_dict.pop("multi_run", None)

        # Re-render Jinja2 for this variation so sweep-overridden values
        # propagate to any templates that reference them. `variables` is
        # preserved on the resolved variation so run-time renderers (e.g.
        # artifacts.user_files) can resolve `{{ ... }}` expressions again.
        context = build_template_context(variation_dict)
        variation_dict = render_jinja2_templates(variation_dict, context)

        benchmark_config = BenchmarkConfig.model_validate(variation_dict)
        configs.append(benchmark_config)
        variations.append(variation_meta)

    # If no sweep produced variations, ensure we have a default variation
    if not variations:
        variations = [SweepVariation(index=0, label="base", values={})]

    plan_kwargs: dict[str, Any] = dict(
        configs=configs,
        variations=variations,
        trials=multi_run.get("num_runs", 1),
        cooldown_seconds=multi_run.get("cooldown_seconds", 0.0),
        confidence_level=multi_run.get("confidence_level", 0.95),
        set_consistent_seed=multi_run.get("set_consistent_seed", True),
        disable_warmup_after_first=multi_run.get("disable_warmup_after_first", True),
        parameter_sweep_cooldown_seconds=multi_run.get(
            "parameter_sweep_cooldown_seconds", 0.0
        ),
        parameter_sweep_same_seed=multi_run.get("parameter_sweep_same_seed", False),
    )
    for key in (
        "convergence_metric",
        "convergence_mode",
        "convergence_threshold",
        "convergence_stat",
    ):
        if key in multi_run and multi_run[key] is not None:
            plan_kwargs[key] = multi_run[key]
    plan = BenchmarkPlan(**plan_kwargs)
    _apply_sweep_seed_derivation(plan, config)
    return plan


def _apply_sweep_seed_derivation(plan: BenchmarkPlan, config: AIPerfConfig) -> None:
    """Derive a unique random_seed per sweep variation when not pinned.

    If ``parameter_sweep_same_seed`` is False (the default), each variation
    after the first gets ``base_seed + variation.index`` so independent
    samples don't degenerate to identical workloads. When the user pinned
    same-seed, leaves seeds untouched. Variation-0 always keeps the base
    seed so single-config runs are unaffected.
    """
    if plan.parameter_sweep_same_seed or not plan.is_sweep:
        return
    base_seed = config.random_seed
    if base_seed is None:
        return
    for variation_idx, cfg in enumerate(plan.configs):
        if variation_idx > 0:
            cfg.random_seed = base_seed + variation_idx


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
