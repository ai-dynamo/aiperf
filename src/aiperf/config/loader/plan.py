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

    Expands sweep variations and extracts multi_run settings, OR — when
    config.multi_run.adaptive_search is set — produces a single-config plan
    with plan.adaptive_search populated. Sweep + adaptive_search are mutually exclusive.
    """
    from aiperf.config.sweep import SweepVariation

    adaptive_search = (
        config.multi_run.adaptive_search
    )  # already typed AdaptiveSearchConfig | None
    post_process = config.multi_run.post_process  # PostProcessSpec | None
    sla_filters = list(config.multi_run.sla_filters)

    config_dict = config.model_dump(mode="json", exclude_none=True, exclude_unset=True)
    sweep_dict = config_dict.pop("sweep", None)
    multi_run = config_dict.pop("multi_run", {})
    multi_run.pop(
        "adaptive_search", None
    )  # propagated separately as `adaptive_search` kwarg
    multi_run.pop("post_process", None)  # propagated separately
    multi_run.pop("sla_filters", None)  # propagated separately

    if sweep_dict is not None and adaptive_search is not None:
        raise ValueError(
            "sweep block and --search-* flags are mutually exclusive: BO drives "
            "variation choice adaptively, while sweep enumerates them up-front. "
            "Drop the sweep block to use BO, or drop the --search-* flags."
        )

    if adaptive_search is not None:
        # BO path: single base config, single placeholder variation. The
        # planner synthesizes per-iteration variations during execution.
        configs = [BenchmarkConfig.model_validate(config_dict)]
        variations = [SweepVariation(index=0, label="base", values={})]
    else:
        configs, variations = _expand_grid_variations(config_dict, sweep_dict)

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
        parameter_sweep_mode=multi_run.get("mode", "repeated"),
        adaptive_search=adaptive_search,
        post_process=post_process,
        sla_filters=sla_filters,
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
    if adaptive_search is None:
        _apply_sweep_seed_derivation(plan, config)
    return plan


def _expand_grid_variations(
    config_dict: dict[str, Any],
    sweep_dict: dict[str, Any] | None,
) -> tuple[list[BenchmarkConfig], list[Any]]:
    """Expand the (optional) sweep block into per-variation BenchmarkConfigs.

    Returns the parallel ``(configs, variations)`` lists. Re-renders Jinja2
    templates per variation so sweep-overridden values propagate. Falls back
    to a single ``base`` variation when no sweep is present.
    """
    from aiperf.config.sweep import SweepVariation, expand_sweep

    if sweep_dict is not None:
        config_dict["sweep"] = sweep_dict
    expanded = expand_sweep(config_dict)
    configs: list[BenchmarkConfig] = []
    variations: list[Any] = []
    for variation_dict, variation_meta in expanded:
        variation_dict.pop("sweep", None)
        variation_dict.pop("multi_run", None)
        context = build_template_context(variation_dict)
        variation_dict = render_jinja2_templates(variation_dict, context)
        configs.append(BenchmarkConfig.model_validate(variation_dict))
        variations.append(variation_meta)
    if not variations:
        variations = [SweepVariation(index=0, label="base", values={})]
    return configs, variations


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
