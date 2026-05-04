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

    Sweep + adaptive_search are mutually exclusive. When sweep is
    present, expands variations on the envelope dict and validates each
    variation's body as a BenchmarkConfig. When sweep is absent, the
    plan carries the single config.benchmark.
    """
    from aiperf.config.sweep import SweepVariation

    adaptive_search = config.multi_run.adaptive_search
    config_dict = config.model_dump(mode="json", exclude_none=True, exclude_unset=True)
    sweep_dict = config_dict.pop("sweep", None)

    if sweep_dict is not None and adaptive_search is not None:
        raise ValueError(
            "sweep block and --search-* flags are mutually exclusive: BO drives "
            "variation choice adaptively, while sweep enumerates them up-front. "
            "Drop the sweep block to use BO, or drop the --search-* flags."
        )

    if adaptive_search is not None or sweep_dict is None:
        configs = [config.benchmark.model_copy(deep=True)]
        variations = [SweepVariation(index=0, label="base", values={})]
    else:
        configs, variations = _expand_envelope_variations(config_dict, sweep_dict)

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
    """Assemble a BenchmarkPlan from envelope-level execution settings.

    Reads ``config.multi_run`` (and seed-derivation rules) at the envelope
    level. Shared by every dispatch path in ``build_benchmark_plan`` so
    the plan-kwargs surface stays one place.
    """
    adaptive_search = config.multi_run.adaptive_search
    post_process = config.multi_run.post_process
    sla_filters = list(config.multi_run.sla_filters)

    multi_run = config.multi_run.model_dump(
        mode="json", exclude_none=True, exclude_unset=True
    )
    multi_run.pop("adaptive_search", None)
    multi_run.pop("post_process", None)
    multi_run.pop("sla_filters", None)

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
    _apply_sweep_seed_derivation(plan, config)
    return plan


def _apply_sweep_seed_derivation(plan: BenchmarkPlan, config: AIPerfConfig) -> None:
    """Populate plan.variation_seeds from the envelope random_seed.

    Variation 0 carries the base seed; variation N gets ``base + N``
    unless ``parameter_sweep_same_seed`` is True (in which case all
    variations share the base seed). When ``random_seed`` is None on
    the envelope, all entries are None.
    """
    base_seed = config.random_seed
    plan.variation_seeds = []
    for variation_idx in range(len(plan.configs)):
        if base_seed is None:
            plan.variation_seeds.append(None)
        elif plan.parameter_sweep_same_seed or not plan.is_sweep:
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
