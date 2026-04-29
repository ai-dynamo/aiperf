# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for v1 LoadGeneratorConfig -> AIPerfConfig sweep promotion.

Covers the integration point where ``--concurrency 10,20,30`` becomes a
sweep block on the resolved AIPerfConfig before PhaseConfig validation
sees the list.
"""

from __future__ import annotations

from aiperf.common.enums import SweepMode
from aiperf.config.loader import build_benchmark_plan
from aiperf.config.v1 import ServiceConfig, UserConfig
from aiperf.config.v1.converter import convert_user_to_aiperf

_BASE = {
    "endpoint": {"model_names": ["m"], "url": "http://localhost:8000"},
}


def _convert(loadgen: dict | None = None, **extra) -> object:
    payload = dict(_BASE, **extra)
    if loadgen is not None:
        payload["loadgen"] = loadgen
    user = UserConfig.model_validate(payload)
    return convert_user_to_aiperf(user, ServiceConfig())


def test_concurrency_list_lifts_to_sweep_variables() -> None:
    cfg = _convert(loadgen={"concurrency": [10, 20, 30]})
    assert cfg.sweep is not None
    assert cfg.sweep.variables == {"phases.profiling.concurrency": [10, 20, 30]}
    # PhaseConfig assigned its own default (concurrency=1) since the list was
    # stripped before validation; expand_sweep overwrites this per variation.
    profiling_phase = next(p for p in cfg.phases if p.name == "profiling")
    assert profiling_phase.concurrency in (None, 1)


def test_concurrency_csv_string_lifts_to_sweep_variables() -> None:
    cfg = _convert(loadgen={"concurrency": "10,20,30"})
    assert cfg.sweep is not None
    assert cfg.sweep.variables == {"phases.profiling.concurrency": [10, 20, 30]}


def test_concurrency_scalar_does_not_create_sweep() -> None:
    cfg = _convert(loadgen={"concurrency": 10})
    assert cfg.sweep is None
    profiling_phase = next(p for p in cfg.phases if p.name == "profiling")
    assert profiling_phase.concurrency == 10


def test_concurrency_list_produces_three_plan_configs() -> None:
    cfg = _convert(loadgen={"concurrency": [10, 20, 30]})
    plan = build_benchmark_plan(cfg)
    assert plan.is_sweep is True
    assert len(plan.configs) == 3
    concurrencies = [
        next(p for p in c.phases if p.name == "profiling").concurrency
        for c in plan.configs
    ]
    assert concurrencies == [10, 20, 30]


def test_seed_derivation_independent_per_variation() -> None:
    cfg = _convert(
        loadgen={"concurrency": [10, 20, 30]},
        input={"random_seed": 100},
    )
    plan = build_benchmark_plan(cfg)
    seeds = [c.random_seed for c in plan.configs]
    assert seeds == [100, 101, 102]


def test_seed_derivation_same_seed_pinned() -> None:
    cfg = _convert(
        loadgen={"concurrency": [10, 20, 30], "parameter_sweep_same_seed": True},
        input={"random_seed": 100},
    )
    plan = build_benchmark_plan(cfg)
    seeds = [c.random_seed for c in plan.configs]
    assert seeds == [100, 100, 100]


def test_repeated_sweep_mode_flows_through_to_multi_run() -> None:
    """``--parameter-sweep-mode=repeated`` lands as ``multi_run.mode='repeated'``."""
    cfg = _convert(
        loadgen={
            "concurrency": [10, 20],
            "num_profile_runs": 2,
            "parameter_sweep_mode": "repeated",
        },
    )
    assert cfg.multi_run.mode == SweepMode.REPEATED


def test_parameter_sweep_mode_default_is_repeated() -> None:
    """Omitted flag yields ``multi_run.mode == REPEATED`` (the v2 default)."""
    cfg = _convert(
        loadgen={"concurrency": [10, 20], "num_profile_runs": 2},
    )
    assert cfg.multi_run.mode == SweepMode.REPEATED
