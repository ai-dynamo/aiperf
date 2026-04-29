# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for v1 LoadGeneratorConfig -> AIPerfConfig sweep promotion.

Covers the integration point where ``--concurrency 10,20,30`` becomes a
sweep block on the resolved AIPerfConfig before PhaseConfig validation
sees the list.
"""

from __future__ import annotations

import pytest

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


def test_repeated_sweep_mode_rejected_if_present() -> None:
    """If a repeated sweep mode were ever set on loadgen, the converter rejects it.

    The field is not currently exposed on v1, but the converter guards
    defensively so a future field rename does not silently activate the
    unimplemented REPEATED execution path.
    """
    user = UserConfig.model_validate({**_BASE, "loadgen": {"concurrency": 10}})
    # Inject a repeated sentinel post-construction since v1 doesn't expose it
    object.__setattr__(user.loadgen, "parameter_sweep_mode", "repeated")
    user.loadgen.model_fields_set.add("parameter_sweep_mode")
    with pytest.raises(ValueError, match="repeated"):
        convert_user_to_aiperf(user, ServiceConfig())
