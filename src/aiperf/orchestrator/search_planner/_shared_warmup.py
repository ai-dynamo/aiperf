# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared warmup and config-mutation utilities for SLA-saturation planners.

Extracted from ``smooth_isotonic.py`` so that ``MultiTierPlanner`` can reuse
the same warmup/mutate logic without subclassing.
"""

from __future__ import annotations

from typing import Any

from aiperf.common.environment import Environment
from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import AdaptiveSearchSweep, _set_nested_value
from aiperf.config.sweep.adaptive import SearchSpaceDimension

_WARMUP_LOAD_KEYS: tuple[str, ...] = ("type", "rate", "rate_series", "concurrency")
"""Allowlist of load-defining keys copied to warmup; ramps, grace, and cancellation are dropped."""


def find_phase_index(phases: list[dict[str, Any]], name: str) -> int | None:
    """Return the first phase matching a literal name or semantic kind.

    Defensive against malformed fixtures where a phase entry is not a dict
    (e.g. test stubs); such entries are skipped rather than raising.
    """
    for idx, phase in enumerate(phases):
        if not isinstance(phase, dict):
            continue
        if phase.get("name") == name or phase.get("kind") == name:
            return idx
    return None


def apply_sla_warmup(
    cfg_dict: dict[str, Any],
    value: int | float,
    *,
    cfg: AdaptiveSearchSweep,
    first_probe_at: set[int | float],
) -> None:
    """Prepend a per-iteration ``warmup`` phase to ``cfg_dict["phases"]``.

    Skipped when ``cfg.sla_warmup_seconds == 0`` (explicit user opt-out)
    or when the profiling phase cannot be located. The warmup mirrors the
    profiling phase's load via ``_WARMUP_LOAD_KEYS`` and is excluded from results.

    Mutates ``first_probe_at`` to record which values have been warmed up.
    """
    if cfg.sla_warmup_seconds == 0:
        first_probe_at.add(value)
        return
    phases = cfg_dict.get("phases")
    if not phases:
        return
    idx = find_phase_index(phases, "profiling")
    if idx is None:
        return

    base_warmup = (
        cfg.sla_warmup_seconds
        if cfg.sla_warmup_seconds is not None
        else Environment.SEARCH_PLANNER.DEFAULT_WARMUP_SECONDS
    )
    if value not in first_probe_at:
        duration = max(Environment.SEARCH_PLANNER.FIRST_PROBE_WARMUP_FLOOR, base_warmup)
        first_probe_at.add(value)
    else:
        duration = max(Environment.SEARCH_PLANNER.REPLICATE_WARMUP_FLOOR, base_warmup)

    profiling_phase = phases[idx]
    warmup_phase: dict[str, Any] = {
        "name": "warmup",
        "kind": "warmup",
        "duration": duration,
        "exclude_from_results": True,
    }
    for key in _WARMUP_LOAD_KEYS:
        if key in profiling_phase:
            warmup_phase[key] = profiling_phase[key]
    if phases and isinstance(phases[0], dict) and phases[0].get("name") == "warmup":
        phases[0] = warmup_phase
    else:
        phases.insert(0, warmup_phase)


def apply_sla_precision(
    cfg_dict: dict[str, Any],
    cfg: AdaptiveSearchSweep,
) -> None:
    """Override profiling-phase ``requests`` per ``cfg.sla_precision``.

    Only fills in when the user did not specify ``requests`` on the
    profiling phase already; explicit user values always win.
    """
    target = Environment.SEARCH_PLANNER.SLA_PRECISION_REQUESTS.get(cfg.sla_precision)
    if target is None:
        return
    phases = cfg_dict.get("phases")
    if not phases:
        return
    idx = find_phase_index(phases, "profiling")
    if idx is None:
        return
    existing = phases[idx].get("requests")
    if existing is None:
        phases[idx]["requests"] = target


def mutate_base(
    base_config: BenchmarkConfig,
    dim: SearchSpaceDimension,
    value: int | float,
    *,
    cfg: AdaptiveSearchSweep,
    first_probe_at: set[int | float],
) -> BenchmarkConfig:
    """Return a deep-copied BenchmarkConfig with ``value`` patched in at the dim path.

    Applies SLA precision and warmup injection.
    """
    cfg_dict = base_config.model_dump(
        mode="python", exclude_none=True, context={"include_secrets": True}
    )
    _set_nested_value(cfg_dict, dim.path, value)
    apply_sla_precision(cfg_dict, cfg)
    apply_sla_warmup(cfg_dict, value, cfg=cfg, first_probe_at=first_probe_at)
    return BenchmarkConfig.model_validate(cfg_dict)
