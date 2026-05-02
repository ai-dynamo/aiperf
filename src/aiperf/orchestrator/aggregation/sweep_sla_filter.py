# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SLA-filter helpers for :class:`SweepAnalyzer`.

Lives in a sibling module to ``sweep.py`` purely to keep both files under the
500-line ergonomics cap. The single public entry point is
:func:`apply_sla_filters`, which produces the feasible subset; the analyzer
calls it once and routes ``best_configurations`` / ``pareto_optimal`` through
the resulting dict.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiperf.orchestrator.aggregation.sweep import ParameterCombination


_OP_TO_FN: dict[str, Callable[[float, float], bool]] = {
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
}


def filter_feasible(
    per_combination_stats: dict[ParameterCombination, dict],
    sla_filters: list[Any],
) -> dict[ParameterCombination, dict]:
    """Return the subset of combinations satisfying every SLA filter.

    A combination passes when, for every filter, ``stat(metric_tag) op threshold``
    holds on the combination's per-cell metrics dict. Combinations missing a
    referenced metric are treated as infeasible -- silent skip would mask a
    misconfigured filter, which is worse than emitting an empty feasible set.
    """
    feasible: dict[ParameterCombination, dict] = {}
    for combo, stats in per_combination_stats.items():
        if all(_passes_filter(stats, f) for f in sla_filters):
            feasible[combo] = stats
    return feasible


def sla_filter_to_dict(f: Any) -> dict[str, Any]:
    """Project an ``SLAFilter`` (or dict) to its serialized shape.

    Tolerates both Pydantic ``SLAFilter`` instances (have ``model_dump``) and
    plain dicts (already in the right shape) so the metadata block round-trips
    through the converter -> plan -> sweep-aggregate path without coercion.
    """
    if hasattr(f, "model_dump"):
        return f.model_dump(mode="json")
    if isinstance(f, dict):
        return dict(f)
    raise TypeError(
        f"SLA filter must be SLAFilter or dict, got {type(f).__name__}: {f!r}"
    )


def _passes_filter(stats: dict[str, Any], filter_obj: Any) -> bool:
    """Check one SLA filter against one combination's stats dict.

    The flat key follows :class:`SweepAnalyzer` convention
    (``<metric_tag>_<stat>``); ``stats[flat_key]["mean"]`` is the value
    compared to ``threshold`` via the named operator.
    """
    metric_tag = _attr_or_key(filter_obj, "metric_tag")
    stat = _attr_or_key(filter_obj, "stat")
    op = _attr_or_key(filter_obj, "op")
    threshold = float(_attr_or_key(filter_obj, "threshold"))
    flat_key = f"{metric_tag}_{stat}"
    block = stats.get(flat_key)
    if block is None or "mean" not in block:
        return False
    fn = _OP_TO_FN.get(op)
    if fn is None:
        raise ValueError(
            f"unknown SLA filter operator {op!r}; expected one of {sorted(_OP_TO_FN)}."
        )
    return bool(fn(float(block["mean"]), threshold))


def _attr_or_key(obj: Any, name: str) -> Any:
    """Read ``name`` off ``obj`` whether it's a Pydantic model or plain dict.

    SLA filters reach this module as either ``SLAFilter`` instances (BO path,
    typed) or dicts (grid path round-tripped through ``model_dump``); accepting
    both keeps the caller free of branching.
    """
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)
