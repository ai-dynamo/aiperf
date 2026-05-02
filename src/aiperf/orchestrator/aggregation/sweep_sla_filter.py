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
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from aiperf.config.adaptive_search import SLAFilter
    from aiperf.orchestrator.aggregation.sweep import ParameterCombination


# Public alias documenting the input contract: callers may pass either typed
# `SLAFilter` instances (BO path) or already-dumped dicts (grid path round-tripped
# through `MultiRunConfig.sla_filters` before reaching this module). Both shapes
# carry the same four logical fields; ``_attr_or_key`` reads through either.
SLAFilterInput: TypeAlias = "SLAFilter | dict[str, Any]"


_OP_TO_FN: dict[str, Callable[[float, float], bool]] = {
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
}


def filter_feasible(
    per_combination_stats: dict[ParameterCombination, dict],
    sla_filters: list[SLAFilterInput],
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


def sla_filter_to_dict(f: SLAFilterInput) -> dict[str, Any]:
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
        f"SLA filter must be SLAFilter or dict (from MultiRunConfig.sla_filters); "
        f"got {type(f).__name__}: {f!r}. If this is a custom filter type, "
        "implement model_dump() or convert to a dict before passing in."
    )


def _passes_filter(stats: dict[str, Any], filter_obj: SLAFilterInput) -> bool:
    """Check one SLA filter against one combination's stats dict.

    Reads the metric value through two possible key shapes:

    * **Multi-trial layout** (`ConfidenceAggregation`) flattens to
      ``"<metric_tag>_<stat>"`` keyed at top level; the value is a stats block
      with a ``"mean"`` field.
    * **Single-trial layout** (`_json_metric_to_stats`) keys on ``metric_tag``
      alone; the value is a stats block whose direct attribute named after
      ``stat`` (e.g. ``"avg"``, ``"p99"``) carries the number.

    Falls back from flat to tag-only so single-trial sweeps don't silently mark
    every combo infeasible due to key-shape mismatch (matches the same fallback
    used by ``post_process._extract_points``).
    """
    metric_tag = _attr_or_key(filter_obj, "metric_tag")
    stat = _attr_or_key(filter_obj, "stat")
    op = _attr_or_key(filter_obj, "op")
    threshold = float(_attr_or_key(filter_obj, "threshold"))
    fn = _OP_TO_FN.get(op)
    if fn is None:
        raise ValueError(
            f"unknown SLA filter operator {op!r}; expected one of {sorted(_OP_TO_FN)}."
        )
    value = _read_metric_value(stats, metric_tag, stat)
    if value is None:
        return False
    return bool(fn(value, threshold))


def _read_metric_value(
    stats: dict[str, Any], metric_tag: str, stat: str
) -> float | None:
    """Pull a single metric value out of the stats dict, tolerating both layouts.

    Returns ``None`` when the value cannot be located; the caller treats that
    as infeasibility. The two layouts are documented in ``_passes_filter``.
    """
    flat_key = f"{metric_tag}_{stat}"
    block = stats.get(flat_key)
    if isinstance(block, dict) and "mean" in block:
        return float(block["mean"])
    block = stats.get(metric_tag)
    if isinstance(block, dict):
        raw = block.get(stat)
        if raw is not None:
            return float(raw)
    return None


def _attr_or_key(obj: Any, name: str) -> Any:
    """Read ``name`` off ``obj`` whether it's a Pydantic model or plain dict.

    SLA filters reach this module as either ``SLAFilter`` instances (BO path,
    typed) or dicts (grid path round-tripped through ``model_dump``); accepting
    both keeps the caller free of branching.
    """
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)
