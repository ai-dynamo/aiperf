# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data models for multi-run orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
from pydantic import Field

from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.export_models import JsonMetricResult


class RunResult(AIPerfBaseModel):
    """Result from executing a single benchmark run."""

    label: str = Field(description="Label identifying this run")
    success: bool = Field(description="Whether the run completed successfully")
    summary_metrics: dict[str, JsonMetricResult] = Field(
        default_factory=dict,
        description="Run-level summary statistics (e.g., {'time_to_first_token': JsonMetricResult(unit='ms', avg=150, p99=195)})",
    )
    error: str | None = Field(default=None, description="Error message if run failed")
    artifacts_path: Path | None = Field(
        default=None, description="Path to run artifacts directory"
    )
    variation_label: str = Field(
        default="",
        description="Sweep variation label (matches BenchmarkRun.variation.label).",
    )
    variation_values: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter values for this run's variation; mirror of variation.values.",
    )
    variation_index: int = Field(
        default=0,
        ge=0,
        description="Zero-based variation index; mirror of variation.index. Used to derive a unique fallback directory name.",
    )
    trial_index: int = Field(
        default=0, description="Zero-based trial index within the variation."
    )


VariationKey = tuple[str, tuple[tuple[str, Any], ...]]
"""Hashable key for grouping :class:`RunResult` by variation identity.

A 2-tuple of ``(variation_label, sorted_values_tuple)``. The label MUST
be part of the key because QMC samplers (Sobol/LHS) over coarse integer
dimensions routinely produce two distinct sample rows that collapse to
the same ``values`` dict - those are distinct sweep cells (they were
sampled independently and may differ in non-integer dims after rounding)
and must NOT be pooled. Per the user's
``feedback_never_aggregate_across_runs.md`` rule, only runs that share
ns + model + settings AND differ in exactly one swept dimension may be
aggregated; collisions on the values dict are not "the same cell".

The values tuple is retained alongside the label so SweepAnalyzer can
still surface the parameter combination for reporting.

Shared by ``aiperf.cli_runner._sweep_aggregate`` (cross-variation
aggregation), ``aiperf.cli_runner._multi_run`` (failure summary), and
``aiperf.orchestrator.orchestrator`` (per-cell callback key) so all three
construct the key the same hashable way.
"""


def _hashable_value(value: Any) -> Any:
    """Return ``value``, or its sorted-key JSON string when it is unhashable.

    Scenario sweeps put nested override dicts in ``variation_values``, and
    those can't go in a dict key directly. Sorted keys keep equal overrides
    grouping together.
    """
    try:
        hash(value)
    except TypeError:
        return orjson.dumps(value, option=orjson.OPT_SORT_KEYS).decode()
    return value


def _variation_key(label: str, values: dict[str, Any]) -> VariationKey:
    """Hashable, order-stable key for a variation cell.

    Pairs the variation label with the sorted values tuple so QMC cells
    with collision-prone integer dims (e.g. Sobol over ``lo=1, hi=4``)
    each get a distinct group even when ``values`` happens to match.

    Example:
        >>> _variation_key("sobol_0001", {"concurrency": 3})
        ('sobol_0001', (('concurrency', 3),))
        >>> _variation_key("sobol_0006", {"concurrency": 3})
        ('sobol_0006', (('concurrency', 3),))
    """
    return (label, tuple(sorted((k, _hashable_value(v)) for k, v in values.items())))
