# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI grammar primitives for --search-space, --search-sla, and --search-sla-tier.

Pure parsing - no Optuna / BoTorch import, so import cost is negligible. The
objective shape (metric / stat / direction) is three separate Pydantic-validated
fields and needs no parser.

Grammar:
    --search-space    "PATH:LO,HI[:KIND]"                  (repeatable; KIND in int/real)
    --search-sla      "TAG:STAT:OP:THRESHOLD"              (repeatable)
    --search-sla-tier "LABEL:FILTER[,FILTER...]"           (repeatable; 2-10 tiers)

Errors raise ``TypeError`` naming the offending flag so cyclopts / click
surface the message cleanly.
"""

from __future__ import annotations

from typing import Any, cast

from aiperf.config.sweep.adaptive import SearchSpaceDimension, SLAFilter
from aiperf.orchestrator.search_planner.multi_tier_models import SLOTier

_VALID_KINDS = ("int", "real")
_VALID_SLA_STATS: tuple[str, ...] = (
    "avg",
    "p1",
    "p5",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
    "p99",
    "min",
    "max",
)
_VALID_SLA_OPS: tuple[str, ...] = ("lt", "le", "gt", "ge")


def parse_search_space(values: list[str]) -> list[SearchSpaceDimension]:
    """Parse one or more `--search-space "path:lo,hi[:kind]"` strings.

    Examples::

        parse_search_space(["phases.profiling.concurrency:1,1000:int"])
        # -> [SearchSpaceDimension(path='phases.profiling.concurrency',
        #                          lo=1.0, hi=1000.0, kind='int', prior='uniform')]
        parse_search_space(["x:0,1"])  # default kind=real
        # -> [SearchSpaceDimension(path='x', lo=0.0, hi=1.0,
        #                          kind='real', prior='uniform')]
    """
    out: list[SearchSpaceDimension] = []
    for raw in values:
        out.append(_parse_one_dim(raw))
    return out


def _parse_one_dim(raw: str) -> SearchSpaceDimension:
    if ":" not in raw or "," not in raw:
        raise TypeError(
            f"--search-space {raw!r}: expected 'path:lo,hi[:kind]', e.g. "
            f"'phases.profiling.concurrency:1,1000:int'."
        )
    parts = raw.split(":")
    if len(parts) == 2:
        path, bounds = parts
        kind = "real"
    elif len(parts) == 3:
        path, bounds, kind = parts
    else:
        raise TypeError(
            f"--search-space {raw!r}: expected 'path:lo,hi[:kind]', got {len(parts)} parts."
        )
    if kind not in _VALID_KINDS:
        raise TypeError(
            f"--search-space {raw!r}: kind must be 'int' or 'real', got {kind!r}."
        )
    if "," not in bounds:
        raise TypeError(
            f"--search-space {raw!r}: expected 'path:lo,hi[:kind]', missing ',' in bounds."
        )
    lo_s, hi_s = bounds.split(",", 1)
    try:
        lo, hi = float(lo_s), float(hi_s)
    except ValueError as e:
        raise TypeError(
            f"--search-space {raw!r}: could not parse bound as float ({e})."
        ) from e
    if hi <= lo:
        raise TypeError(f"--search-space {raw!r}: hi ({hi}) must be > lo ({lo}).")
    return SearchSpaceDimension(path=path, lo=lo, hi=hi, kind=kind)


def parse_sla_filter(value: str) -> SLAFilter:
    """Parse a single ``--search-sla "metric_tag:stat:op:threshold"`` flag value.

    Strict colon-delimited 4-tuple — exactly 3 colons. ``stat`` must be one of
    ``avg|p50|p90|p95|p99``; ``op`` one of ``lt|le|gt|ge``; ``threshold`` a
    float. Whitespace is NOT stripped — users get a clean parse error if they
    add leading/trailing space, which is preferable to silently coercing a
    surface-level user mistake.

    Examples::

        parse_sla_filter("time_to_first_token:p95:lt:200")
        # -> SLAFilter(metric_tag='time_to_first_token', stat='p95',
        #              op='lt', threshold=200.0)
        parse_sla_filter("request_error_rate:p99:lt:0.05")
        # -> SLAFilter(metric_tag='request_error_rate', stat='p99',
        #              op='lt', threshold=0.05)

    Raises:
        TypeError: when the value has the wrong colon count, an unknown stat
            or op, or a non-float threshold. The error message names the
            offending flag value so cyclopts can surface it directly.
    """
    parts = value.split(":")
    if len(parts) != 4:
        raise TypeError(
            f"--search-sla {value!r}: expected 4 colon-separated parts "
            f"'metric_tag:stat:op:threshold', got {len(parts)}. Example: "
            f"'time_to_first_token:p95:lt:200'."
        )
    metric_tag, stat, op, threshold_str = parts
    if stat not in _VALID_SLA_STATS:
        raise TypeError(
            f"--search-sla {value!r}: unknown stat {stat!r}; "
            f"expected one of {_VALID_SLA_STATS}."
        )
    if op not in _VALID_SLA_OPS:
        raise TypeError(
            f"--search-sla {value!r}: unknown op {op!r}; "
            f"expected one of {_VALID_SLA_OPS}."
        )
    try:
        threshold = float(threshold_str)
    except ValueError as e:
        raise TypeError(
            f"--search-sla {value!r}: threshold {threshold_str!r} is not a float."
        ) from e
    return SLAFilter(
        metric_tag=metric_tag,
        stat=cast(Any, stat),
        op=cast(Any, op),
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# --search-sla-tier parsing
# ---------------------------------------------------------------------------

_TIER_COUNT_MIN = 2
_TIER_COUNT_MAX = 10


def _has_label(value: str) -> bool:
    """Determine if the tier string starts with an explicit label.

    A valid filter has exactly 3 colons (metric_tag:stat:op:threshold).
    If the text before the first comma has != 3 colons, a label is present:
    - More than 3 colons means a label: prefix adds an extra colon separator
    - Fewer than 3 colons means it's a bare label without any filter content
    """
    first_segment = value.split(",", 1)[0]
    return first_segment.count(":") != 3


def parse_sla_tier(value: str, *, _auto_index: int = 0) -> SLOTier:
    """Parse a single ``--search-sla-tier`` flag value.

    Grammar: ``"LABEL:FILTER[,FILTER...]"`` or ``"FILTER[,FILTER...]"``

    When the first colon-delimited segment before the first comma has fewer
    than 3 colons it is treated as a label. Otherwise an auto-generated
    label (``tier_1``, ``tier_2``, ...) is assigned using ``_auto_index``.

    Each FILTER uses the existing ``metric_tag:stat:op:threshold`` grammar
    parsed via :func:`parse_sla_filter`.

    Raises:
        TypeError: when the value cannot be parsed or produces zero filters.
    """
    if not value.strip():
        raise TypeError("--search-sla-tier: value must not be empty.")

    if _has_label(value):
        # Split label from the rest at the first colon
        label, _, remainder = value.partition(":")
        if not label:
            raise TypeError(
                f"--search-sla-tier {value!r}: label before ':' must not be empty."
            )
    else:
        label = f"tier_{_auto_index + 1}"
        remainder = value

    # Split remainder by comma to get individual filter strings
    filter_strs = [f.strip() for f in remainder.split(",") if f.strip()]
    if not filter_strs:
        raise TypeError(
            f"--search-sla-tier {value!r}: tier '{label}' contains zero filters. "
            f"Each tier must have at least one filter in 'metric_tag:stat:op:threshold' format."
        )

    filters: list[SLAFilter] = []
    for fs in filter_strs:
        try:
            filters.append(parse_sla_filter(fs))
        except TypeError as e:
            raise TypeError(
                f"--search-sla-tier {value!r}: error parsing filter {fs!r} in tier '{label}': {e}"
            ) from e

    return SLOTier(label=label, filters=filters)


def validate_tier_list(tiers: list[SLOTier]) -> list[SLOTier]:
    """Validate a list of parsed SLO tiers.

    Checks:
    - Tier count is in [{_TIER_COUNT_MIN}, {_TIER_COUNT_MAX}].
    - No duplicate labels.

    Raises:
        ValueError: when tier count is out of range or labels are duplicated.
    """
    count = len(tiers)
    if count < _TIER_COUNT_MIN or count > _TIER_COUNT_MAX:
        raise ValueError(
            f"--search-sla-tier: expected between {_TIER_COUNT_MIN} and "
            f"{_TIER_COUNT_MAX} tiers, got {count}."
        )

    seen: dict[str, int] = {}
    for i, tier in enumerate(tiers):
        if tier.label in seen:
            raise ValueError(
                f"--search-sla-tier: duplicate label {tier.label!r} "
                f"(tiers {seen[tier.label] + 1} and {i + 1}). "
                f"Each tier must have a unique label."
            )
        seen[tier.label] = i

    return tiers
