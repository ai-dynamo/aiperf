# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI grammar primitives for --search-space and --search-sla.

Pure parsing - no Optuna / BoTorch import, so import cost is negligible. The
objective shape (metric / stat / direction) is three separate Pydantic-validated
fields and needs no parser.

Grammar:
    --search-space "PATH:LO,HI[:KIND]"      (repeatable; KIND in int/real)
    --search-sla   "TAG:STAT:OP:THRESHOLD"  (repeatable)

Errors raise ``TypeError`` naming the offending flag so cyclopts / click
surface the message cleanly.
"""

from __future__ import annotations

from typing import Any, cast

from aiperf.config.sweep.adaptive import SearchSpaceDimension, SLAFilter

_VALID_KINDS = ("int", "real")
_VALID_SLA_STATS: tuple[str, ...] = ("avg", "p50", "p90", "p95", "p99")
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
