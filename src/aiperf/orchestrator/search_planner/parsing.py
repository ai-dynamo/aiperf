# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI grammar primitive for --search-space.

Pure parsing - no skopt import, so import cost is negligible. The objective
shape (metric / stat / direction) is three separate Pydantic-validated fields
and needs no parser.

Grammar:
    --search-space "PATH:LO,HI[:KIND]"      (repeatable; KIND in int/real)

Errors raise TypeError naming the offending flag, matching the pattern used
by ``parse_int_or_int_list`` in ``src/aiperf/config/parsing.py`` so cyclopts
surfaces the message cleanly.
"""

from __future__ import annotations

from aiperf.config.adaptive_search import SearchSpaceDimension

_VALID_KINDS = ("int", "real")


def parse_search_space(values: list[str]) -> list[SearchSpaceDimension]:
    """Parse one or more `--search-space "path:lo,hi[:kind]"` strings.

    Examples:
        >>> parse_search_space(["phases.profiling.concurrency:1,1000:int"])
        [SearchSpaceDimension(path='phases.profiling.concurrency', lo=1.0, hi=1000.0, kind='int')]
        >>> parse_search_space(["x:0,1"])  # default kind=real
        [SearchSpaceDimension(path='x', lo=0.0, hi=1.0, kind='real')]
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
