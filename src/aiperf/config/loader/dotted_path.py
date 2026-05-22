# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared dotted-path validator for sweep dimensions.

Used by all sweep surfaces (grid, zip, scenarios, QMC, adaptive) so they
reject the same malformed/non-sweepable paths before they reach
`_set_nested_value` and corrupt the variant dict.
"""

from __future__ import annotations

# First-segment values that are envelope-level rather than body-rooted.
# They are not sweepable as benchmark fields and get rejected at the
# dimension-validator level so the user gets a clear error rather than
# a silent phantom-key write into the envelope.
_NON_SWEEPABLE_FIRST_SEGMENTS = frozenset({"sweep", "multi_run", "random_seed"})

# Bare-name sugar for the most-swept phase fields. A path equal to one of
# these keys (no dots) is rewritten to the canonical phases.profiling.X
# form before validation. Compound paths like ``concurrency.value`` are
# left untouched -- only the standalone token is sugar. Resolution then
# chains with the recipe-fallback in expand.py::_find_phase_or_recipe_alias,
# so ``phases: {type: concurrency}`` YAML (which emits a phase named
# ``default``) still receives the swept value.
_SWEEP_PATH_ALIASES = {
    "concurrency": "phases.profiling.concurrency",
    "prefill_concurrency": "phases.profiling.prefill_concurrency",
    "rate": "phases.profiling.rate",
    "requests": "phases.profiling.requests",
    "duration": "phases.profiling.duration",
    "sessions": "phases.profiling.sessions",
    "users": "phases.profiling.users",
    "smoothness": "phases.profiling.smoothness",
    "grace_period": "phases.profiling.grace_period",
    "concurrency_ramp": "phases.profiling.concurrency_ramp",
    "prefill_ramp": "phases.profiling.prefill_ramp",
    "rate_ramp": "phases.profiling.rate_ramp",
}


def _resolve_path_alias(p: str) -> str:
    """Rewrite a bare-name sweep path through ``_SWEEP_PATH_ALIASES``.

    Returns the input unchanged if it contains a dot or isn't a known
    alias. Pure string transform; no validation. Called by
    ``_validate_dotted_path`` so all four sweep surfaces (grid, zip, QMC,
    adaptive) share the same sugar table.
    """
    if not isinstance(p, str) or "." in p:
        return p
    return _SWEEP_PATH_ALIASES.get(p, p)


def _validate_dotted_path(p: str) -> str:
    """Validate (and alias-resolve) a dotted-path string for sweep dimensions.

    Applies ``_resolve_path_alias`` first so bare names like ``concurrency``
    are rewritten to their canonical ``phases.profiling.X`` form. The
    returned path is what callers should use as the dict key / dimension
    path; the original sugar form is not preserved downstream.

    Rejects empty strings, leading/trailing dots, and consecutive dots
    (which would create phantom empty-string keys when written via
    ``_set_nested_value``). Forbids first segments that target the
    envelope itself (``sweep``, ``multi_run``, ``random_seed``) -- note
    that ``sweep`` is special-cased first with a dedicated "sweep config
    itself" message; the remaining members of
    ``_NON_SWEEPABLE_FIRST_SEGMENTS`` fall through to a generic
    non-sweepable-top-level-field error. Also rejects the redundant
    ``benchmark.`` prefix to match the grid-sweep convention -- dimension
    paths are body-rooted under the benchmark block.
    """
    p = _resolve_path_alias(p)
    if not p:
        raise ValueError("dimension path must be a non-empty string.")
    if p.startswith("."):
        raise ValueError(f"dimension path {p!r} must not start with '.'.")
    if p.endswith("."):
        raise ValueError(f"dimension path {p!r} must not end with '.'.")
    if ".." in p:
        raise ValueError(
            f"dimension path {p!r} must not contain consecutive dots ('..')."
        )
    first = p.split(".", 1)[0]
    if first == "sweep":
        raise ValueError(
            f"dimension path {p!r} targets the sweep config itself; "
            f"'sweep.*' paths are not sweepable."
        )
    if first in _NON_SWEEPABLE_FIRST_SEGMENTS:
        raise ValueError(
            f"dimension path {p!r} targets non-sweepable top-level field {first!r}."
        )
    if first == "benchmark":
        raise ValueError(
            f"dimension path {p!r} must not include the redundant "
            f"'benchmark.' prefix. Paths target fields under the "
            f"benchmark block; drop the prefix (e.g. "
            f"'phases.profiling.rate' instead of "
            f"'benchmark.phases.profiling.rate')."
        )
    return p
