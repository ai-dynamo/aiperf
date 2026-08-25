# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sweep child-name derivation and label-value sanitization.

Extracted from ``k8s_executor`` so the executor module stays under the
500-line ergonomics cap. The naming format is intentionally narrow
(``<sweep>-v<NN>-t<N>``) — see :func:`build_child_name` for the budget
arithmetic.
"""

from __future__ import annotations

import re

__all__ = [
    "MAX_SWEEP_NAME_LENGTH",
    "build_child_name",
    "child_name_suffix_length",
    "derive_child_name",
    "max_sweep_name_length",
    "needs_trial_suffix",
    "sanitize_for_label",
]

MAX_CHILD_JOB_ID_LENGTH = 35
MAX_SWEEP_VARIATIONS = 200
MAX_SWEEP_TRIALS = 10


def needs_trial_suffix(
    multi_run_trials: int | None,
    has_convergence: bool,
) -> bool:
    """Whether child names should include a `-tN` trial suffix."""
    if has_convergence:
        return True
    return (multi_run_trials or 1) > 1


def child_name_suffix_length(
    n_variations: int,
    max_total_runs: int,
    *,
    has_convergence: bool,
) -> int:
    """Return the worst-case ``-v<NN>[-t<N>]`` suffix length."""
    max_var_idx = max(n_variations - 1, 0)
    var_digits = max(2, len(str(max_var_idx)))
    trials = max_total_runs // n_variations if n_variations else 1
    return 2 + var_digits + (3 if needs_trial_suffix(trials, has_convergence) else 0)


def max_sweep_name_length(
    n_variations: int,
    max_total_runs: int,
    *,
    has_convergence: bool,
) -> int:
    """Return the sweep-name budget that keeps every child job ID valid."""
    return MAX_CHILD_JOB_ID_LENGTH - child_name_suffix_length(
        n_variations,
        max_total_runs,
        has_convergence=has_convergence,
    )


MAX_SWEEP_NAME_LENGTH = max_sweep_name_length(
    MAX_SWEEP_VARIATIONS,
    MAX_SWEEP_VARIATIONS * MAX_SWEEP_TRIALS,
    has_convergence=True,
)


def _validate_child_name_indexes(
    *,
    variation_index: int,
    trial_index: int | None,
) -> None:
    # Upper bound tracks the adaptive-search budget: BO/SLA planners stamp
    # ``SweepVariation.index`` with the monotonic iteration counter (0-based),
    # which runs up to ``AdaptiveSearchSweep.max_iterations`` (Field le=200, so
    # indices 0..199). A 0..99 cap crashed adaptive sweeps mid-run at iteration
    # 100 even though far fewer cells ran concurrently. ``f"{var_idx:02d}"``
    # renders 3-digit indices unambiguously (``v100``), and the child-name
    # length budget already reserves a multi-digit suffix (see
    # ``_name_from_config_file`` and ``build_child_name``).
    if not 0 <= variation_index <= 199:
        raise ValueError(
            f"variation index {variation_index} is outside the supported 0..199 range"
        )
    if trial_index is not None and not 0 <= trial_index <= 9:
        raise ValueError(
            f"trial index {trial_index} is outside the supported 0..9 range"
        )


def derive_child_name(
    sweep_name: str,
    var_idx: int,
    trial: int,
    *,
    with_trial_suffix: bool,
) -> str:
    """Deterministic DNS-safe child name from (sweep, var_idx, trial)."""
    trial_index = trial if with_trial_suffix else None
    _validate_child_name_indexes(variation_index=var_idx, trial_index=trial_index)
    base = f"{sweep_name}-v{var_idx:02d}"
    if with_trial_suffix:
        return f"{base}-t{trial:01d}"
    return base


def build_child_name(
    *,
    sweep_name: str,
    variation_index: int,
    trial_index: int | None,
    sweep_run_epoch: str | None = None,  # back-compat; ignored.
) -> str:
    """Deterministic child AIPerfJob name from (sweep, variation, trial).

    Format: ``<sweep>-v<vari:02d>-t<trial:01d>`` (or no -t suffix if
    ``trial_index is None``). Variation budget is 200 (00..199, matching
    ``AdaptiveSearchSweep.max_iterations`` le=200), trial budget is 10 (0..9).
    Bounded by the 35-char ``job_id`` cap (KubernetesDeployment validator), so
    sweep CR name must be <=29 chars.

    The sweep-run epoch is **not** in the name; it lives on the
    ``aiperf.nvidia.com/sweep-run-epoch`` label and on ``status.runEpoch``
    of the parent sweep CR. Across-rerun isolation is provided by
    ``K8sChildJobExecutor._wait_for_stale_child``. The ``sweep_run_epoch``
    keyword is accepted for source compatibility with callers from before
    the epoch-out-of-name refactor and is ignored.
    """
    _ = sweep_run_epoch  # back-compat shim: accepted but unused
    _validate_child_name_indexes(
        variation_index=variation_index,
        trial_index=trial_index,
    )
    suffix = f"-t{trial_index:01d}" if trial_index is not None else ""
    return f"{sweep_name}-v{variation_index:02d}{suffix}"


def sanitize_for_label(value: str) -> str:
    """Reduce a free-form string to a valid k8s label value.

    Label values must match ``(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?`` and
    be at most 63 characters. We:

    1. Lowercase + replace runs of disallowed chars with a single ``-``.
    2. Strip leading/trailing non-alnum.
    3. Truncate to 63.
    4. Re-strip leading/trailing non-alnum (the truncation may have left a
       trailing ``.``/``_``/``-``, which would re-fail validation).
    5. Fall back to ``"v"`` when sanitization eats every character.
    """
    sanitized = re.sub(r"[^a-z0-9._-]+", "-", value.lower())
    sanitized = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", sanitized)
    sanitized = sanitized[:63]
    sanitized = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", sanitized)
    return sanitized or "v"
