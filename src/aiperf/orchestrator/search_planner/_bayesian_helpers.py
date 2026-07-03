# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module-scope helpers for :class:`BayesianSearchPlanner`.

Lives in a sibling module to keep ``bayesian.py`` under the 500-line file-size
cap. Each function reads no planner state, so module-scope is the right home.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiperf.config.sweep.adaptive import SLAFilter


# Plateau-detection guard: coefficient of variation has no meaning when |mean|
# collapses to zero; this floor refuses to claim convergence in that regime.
PLATEAU_MEAN_EPSILON = 1e-9

# Sentinel loss told to skopt when an iteration has no usable objective AND no
# successful prior to scale against. Large enough to be unambiguously worse
# than any plausible real metric, finite enough not to poison the GP kernel.
# Not user-tunable: the no-data branch is degraded mode and BO is essentially
# random until the first success.
NO_DATA_SENTINEL_LOSS = 1.0e6

# Soft-penalty multiplier on max(|loss|) for SLA-filter violations. Finite
# (not ±inf / 1e18) to avoid GP variance distortion; tune up if BO ignores
# soft constraints, down if it dominates exploration too early.
PENALTY_WEIGHT_MULTIPLIER: float = 100.0


def signed_violation(value: float, sla: SLAFilter) -> float:
    """Signed how-much-``value``-violates-``sla``: positive = violation, negative = slack.

    Caller clamps to ``max(0, .)`` so only violations contribute. ``lt``/``le``
    use ``value - threshold``; ``gt``/``ge`` use ``threshold - value``.
    """
    if sla.op in ("lt", "le"):
        return value - sla.threshold
    return sla.threshold - value
