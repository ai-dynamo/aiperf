# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expected Improvement with Constraints (EIC) for the Bayesian planner.

Implements the constrained-BO acquisition from
Gardner et al. 2014, "Bayesian Optimization with Inequality Constraints"
(https://proceedings.mlr.press/v32/gardner14.pdf):

    EIC(x) = EI(x) * P(feasible)(x)

where EI is the standard objective Expected Improvement (Mockus 1978) and
P(feasible)(x) = ∏_i Φ((0 - μ_i(x)) / σ_i(x)) is the per-constraint
predictive probability that the GP-modeled signed violation (negative =
feasible, positive = violation) lies at or below zero, multiplied across
independent constraints (Gardner 2014 §3.1).

Path-B integration rationale: skopt's ``Optimizer.__init__`` hard-rejects
callables passed to ``acq_func`` (only the eight named strings are
allowed; see ``optimizer.py``). The cleanest route is therefore to keep
the objective GP managed by skopt (so EI is consistent with the rest of
the codebase) and layer a sklearn-fitted constraint GP alongside it. At
``ask()`` time we sample K candidates from the search space, score
``EI(x) * P(feasible)(x)`` per candidate using the two GPs, and return
the argmax. ``tell()`` reports the objective to skopt and the per-filter
signed violation to our constraint GP.

Behavior is opt-in via ``AdaptiveSearchSweep.constraint_mode == "eic"``;
``"penalty"`` (default) keeps the existing soft-penalty path
bit-identical for users who don't change anything.

This module owns:

- :class:`ConstraintSurrogate` — per-filter sklearn GP wrapper plus the
  joint-feasibility CDF computation.
- :func:`compute_ei` — closed-form EI over a fitted skopt GP.
- :func:`select_eic_candidate` — sample, score, and argmax wrapper.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import norm

from aiperf.orchestrator.search_planner._bayesian_helpers import (
    signed_violation,
)
from aiperf.orchestrator.search_planner._sla_helpers import averaged_metric_value

if TYPE_CHECKING:
    from aiperf.config.sweep.adaptive import SLAFilter
    from aiperf.orchestrator.models import RunResult


__all__ = [
    "ConstraintSurrogate",
    "EIC_CANDIDATE_POOL_SIZE",
    "UNMEASURABLE_VIOLATION_SENTINEL",
    "compute_ei",
    "compute_feasibility_probability",
    "select_eic_candidate",
]


logger = logging.getLogger(__name__)

# Number of random candidates drawn from the search space at each ask()
# under EIC mode. Mirrors skopt's internal candidate-pool size for "sampling"
# acquisition optimization (Optimizer.n_points default = 10000). Picked large
# enough that argmax is a stable estimator of the true EIC optimum without
# triggering O(n^2) GP-prediction blowups for typical search spaces.
EIC_CANDIDATE_POOL_SIZE: int = 1000

# Sentinel signed-violation magnitude told to the constraint GP when a tell()
# can't measure the constraint metric. Must be large enough that the GP
# predicts μ >> 0 at this point (so P(feasible) collapses toward 0) without
# being so large that it dominates the GP's length-scale fit and starves
# nearby informative observations. A normalized-to-threshold magnitude of
# 10× is large vs. typical violation deltas (which sit near 1.0×) yet finite.
UNMEASURABLE_VIOLATION_SENTINEL: float = 10.0


def compute_ei(
    x_transformed: np.ndarray,
    objective_gp: Any,
    y_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Closed-form Expected Improvement of ``objective_gp`` minimizing.

    Per Mockus 1978 / skopt's ``gaussian_ei``:

    .. code-block:: text

        z(x) = (y_best - μ(x) - xi) / σ(x)
        EI(x) = (y_best - μ(x) - xi) * Φ(z) + σ(x) * φ(z)

    Returns a vector of EI scores, one per row of ``x_transformed``. ``xi``
    is the standard exploration-exploitation tradeoff parameter (0.01 is
    skopt's default). Zero-σ rows fall back to the deterministic case
    ``max(y_best - μ - xi, 0)``.

    Caller is responsible for transforming ``x_transformed`` into the GP's
    input space (skopt's ``space.transform``).
    """
    mu, sigma = objective_gp.predict(x_transformed, return_std=True)
    # Guard against zero / near-zero σ producing inf or NaN in the z-score.
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    improvement = y_best - mu - xi
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Zero out the EI for points where σ=0 (deterministic prediction):
    # treat them as having no improvement headroom unless their mean already
    # beats y_best, which is captured by the closed form above with σ→0.
    ei = np.where(sigma > 1e-12, ei, np.maximum(improvement, 0.0))
    return np.asarray(ei, dtype=float)


def compute_feasibility_probability(
    x_transformed: np.ndarray,
    constraint_gps: list[Any],
) -> np.ndarray:
    """Joint P(feasible)(x) across one or more constraint surrogates.

    Each GP in ``constraint_gps`` models the signed-violation magnitude
    of one SLA filter (negative = feasible, positive = violation), so per
    Gardner 2014:

    .. code-block:: text

        P(constraint_i_satisfied)(x) = Φ((0 - μ_i(x)) / σ_i(x))
        P(feasible)(x) = ∏_i P(constraint_i_satisfied)(x)

    Empty ``constraint_gps`` returns an all-ones vector — degenerates to
    standard EI when there are no constraints (Gardner 2014 reduces to
    Mockus 1978 in this limit).
    """
    n = x_transformed.shape[0]
    if not constraint_gps:
        return np.ones(n, dtype=float)
    joint = np.ones(n, dtype=float)
    for gp in constraint_gps:
        mu, sigma = gp.predict(x_transformed, return_std=True)
        sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
        z = (0.0 - mu) / sigma_safe
        prob = norm.cdf(z)
        # Deterministic σ=0 fallback: feasible iff μ ≤ 0.
        prob = np.where(sigma > 1e-12, prob, np.where(mu <= 0.0, 1.0, 0.0))
        joint = joint * prob
    return joint


class ConstraintSurrogate:
    """Per-filter constraint GP plus the joint-feasibility probability.

    One sklearn ``GaussianProcessRegressor`` per :class:`SLAFilter`,
    fitted on the signed-violation observations recorded at each
    :meth:`tell`. ``observe()`` accepts a list of :class:`RunResult` and a
    skopt-transformed coordinate; it computes one signed-violation scalar
    per filter (averaged over successful trials, or a positive sentinel
    when unmeasurable) and appends to that filter's training set.

    The fit is deferred until ``feasibility_probability()`` is called, so
    a planner that never asks under EIC mode pays no fit cost. We refit
    on every call once the n-initial-points bootstrap is past — sklearn
    GPs are O(n^3) but n is bounded by ``max_iterations`` (≤ 200 per the
    config schema), so this is cheap in absolute terms.
    """

    def __init__(self, sla_filters: list[SLAFilter]) -> None:
        self._filters = list(sla_filters)
        # X_train shared across all filters (every observation has the
        # same x); each filter has its own y_train (its own violation
        # series) since per-filter measurability differs.
        self._x_train: list[list[float]] = []
        self._y_per_filter: list[list[float]] = [[] for _ in self._filters]
        self._fitted_gps: list[Any] | None = None

    def observe(
        self,
        x_transformed: list[float],
        results: list[RunResult],
        unmeasurable_sentinel: float = UNMEASURABLE_VIOLATION_SENTINEL,
    ) -> bool:
        """Record per-filter signed violations at ``x_transformed``.

        Returns True iff at least one filter could not be measured
        (unmeasurable in :func:`averaged_metric_value`'s sense), so the
        caller can mirror the penalty path's ``has_unmeasurable`` flag
        for the iteration's ``feasible`` history field. Each unmeasurable
        filter is recorded with a normalized-to-threshold positive
        sentinel so the GP later predicts μ >> 0 there.

        Invalidates the cached fit; ``feasibility_probability()`` will
        re-fit on next call.
        """
        self._x_train.append(list(x_transformed))
        has_unmeasurable = False
        for idx, sla in enumerate(self._filters):
            value = averaged_metric_value(results, sla.metric_tag, sla.stat)
            if value is None:
                has_unmeasurable = True
                self._y_per_filter[idx].append(unmeasurable_sentinel)
                continue
            denom = abs(sla.threshold) if sla.threshold != 0 else 1.0
            normalized = signed_violation(value, sla) / denom
            self._y_per_filter[idx].append(float(normalized))
        self._fitted_gps = None
        return has_unmeasurable

    def feasibility_probability(self, x_transformed: np.ndarray) -> np.ndarray:
        """Joint P(feasible) at each row of ``x_transformed``.

        Empty filter list returns an all-ones vector (degenerates to pure
        EI). Empty training set also returns all-ones — with no
        observations the constraint GP has nothing to say, so we fall
        back to the unconstrained acquisition (BO is essentially random
        during the initial-points phase anyway).
        """
        if not self._filters or not self._x_train:
            return np.ones(x_transformed.shape[0], dtype=float)
        if self._fitted_gps is None:
            self._fitted_gps = self._fit()
        return compute_feasibility_probability(x_transformed, self._fitted_gps)

    def _fit(self) -> list[Any]:
        # Local import keeps sklearn out of the import path for users who
        # never enable EIC. sklearn is already a transitive skopt dep.
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

        x_arr = np.asarray(self._x_train, dtype=float)
        gps: list[Any] = []
        for y_series in self._y_per_filter:
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(
                noise_level=1e-4
            )
            gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=2,
                random_state=0,
            )
            gp.fit(x_arr, np.asarray(y_series, dtype=float))
            gps.append(gp)
        return gps


def select_eic_candidate(
    *,
    space: Any,
    objective_gp: Any,
    y_best: float,
    constraint_surrogate: ConstraintSurrogate,
    rng: Any,
    pool_size: int = EIC_CANDIDATE_POOL_SIZE,
) -> list[Any]:
    """Sample ``pool_size`` candidates from ``space`` and return the EIC argmax.

    Uses skopt's ``space.rvs(n_samples, random_state)`` for sampling — same
    primitive skopt uses internally — so the result respects the configured
    Integer / Real dimensions and per-dim transforms. The returned point is
    in the *original* (untransformed) coordinate system, ready to feed back
    into ``Optimizer.tell()``.

    EIC = EI(x) * P(feasible)(x) per Gardner 2014. With no constraints,
    P(feasible)=1 and the result reduces to standard EI argmax.
    """
    candidates = space.rvs(n_samples=pool_size, random_state=rng)
    x_transformed = np.asarray(space.transform(candidates), dtype=float)
    ei = compute_ei(x_transformed, objective_gp, y_best)
    p_feasible = constraint_surrogate.feasibility_probability(x_transformed)
    eic = ei * p_feasible
    best_idx = int(np.argmax(eic))
    return list(candidates[best_idx])
