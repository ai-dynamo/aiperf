# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module-scope helpers for :class:`OptunaSearchPlanner`.

Lives in a sibling module to keep ``optuna_planner.py`` focused on the
planner class itself. Each function reads no planner state, so
module-scope is the right home.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from aiperf.config.sweep import (
        AdaptiveSearchSweep,
        Objective,
        OutcomeConstraint,
    )
    from aiperf.config.sweep.adaptive import SLAFilter


__all__ = [
    "build_constraints_func",
    "build_outcome_constraints_func",
    "build_qlognei_candidates_func",
    "build_qnehvi_candidates_func",
    "build_sampler",
    "compute_hypervolume",
    "derive_reference_point",
]


# Auto-derived reference points sit slightly worse than the worst-observed
# value per objective so that initial Sobol points contribute non-zero
# hypervolume; per BoTorch's qNEHVI tutorial recommendation.
_REFERENCE_POINT_SLACK: float = 0.05


# Magnitude rationale: must be (a) large enough to dominate any plausible
# objective in the constraint surrogate, (b) small enough to keep the GP
# kernel well-posed. Scaled for constraints rather than penalty losses.
# 1e6 is conservative; revisit if benchmark evidence shows it dominates
# legitimate signal.
_UNMEASURABLE_VIOLATION: float = 1.0e6


def _attr_key(sla: SLAFilter) -> str:
    """Build a unique trial.user_attr key per SLA filter.

    Multiple filters can reference the same metric_tag with different
    stat/op (e.g. p95 lt 200 AND p99 lt 500 on time_to_first_token), so
    the key must encode all four fields.
    """
    return f"sla:{sla.metric_tag}:{sla.stat}:{sla.op}:{sla.threshold}"


def _signed_violation(observed: float | None, sla: SLAFilter) -> float:
    """Optuna's contract: positive violates, <=0 is feasible.

    For ``op=lt|le, threshold=T, observed=O``: violation = ``O - T``. For
    ``op=gt|ge``: violation = ``T - O``. Missing observation collapses to
    a fixed-magnitude penalty (``_UNMEASURABLE_VIOLATION``) — matches the
    BO planner's "treat unmeasurable as infeasible" policy.
    """
    if observed is None:
        return _UNMEASURABLE_VIOLATION
    if sla.op in ("lt", "le"):
        return observed - sla.threshold
    return sla.threshold - observed


def build_constraints_func(
    sla_filters: list[SLAFilter],
) -> Callable[[Any], Sequence[float]]:
    """Build a ``constraints_func`` that reads observations from trial.user_attrs.

    Optuna's contract: a constraint value strictly > 0 means violated; <= 0
    means feasible. Per-filter contribution is computed by
    ``_signed_violation`` against the observation written to
    ``trial.user_attrs`` (keyed via ``_attr_key``) at ``study.tell()`` time.
    A missing observation contributes ``_UNMEASURABLE_VIOLATION`` so the
    constraint surrogate steers the sampler away from regions the run
    produced no measurement for.
    """

    def constraints_func(trial: Any) -> Sequence[float]:
        out: list[float] = []
        for sla in sla_filters:
            observed = trial.user_attrs.get(_attr_key(sla))
            out.append(_signed_violation(observed, sla))
        return out

    return constraints_func


def build_sampler(cfg: AdaptiveSearchSweep) -> Any:
    """Build an Optuna sampler honoring ``AdaptiveSearchSweep.optuna_sampler``.

    Constraints support varies by sampler:
    - GPSampler: native inequality constraints since Optuna 4.2. Requires
      ``torch`` separately for the post-startup GP fit phase — eagerly
      validated here so the failure surfaces at planner construction
      rather than mid-search after ``n_initial_points`` random trials.
    - TPESampler: native via constraints_func since Optuna 3.0.
    - BoTorchSampler: native via constraints_func; requires
      ``optuna_integration`` + ``botorch`` separately — eagerly validated
      here with a planner-friendly message naming the install command.
    """
    sla_func = build_constraints_func(cfg.sla_filters) if cfg.sla_filters else None
    outcome_func = (
        build_outcome_constraints_func(cfg.outcome_constraints)
        if cfg.outcome_constraints
        else None
    )
    if sla_func is not None and outcome_func is not None:

        def constraints_func(trial: Any) -> Sequence[float]:
            return [*sla_func(trial), *outcome_func(trial)]
    else:
        constraints_func = sla_func or outcome_func
    seed = cfg.random_seed

    if cfg.optuna_sampler == "gp":
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "GPSampler requires `torch` for post-startup GP fitting. "
                "Install torch or pick "
                "a different sampler (`--optuna-sampler tpe` is the "
                "dep-light default; `--optuna-sampler botorch` is the "
                "heavyweight constrained-EI option). Underlying "
                f"ImportError: {e}"
            ) from e
        from optuna.samplers import GPSampler

        return GPSampler(
            n_startup_trials=cfg.n_initial_points,
            seed=seed,
            constraints_func=constraints_func,
        )
    if cfg.optuna_sampler == "tpe":
        from optuna.samplers import TPESampler

        return TPESampler(
            n_startup_trials=cfg.n_initial_points,
            seed=seed,
            constraints_func=constraints_func,
        )
    if cfg.optuna_sampler == "botorch":
        try:
            from optuna_integration import BoTorchSampler
        except ImportError as e:
            raise ImportError(
                "BoTorch sampler requires the optional `botorch` extra. "
                "Install via `uv pip install -e '.[botorch]'`."
            ) from e
        n_obj = len(cfg.objectives)
        if n_obj > 1:
            if cfg.optuna_acquisition not in ("qehvi", "qnehvi", "qlognehvi"):
                raise ValueError(
                    f"Multi-objective runs require a multi-objective acquisition; "
                    f"got {cfg.optuna_acquisition!r}. Use --optuna-acquisition qlognehvi."
                )
            # qNEHVI ref point depends on observed Sobol initial points and is
            # not yet known. The planner installs the real candidates_func via
            # _maybe_install_qnehvi_candidates_func once enough points exist.
            candidates_func = None
        else:
            candidates_func = _resolve_candidates_func(cfg.optuna_acquisition)
        return BoTorchSampler(
            n_startup_trials=cfg.n_initial_points,
            seed=seed,
            constraints_func=constraints_func,
            candidates_func=candidates_func,
        )
    raise ValueError(f"unknown optuna_sampler: {cfg.optuna_sampler!r}")


def _resolve_candidates_func(acquisition: str | None) -> Any | None:
    """Resolve the BoTorch ``candidates_func`` from ``--optuna-acquisition``.

    ``None`` (no override) returns ``None`` so BoTorchSampler falls back to
    its built-in default-selection logic (single-objective unconstrained ->
    LogEI; constrained -> qEI). Otherwise we hand back the matching helper
    from ``optuna_integration.botorch``, with ``qlognei`` falling back to a
    locally-built candidates_func because Optuna v4.x ships no helper for
    BoTorch's modern ``qLogNoisyExpectedImprovement``.
    """
    if acquisition is None:
        return None
    if acquisition in ("qehvi", "qnehvi", "qlognehvi"):
        raise NotImplementedError(
            "Multi-objective candidates_func must be built via "
            "build_qnehvi_candidates_func() with a reference_point. "
            "Use build_sampler(cfg) — it handles the wiring."
        )
    try:
        from optuna_integration.botorch import (
            logei_candidates_func,
            qnei_candidates_func,
        )
    except ImportError as e:
        raise ImportError(
            "--optuna-acquisition requires the optional `botorch` extra."
            "Install via `uv pip install -e '.[botorch]'`."
        ) from e
    if acquisition in ("logei", "qlogei"):
        return logei_candidates_func
    if acquisition == "qnei":
        return qnei_candidates_func
    if acquisition == "qlognei":
        return build_qlognei_candidates_func()
    raise ValueError(f"unknown optuna_acquisition: {acquisition!r}")


def _qlognei_constraint_kwargs(
    train_obj: Any, train_con: Any
) -> tuple[Any, dict[str, Any]]:
    """Build the (train_y, kwargs) tuple for the constrained qLogNEI acquisition.

    When constraints are present, ``train_y`` is the cat-stack of objective +
    signed-violation columns; the acquisition's ``objective`` projects the
    first column and ``constraints`` lists per-column feasibility callables.
    """
    import torch
    from botorch.acquisition.objective import GenericMCObjective

    train_y = torch.cat([train_obj, train_con], dim=-1)
    n_constraints = train_con.size(1)
    return train_y, {
        "objective": GenericMCObjective(lambda Z, X: Z[..., 0]),
        "constraints": [
            (lambda Z, idx=i: Z[..., -n_constraints + idx])
            for i in range(n_constraints)
        ],
    }


def build_qlognei_candidates_func() -> Any:
    """Build a BoTorch ``candidates_func`` using ``qLogNoisyExpectedImprovement``.

    Optuna v4.x ships ``qnei_candidates_func`` (Letham et al. 2019,
    Balandat et al. 2020) but no helper for the modern
    ``qLogNoisyExpectedImprovement`` (Ament 2023,
    https://arxiv.org/abs/2310.20708). BoTorch docs strongly recommend
    qLogNEI over plain qNEI for numerical-stability reasons. Mirrors
    Optuna's ``qnei_candidates_func`` callable shape and swaps the
    acquisition class. Requires ``botorch>=0.10``; import is eager so the
    failure surfaces at planner construction.
    """
    try:
        import torch
        from botorch.acquisition.logei import qLogNoisyExpectedImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Standardize
        from botorch.optim import optimize_acqf
        from botorch.sampling import SobolQMCNormalSampler
        from botorch.utils.transforms import normalize, unnormalize
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except ImportError as e:
        raise ImportError(
            "--optuna-acquisition qlognei requires the optional `botorch` extra "
            "(for `qLogNoisyExpectedImprovement`). Install via "
            "`uv pip install -e '.[botorch]'`."
        ) from e

    from aiperf.orchestrator.search_planner._botorch_kernel import make_dsp_kernel

    # Optuna calls candidates_func with 5 positional args (see
    # optuna_integration.botorch.BoTorchSampler._sample_relative); we accept
    # *args to satisfy the keyword-only-args ergonomics rule while honoring
    # Optuna's externally-fixed callback contract.
    def qlognei_candidates_func(*args: Any) -> Any:
        train_x, train_obj, train_con, bounds, pending_x = args
        if train_obj.size(-1) != 1:
            raise ValueError("Objective may only contain single values with qLogNEI.")
        if train_con is not None:
            train_y, additional_kwargs = _qlognei_constraint_kwargs(
                train_obj, train_con
            )
        else:
            train_y = train_obj
            additional_kwargs = {}

        train_x_n = normalize(train_x, bounds=bounds)
        pending_x_n = (
            normalize(pending_x, bounds=bounds) if pending_x is not None else None
        )
        model = SingleTaskGP(
            train_x_n,
            train_y,
            covar_module=make_dsp_kernel(d=train_x_n.size(-1)),
            outcome_transform=Standardize(m=train_y.size(-1)),
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))

        acqf = qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=train_x_n,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([256])),
            X_pending=pending_x_n,
            **additional_kwargs,
        )
        standard_bounds = torch.zeros_like(bounds)
        standard_bounds[1] = 1
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=standard_bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )
        return unnormalize(candidates.detach(), bounds=bounds)

    return qlognei_candidates_func


def build_qnehvi_candidates_func(*, reference_point: list[float]) -> Any:
    """Build a BoTorch ``candidates_func`` using qLogNoisyExpectedHypervolumeImprovement.

    Multi-objective sibling of ``build_qlognei_candidates_func``. Uses a
    ``ModelListGP`` (one GP per objective) for decoupled noise modeling, per
    BoTorch's recommendation for noisy multi-objective settings (Daulton et al.
    2021, https://arxiv.org/abs/2105.08195).

    Reference point: required by qNEHVI for hypervolume computation. Trials
    worse than the reference point on any objective do not contribute to
    hypervolume. Auto-derived from Sobol initial points when the user does
    not specify per-objective ``Objective.threshold`` (see
    ``derive_reference_point``).
    """
    try:
        import torch
        from botorch.acquisition.multi_objective.logei import (
            qLogNoisyExpectedHypervolumeImprovement,
        )
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import ModelListGP, SingleTaskGP
        from botorch.models.transforms import Standardize
        from botorch.optim import optimize_acqf
        from botorch.sampling import SobolQMCNormalSampler
        from botorch.utils.transforms import normalize, unnormalize
        from gpytorch.mlls import SumMarginalLogLikelihood
    except ImportError as e:
        raise ImportError(
            "--optuna-acquisition qlognehvi requires the optional `botorch` extra. "
            "Install via `uv pip install -e '.[botorch]'`."
        ) from e

    ref_point_t = torch.tensor(reference_point, dtype=torch.double)

    from aiperf.orchestrator.search_planner._botorch_kernel import make_dsp_kernel

    def qnehvi_candidates_func(*args: Any) -> Any:
        train_x, train_obj, train_con, bounds, pending_x = args
        if train_obj.size(-1) < 2:
            raise ValueError(
                f"qLogNEHVI requires >=2 objectives; got {train_obj.size(-1)}."
            )

        train_x_n = normalize(train_x, bounds=bounds)
        pending_x_n = (
            normalize(pending_x, bounds=bounds) if pending_x is not None else None
        )

        d = train_x_n.size(-1)
        models = [
            SingleTaskGP(
                train_x_n,
                train_obj[..., i : i + 1],
                covar_module=make_dsp_kernel(d=d),
                outcome_transform=Standardize(m=1),
            )
            for i in range(train_obj.size(-1))
        ]
        model = ModelListGP(*models)
        fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))

        constraint_kwargs: dict[str, Any] = {}
        if train_con is not None:
            n_constraints = train_con.size(1)
            constraint_kwargs["constraints"] = [
                (lambda Z, idx=i: Z[..., -n_constraints + idx])
                for i in range(n_constraints)
            ]

        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point_t,
            X_baseline=train_x_n,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
            X_pending=pending_x_n,
            prune_baseline=True,
            **constraint_kwargs,
        )
        standard_bounds = torch.zeros_like(bounds)
        standard_bounds[1] = 1
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=standard_bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )
        return unnormalize(candidates.detach(), bounds=bounds)

    return qnehvi_candidates_func


def derive_reference_point(
    objectives: list[Objective], observed: list[list[float]]
) -> list[float]:
    """Compute the qNEHVI reference point from objectives + observed Sobol points.

    Per-objective: if ``Objective.threshold`` is set, use it verbatim.
    Otherwise auto-derive from observed initial points using the
    worst-observed value plus a 5% directional slack so the worst Sobol
    point still contributes non-zero hypervolume (BoTorch tutorial
    recommendation: https://botorch.org/docs/tutorials/multi_objective_bo).
    Raises ``ValueError`` when an objective has no threshold and no Sobol
    observations exist yet — caller must defer ref-point construction to
    after the initial round.
    """
    from aiperf.common.enums import OptimizationDirection

    rp: list[float] = []
    for i, obj in enumerate(objectives):
        if obj.threshold is not None:
            rp.append(float(obj.threshold))
            continue
        if not observed:
            raise ValueError(
                f"Objective {i} ({obj.metric!r}) has no threshold set and no "
                "Sobol initial points have been observed yet. Set "
                "Objective.threshold or provide initial observations."
            )
        col = [row[i] for row in observed]
        if obj.direction == OptimizationDirection.MAXIMIZE:
            worst = min(col)
            rp.append(worst - _REFERENCE_POINT_SLACK * abs(worst))
        else:
            worst = max(col)
            rp.append(worst + _REFERENCE_POINT_SLACK * abs(worst))
    return rp


def _outcome_attr_key(c: OutcomeConstraint) -> str:
    """Trial.user_attr key for an outcome-constraint observation.

    Distinct namespace from SLA filters (``sla:...``) so the two coexist
    when both are configured on the same sweep.
    """
    return f"outcome:{c.metric}"


def _outcome_signed_violation(observed: float | None, c: OutcomeConstraint) -> float:
    """Optuna convention: <= 0 feasible, > 0 violates.

    For ``op="<="``: violation = ``observed - bound``.
    For ``op=">="``: violation = ``bound - observed``.
    For ``op="=="``: violation = ``|observed - bound|`` (always >= 0).
    Missing observation collapses to ``_UNMEASURABLE_VIOLATION`` so the
    constraint surrogate steers away — same policy as ``_signed_violation``.
    """
    if observed is None:
        return _UNMEASURABLE_VIOLATION
    if c.op == "<=":
        return observed - c.bound
    if c.op == ">=":
        return c.bound - observed
    return abs(observed - c.bound)


def build_outcome_constraints_func(
    outcome_constraints: list[OutcomeConstraint],
) -> Callable[[Any], Sequence[float]]:
    """Build an Optuna ``constraints_func`` for ``outcome_constraints``.

    Reads observations from ``trial.user_attrs[f"outcome:{metric}"]`` written
    by the planner at ``study.tell()`` time. Composable with the SLA
    constraints function — see ``build_sampler``.
    """

    def constraints_func(trial: Any) -> Sequence[float]:
        return [
            _outcome_signed_violation(trial.user_attrs.get(_outcome_attr_key(c)), c)
            for c in outcome_constraints
        ]

    return constraints_func


def compute_hypervolume(
    observed: list[list[float]],
    objectives: list[Objective],
    ref_point: list[float],
) -> float:
    """Hypervolume of the observed Pareto front under the given reference point.

    Uses BoTorch's ``Hypervolume``. qNEHVI maximizes by convention, so
    MINIMIZE objectives are sign-flipped; the reference point is sign-flipped
    in lock-step. Returns ``0.0`` when no observed point dominates the
    reference (degenerate but safe — used as the plateau-tracking signal).
    """
    import torch
    from botorch.utils.multi_objective.hypervolume import Hypervolume

    from aiperf.common.enums import OptimizationDirection

    n = len(objectives)
    signs = [
        1.0 if obj.direction == OptimizationDirection.MAXIMIZE else -1.0
        for obj in objectives
    ]
    flipped_obs = [[signs[i] * row[i] for i in range(n)] for row in observed]
    flipped_ref = [signs[i] * ref_point[i] for i in range(n)]
    pareto_y = torch.tensor(flipped_obs, dtype=torch.double)
    ref_t = torch.tensor(flipped_ref, dtype=torch.double)
    hv = Hypervolume(ref_point=ref_t)
    return float(hv.compute(pareto_y))
