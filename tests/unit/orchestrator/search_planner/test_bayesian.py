# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for BayesianSearchPlanner.

Skopt is a soft dep; tests skip when not installed. Local CI must install
the `bo` extra.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

# BayesianSearchPlanner now subclasses OptunaSearchPlanner; these tests patch
# the sampler to TPE so they exercise planner behavior without the heavy BoTorch path.
from aiperf.common.models.export_models import JsonMetricResult  # noqa: E402
from aiperf.config.config import BenchmarkConfig  # noqa: E402
from aiperf.config.sweep import (  # noqa: E402
    AdaptiveObjective,
    AdaptiveSearchSweep,
    SweepVariation,
)
from aiperf.config.sweep.adaptive import SearchSpaceDimension  # noqa: E402
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection  # noqa: E402
from aiperf.orchestrator.models import RunResult  # noqa: E402
from aiperf.orchestrator.search_planner.bayesian import (  # noqa: E402
    BayesianSearchPlanner,
)


@pytest.fixture(autouse=True)
def _use_tpe_sampler_for_planner_logic_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aiperf.orchestrator.search_planner import optuna_planner
    from aiperf.orchestrator.search_planner._optuna_helpers import build_sampler

    def _build_sampler(cfg: AdaptiveSearchSweep) -> Any:
        if cfg.optuna_sampler == "botorch":
            cfg = cfg.model_copy(
                update={"optuna_sampler": "tpe", "optuna_acquisition": None}
            )
        return build_sampler(cfg)

    monkeypatch.setattr(optuna_planner, "build_sampler", _build_sampler)


def _base_config() -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {"urls": ["http://x"], "type": "chat"},
            "datasets": [{"name": "default", "type": "synthetic"}],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "concurrency": 1,
                    "requests": 10,
                }
            ],
        }
    )


def _cfg(max_iterations: int = 5, **overrides) -> AdaptiveSearchSweep:
    # Translate legacy flat objective_* overrides into the nested objective.
    obj_metric = overrides.pop("objective_metric", "output_token_throughput")
    obj_stat = overrides.pop("objective_stat", "avg")
    obj_direction = overrides.pop("objective_direction", OptimizationDirection.MAXIMIZE)
    kwargs: dict = dict(
        planner="bayesian",
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency", lo=1, hi=100, kind="int"
            ),
        ],
        objectives=[
            AdaptiveObjective(
                metric=obj_metric,
                stat=obj_stat,
                direction=obj_direction,
            )
        ],
        max_iterations=max_iterations,
        n_initial_points=2,
        random_seed=42,
    )
    kwargs.update(overrides)
    return AdaptiveSearchSweep(**kwargs)


def test_ask_returns_cfg_and_variation():
    planner = BayesianSearchPlanner(_base_config(), _cfg(max_iterations=5))
    proposal = planner.ask()
    assert proposal is not None
    cfg, variation = proposal
    assert variation.index == 0
    assert variation.label.startswith("search_iter_")
    assert "phases.profiling.concurrency" in variation.values
    proposed = variation.values["phases.profiling.concurrency"]
    assert 1 <= proposed <= 100
    assert isinstance(proposed, int)  # int dim → integer
    # The mutated cfg must reflect the proposed value.
    profiling = next(p for p in cfg.phases if p.name == "profiling")
    assert profiling.concurrency == proposed


def test_ask_returns_none_after_max_iterations():
    planner = BayesianSearchPlanner(_base_config(), _cfg(max_iterations=3))
    for _ in range(3):
        proposal = planner.ask()
        assert proposal is not None
        _, variation = proposal
        planner.tell(variation, [_make_result(variation, throughput=100.0)])
    assert planner.ask() is None


def test_record_extracts_avg_from_summary_metrics_and_signs_for_maximize():
    planner = BayesianSearchPlanner(_base_config(), _cfg(max_iterations=5))
    proposal = planner.ask()
    assert proposal is not None
    _, variation = proposal
    planner.tell(variation, [_make_result(variation, throughput=42.5)])
    history = planner.history()
    assert len(history) == 1
    assert history[0].objective_value == pytest.approx(42.5)


def test_record_skips_failed_runs():
    planner = BayesianSearchPlanner(_base_config(), _cfg(max_iterations=5))
    _, variation = planner.ask()
    failed = RunResult(label="x", success=False, error="boom")
    planner.tell(variation, [failed, _make_result(variation, throughput=10.0)])
    assert planner.history()[0].objective_value == pytest.approx(10.0)


def test_record_with_no_successful_runs_records_none():
    planner = BayesianSearchPlanner(_base_config(), _cfg(max_iterations=5))
    _, variation = planner.ask()
    planner.tell(variation, [RunResult(label="x", success=False)])
    assert planner.history()[0].objective_value is None


def test_minimize_direction_signs_correctly():
    cfg = _cfg(max_iterations=3, objective_direction=OptimizationDirection.MINIMIZE)
    planner = BayesianSearchPlanner(_base_config(), cfg)
    _, v1 = planner.ask()
    planner.tell(v1, [_make_result(v1, throughput=10.0)])
    _, v2 = planner.ask()
    # If skopt sees signed values correctly, asking again does not crash.
    assert v2 is not None


def test_is_converged_on_max_iterations_exhausted():
    planner = BayesianSearchPlanner(
        _base_config(), _cfg(max_iterations=2, n_initial_points=1, plateau_window=2)
    )
    assert not planner.is_converged()
    for _ in range(2):
        _, v = planner.ask()
        planner.tell(v, [_make_result(v, throughput=1.0)])
    assert planner.is_converged()


def test_is_converged_on_plateau():
    cfg = _cfg(max_iterations=20, plateau_window=3, plateau_threshold=0.05)
    planner = BayesianSearchPlanner(_base_config(), cfg)
    for _ in range(3):
        _, v = planner.ask()
        planner.tell(v, [_make_result(v, throughput=100.0)])
    assert planner.is_converged()


def _make_result(variation: SweepVariation, *, throughput: float) -> RunResult:
    return RunResult(
        label="t",
        success=True,
        summary_metrics={
            "output_token_throughput": JsonMetricResult(unit="tok/s", avg=throughput),
        },
        variation_label=variation.label,
        variation_values=variation.values,
    )


# ----------------------------------------------------------------------------
# Mathematical-correctness tests added in the post-integration math fix-up.
# Cover: sample-vs-population variance for plateau, mean-≈-0 incoherence guard,
# failed-iteration loss sign-flip consistency and empty-history sentinel.
# ----------------------------------------------------------------------------


def test_plateau_uses_sample_variance_not_population():
    """Population variance (/n) trips ~12% earlier than sample variance (/n-1).

    With objective values [99.0, 100.0, 101.0] (mean=100, max-min=2):
      population stddev = sqrt((1+0+1)/3) = 0.8165 → CV = 0.008165
      sample stddev     = sqrt((1+0+1)/2) = 1.0    → CV = 0.01
    A threshold of 0.009 must NOT declare convergence (sample CV is over);
    population formula would have falsely declared converged.
    """
    cfg = _cfg(max_iterations=20, plateau_window=3, plateau_threshold=0.009)
    planner = BayesianSearchPlanner(_base_config(), cfg)
    for value in (99.0, 100.0, 101.0):
        _, v = planner.ask()
        planner.tell(v, [_make_result(v, throughput=value)])
    assert not planner.is_converged()


def test_plateau_refuses_convergence_when_mean_is_zero():
    """A zero-mean window has no scale; coefficient of variation is undefined.

    The threshold is *relative* — applying it as absolute against zero-mean
    values is dimensionally wrong (compares unitless ratio against the
    metric's own units). Refuse to declare convergence in that regime.
    """
    cfg = _cfg(max_iterations=20, plateau_window=3, plateau_threshold=0.5)
    planner = BayesianSearchPlanner(_base_config(), cfg)
    # Symmetric values around zero: mean → 0, classical CV undefined.
    for value in (-1e-6, 0.0, 1e-6):
        _, v = planner.ask()
        planner.tell(v, [_make_result(v, throughput=value)])
    assert not planner.is_converged()


@pytest.mark.skip(
    reason="Probes BayesianSearchPlanner private attrs (_opt, "
    "_failed_iteration_loss) from the skopt-direct implementation. "
    "HEAD's planner subclasses OptunaSearchPlanner — coverage lives "
    "in test_optuna_helpers.py / test_optuna_planner.py."
)
def test_failed_iteration_loss_uses_finite_sentinel_with_no_history():
    """No prior successful runs → sentinel loss; never inf/nan; same magnitude
    regardless of direction (so the GP kernel matrix stays well-posed)."""
    from aiperf.orchestrator.search_planner._bayesian_helpers import (
        NO_DATA_SENTINEL_LOSS,
    )

    for direction in (OptimizationDirection.MAXIMIZE, OptimizationDirection.MINIMIZE):
        cfg = _cfg(max_iterations=5, n_initial_points=1, objective_direction=direction)
        planner = BayesianSearchPlanner(_base_config(), cfg)
        loss = planner._failed_iteration_loss()
        assert math.isfinite(loss)
        assert loss == pytest.approx(NO_DATA_SENTINEL_LOSS)


@pytest.mark.skip(
    reason="Probes BayesianSearchPlanner private attrs (_opt, "
    "_failed_iteration_loss) from the skopt-direct implementation. "
    "HEAD's planner subclasses OptunaSearchPlanner — coverage lives "
    "in test_optuna_helpers.py / test_optuna_planner.py."
)
def test_failed_iteration_loss_is_worse_than_worst_real_loss_maximize():
    """With prior MAXIMIZE successes, fallback loss must exceed worst real loss
    (skopt minimizes, so 'worse' = larger loss)."""
    cfg = _cfg(
        max_iterations=10,
        n_initial_points=1,
        objective_direction=OptimizationDirection.MAXIMIZE,
    )
    planner = BayesianSearchPlanner(_base_config(), cfg)
    # Tell two successful iterations; objectives 100.0 and 50.0.
    # In skopt's loss space (MAXIMIZE → negate): -100.0 and -50.0.
    # Worst loss = max(-100, -50) = -50.0.
    for value in (100.0, 50.0):
        _, v = planner.ask()
        planner.tell(v, [_make_result(v, throughput=value)])
    fallback = planner._failed_iteration_loss()
    # Fallback must be strictly worse than -50 (i.e., greater than -50).
    assert fallback > -50.0
    # And finite, not inf or nan.
    assert math.isfinite(fallback)


@pytest.mark.skip(
    reason="Probes BayesianSearchPlanner private attrs (_opt, "
    "_failed_iteration_loss) from the skopt-direct implementation. "
    "HEAD's planner subclasses OptunaSearchPlanner — coverage lives "
    "in test_optuna_helpers.py / test_optuna_planner.py."
)
def test_failed_iteration_loss_is_worse_than_worst_real_loss_minimize():
    """With prior MINIMIZE successes (loss = objective passthrough), fallback
    loss must exceed the largest seen objective."""
    cfg = _cfg(
        max_iterations=10,
        n_initial_points=1,
        objective_direction=OptimizationDirection.MINIMIZE,
    )
    planner = BayesianSearchPlanner(_base_config(), cfg)
    for value in (10.0, 25.0):
        _, v = planner.ask()
        planner.tell(v, [_make_result(v, throughput=value)])
    fallback = planner._failed_iteration_loss()
    assert fallback > 25.0
    assert math.isfinite(fallback)


def test_objective_to_loss_sign_consistency_round_trip():
    """A successful tell and a failed-fallback tell must use the same sign
    convention so skopt's history is internally consistent."""
    cfg_max = _cfg(
        max_iterations=10,
        n_initial_points=1,
        objective_direction=OptimizationDirection.MAXIMIZE,
    )
    planner_max = BayesianSearchPlanner(_base_config(), cfg_max)
    # MAXIMIZE: objective 50 → loss -50.
    assert planner_max._objective_to_loss(50.0) == pytest.approx(-50.0)
    # MAXIMIZE: objective 0 → loss -0 (== 0).
    assert planner_max._objective_to_loss(0.0) == pytest.approx(0.0)

    cfg_min = _cfg(
        max_iterations=10,
        n_initial_points=1,
        objective_direction=OptimizationDirection.MINIMIZE,
    )
    planner_min = BayesianSearchPlanner(_base_config(), cfg_min)
    # MINIMIZE: passthrough.
    assert planner_min._objective_to_loss(50.0) == pytest.approx(50.0)
    assert planner_min._objective_to_loss(-50.0) == pytest.approx(-50.0)


# ----------------------------------------------------------------------------
# Literature-driven improvements added after research-paper review.
# Letham et al. 2017 (arXiv:1706.07094): pass per-trial observations to the GP
# rather than pre-averaging — lets skopt's GP fit the noise term properly.
# Hyperopt no_progress_loss / skopt HollowIterationsStopper: improvement-over-
# best patience as a second termination signal.
# ----------------------------------------------------------------------------


@pytest.mark.skip(
    reason="Probes BayesianSearchPlanner private attrs (_opt, "
    "_failed_iteration_loss) from the skopt-direct implementation. "
    "HEAD's planner subclasses OptunaSearchPlanner — coverage lives "
    "in test_optuna_helpers.py / test_optuna_planner.py."
)
def test_per_trial_observations_passed_to_skopt(monkeypatch):
    """tell() with N>=2 trials must pass N (x, y) pairs to skopt.Optimizer.tell.

    Pre-fix, the planner pre-averaged trials and called `opt.tell(x, mean_y)`.
    Post-fix it calls `opt.tell([x]*N, [y1, y2, ...])` so the GP sees the
    within-point variance — matches Letham et al. 2017 (arXiv:1706.07094).
    """
    planner = BayesianSearchPlanner(_base_config(), _cfg(max_iterations=5))

    captured: dict = {"calls": []}
    real_tell = planner._opt.tell

    def spy_tell(x, y, *args, **kwargs):
        captured["calls"].append((x, y))
        return real_tell(x, y, *args, **kwargs)

    monkeypatch.setattr(planner._opt, "tell", spy_tell)

    _, variation = planner.ask()
    # Three trials at the same point with distinct objectives.
    trial_results = [
        _make_result(variation, throughput=10.0),
        _make_result(variation, throughput=12.0),
        _make_result(variation, throughput=11.0),
    ]
    planner.tell(variation, trial_results)

    assert len(captured["calls"]) == 1
    x_passed, y_passed = captured["calls"][0]
    # Should be a list of x's (one per trial) and a list of y's.
    assert isinstance(x_passed, list)
    assert len(x_passed) == 3
    assert isinstance(y_passed, list)
    assert len(y_passed) == 3
    # All x's identical.
    assert all(xi == x_passed[0] for xi in x_passed)
    # Per-trial losses for MAXIMIZE = -throughput. Order may vary; check sets.
    assert sorted(y_passed) == sorted([-10.0, -12.0, -11.0])
    # History stores the mean for plateau detection.
    assert planner.history()[0].objective_value == pytest.approx(11.0)


@pytest.mark.skip(
    reason="Probes BayesianSearchPlanner private attrs (_opt, "
    "_failed_iteration_loss) from the skopt-direct implementation. "
    "HEAD's planner subclasses OptunaSearchPlanner — coverage lives "
    "in test_optuna_helpers.py / test_optuna_planner.py."
)
def test_single_trial_path_uses_scalar_tell(monkeypatch):
    """When only one trial succeeds, tell() should pass scalar x, y.

    Calling `opt.tell([x], [y])` on skopt 0.10 works but is needlessly
    awkward; the scalar form is what every prior test exercised and what
    the rest of the test suite relies on.
    """
    planner = BayesianSearchPlanner(_base_config(), _cfg(max_iterations=5))

    captured: dict = {"calls": []}
    real_tell = planner._opt.tell

    def spy_tell(x, y, *args, **kwargs):
        captured["calls"].append((x, y))
        return real_tell(x, y, *args, **kwargs)

    monkeypatch.setattr(planner._opt, "tell", spy_tell)

    _, variation = planner.ask()
    planner.tell(variation, [_make_result(variation, throughput=42.0)])

    x_passed, y_passed = captured["calls"][0]
    assert isinstance(y_passed, float)
    assert y_passed == pytest.approx(-42.0)


def test_improvement_patience_stops_after_no_progress():
    """is_converged() returns True once `improvement_patience` consecutive
    iterations show no improvement on best-so-far. Mirrors skopt's
    HollowIterationsStopper / Hyperopt's no_progress_loss."""
    cfg = _cfg(
        max_iterations=20,
        n_initial_points=1,
        improvement_patience=3,
        # Disable plateau detection by setting an unreachable threshold; we
        # want this test to specifically exercise the patience stop, not CV.
        plateau_window=20,
        plateau_threshold=1e-9,
    )
    planner = BayesianSearchPlanner(_base_config(), cfg)

    # First iteration sets the best.
    _, v = planner.ask()
    planner.tell(v, [_make_result(v, throughput=100.0)])
    assert not planner.is_converged()

    # Three subsequent iterations all worse-than-best (throughput=50 << 100).
    for _ in range(3):
        _, v = planner.ask()
        planner.tell(v, [_make_result(v, throughput=50.0)])

    # 3 consecutive no-improvement iterations >= patience(3) → converged.
    assert planner.is_converged()


def test_improvement_patience_resets_on_better_value():
    """A new best resets the patience counter."""
    cfg = _cfg(
        max_iterations=20,
        n_initial_points=1,
        improvement_patience=3,
        plateau_window=20,
        plateau_threshold=1e-9,
    )
    planner = BayesianSearchPlanner(_base_config(), cfg)

    # Iterations: [100 (best), 50, 50, 200 (new best!), 50, 50]
    # Should NOT converge: after the new best at iter 4, only 2 no-improvement.
    for value in (100.0, 50.0, 50.0, 200.0, 50.0, 50.0):
        _, v = planner.ask()
        planner.tell(v, [_make_result(v, throughput=value)])
        # After the 4th iteration (new best=200), counter resets to 0.

    # 2 no-improvement iterations after the 200 new-best < patience(3).
    assert not planner.is_converged()


def test_improvement_patience_handles_failed_iterations():
    """A failed iteration counts toward patience (no progress IS no progress)."""
    cfg = _cfg(
        max_iterations=20,
        n_initial_points=1,
        improvement_patience=2,
        plateau_window=20,
        plateau_threshold=1e-9,
    )
    planner = BayesianSearchPlanner(_base_config(), cfg)

    _, v = planner.ask()
    planner.tell(v, [_make_result(v, throughput=100.0)])
    # Two failed iterations: each produces no objective, both count as
    # no-improvement.
    for _ in range(2):
        _, v = planner.ask()
        planner.tell(v, [RunResult(label="x", success=False)])

    assert planner.is_converged()


# ----------------------------------------------------------------------------
# Convergence-reason tracking — surfaced in logs and search_history.json.
# ----------------------------------------------------------------------------


def test_convergence_reason_max_iterations():
    cfg = _cfg(max_iterations=2, n_initial_points=1, plateau_window=10)
    planner = BayesianSearchPlanner(_base_config(), cfg)
    assert planner.convergence_reason() is None
    for _ in range(2):
        _, v = planner.ask()
        planner.tell(v, [_make_result(v, throughput=1.0)])
    assert planner.is_converged()
    assert planner.convergence_reason() == "max_iterations"


def test_convergence_reason_improvement_patience():
    cfg = _cfg(
        max_iterations=20,
        n_initial_points=1,
        improvement_patience=2,
        plateau_window=20,  # disable CV plateau
        plateau_threshold=1e-9,
    )
    planner = BayesianSearchPlanner(_base_config(), cfg)
    _, v = planner.ask()
    planner.tell(v, [_make_result(v, throughput=100.0)])
    for _ in range(2):
        _, v = planner.ask()
        planner.tell(v, [_make_result(v, throughput=50.0)])
    assert planner.is_converged()
    assert planner.convergence_reason() == "improvement_patience"


def test_convergence_reason_plateau_cv():
    # Patience set high so it can't trigger first; CV should fire.
    cfg = _cfg(
        max_iterations=20,
        n_initial_points=1,
        improvement_patience=99,
        plateau_window=3,
        plateau_threshold=0.05,
    )
    planner = BayesianSearchPlanner(_base_config(), cfg)
    for _ in range(3):
        _, v = planner.ask()
        planner.tell(v, [_make_result(v, throughput=100.0)])
    assert planner.is_converged()
    assert planner.convergence_reason() == "plateau_cv"


def test_convergence_reason_none_until_fired():
    cfg = _cfg(max_iterations=20, n_initial_points=1, improvement_patience=10)
    planner = BayesianSearchPlanner(_base_config(), cfg)
    for i in range(3):
        _, v = planner.ask()
        planner.tell(v, [_make_result(v, throughput=100.0 + i)])  # always improving
    assert not planner.is_converged()
    assert planner.convergence_reason() is None


def test_convergence_reason_in_search_history_json(tmp_path):
    """End-to-end: convergence_reason propagates through write_search_history."""
    import orjson

    from aiperf.exporters.search_history import write_search_history

    cfg = _cfg(max_iterations=3, n_initial_points=1, improvement_patience=10)
    planner = BayesianSearchPlanner(_base_config(), cfg)
    for _ in range(3):
        _, v = planner.ask()
        planner.tell(v, [_make_result(v, throughput=10.0)])
    assert planner.is_converged()
    reason = planner.convergence_reason()

    write_search_history(tmp_path, planner.history(), cfg, convergence_reason=reason)
    payload = orjson.loads((tmp_path / "search_history.json").read_bytes())
    assert payload["convergence_reason"] == "max_iterations"
