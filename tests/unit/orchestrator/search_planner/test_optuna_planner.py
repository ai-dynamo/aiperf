# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for OptunaSearchPlanner.

Optuna is a core dependency. The BoTorch/Torch sampler stack remains optional,
so only BoTorch-specific cases skip when that stack is unavailable.
"""

from __future__ import annotations

import pytest
from pytest import param

optuna = pytest.importorskip("optuna")

# Imports below depend on optuna being importable. pytest.importorskip must
# precede them so the whole module is skipped when the `optuna` extra is absent.
from aiperf.common.models.export_models import JsonMetricResult  # noqa: E402
from aiperf.config.config import BenchmarkConfig  # noqa: E402
from aiperf.config.sweep import (  # noqa: E402
    AdaptiveSearchSweep,
    Objective,
    SweepVariation,
)
from aiperf.config.sweep.adaptive import (  # noqa: E402
    SearchSpaceDimension,
    SLAFilter,
)
from aiperf.orchestrator.aggregation.sweep import OptimizationDirection  # noqa: E402
from aiperf.orchestrator.models import RunResult  # noqa: E402
from aiperf.orchestrator.search_planner._optuna_helpers import (  # noqa: E402
    _UNMEASURABLE_VIOLATION,
    _attr_key,
    _signed_violation,
)
from aiperf.orchestrator.search_planner.optuna_planner import (  # noqa: E402
    OptunaSearchPlanner,
)


def _base_config() -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {"urls": ["http://x"], "type": "chat"},
            "datasets": [{"name": "profiling", "type": "synthetic"}],
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


def _cfg(
    *,
    max_iterations: int = 5,
    n_initial_points: int = 2,
    sla_filters: list[SLAFilter] | None = None,
    optuna_sampler: str = "tpe",
    kind: str = "int",
    extra_dims: list[SearchSpaceDimension] | None = None,
    **overrides,
) -> AdaptiveSearchSweep:
    if sla_filters is None:
        sla_filters = [
            SLAFilter(
                metric_tag="time_to_first_token",
                stat="p95",
                op="lt",
                threshold=200.0,
            )
        ]
    dims: list[SearchSpaceDimension] = [
        SearchSpaceDimension(
            path="phases.profiling.concurrency", lo=1, hi=100, kind=kind
        )
    ]
    if extra_dims:
        dims.extend(extra_dims)
    obj_metric = overrides.pop("objective_metric", "output_token_throughput")
    obj_stat = overrides.pop("objective_stat", "avg")
    obj_direction = overrides.pop("objective_direction", OptimizationDirection.MAXIMIZE)
    kwargs: dict = dict(
        search_space=dims,
        objectives=[
            Objective(
                metric=obj_metric,
                stat=obj_stat,
                direction=obj_direction,
            )
        ],
        max_iterations=max_iterations,
        n_initial_points=n_initial_points,
        random_seed=42,
        sla_filters=sla_filters,
        optuna_sampler=optuna_sampler,
    )
    kwargs.update(overrides)
    return AdaptiveSearchSweep(**kwargs)


def _make_result(
    variation: SweepVariation,
    *,
    throughput: float,
    ttft_p95: float | None = 100.0,
) -> RunResult:
    summary: dict[str, JsonMetricResult] = {
        "output_token_throughput": JsonMetricResult(unit="tok/s", avg=throughput),
    }
    if ttft_p95 is not None:
        summary["time_to_first_token"] = JsonMetricResult(unit="ms", p95=ttft_p95)
    return RunResult(
        label="t",
        success=True,
        summary_metrics=summary,
        variation_label=variation.label,
        variation_values=variation.values,
    )


# ----------------------------------------------------------------------------
# 1. Construction.
# ----------------------------------------------------------------------------


def test_construction_succeeds_with_valid_config():
    """Building with a valid AdaptiveSearchSweep succeeds."""
    planner = OptunaSearchPlanner(_base_config(), _cfg())
    assert planner._study is not None
    assert planner.history() == []


def test_construction_without_optuna_raises_clear_error(monkeypatch):
    """Without optuna installed → ImportError names dependency setup.

    Simulated by patching the import inside the planner constructor so the
    test runs even with optuna installed.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "optuna":
            raise ImportError("simulated missing optuna")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"core `optuna` dependency"):
        OptunaSearchPlanner(_base_config(), _cfg())


# ----------------------------------------------------------------------------
# 2. ask() produces a BenchmarkConfig with the right swept dim values.
# ----------------------------------------------------------------------------


def test_ask_int_dim_returns_int_within_bounds():
    planner = OptunaSearchPlanner(_base_config(), _cfg(kind="int"))
    proposal = planner.ask()
    assert proposal is not None
    cfg, variation = proposal
    proposed = variation.values["phases.profiling.concurrency"]
    assert isinstance(proposed, int)
    assert 1 <= proposed <= 100
    profiling = next(p for p in cfg.phases if p.name == "profiling")
    assert profiling.concurrency == proposed
    assert planner._pending_trial is not None


def test_ask_float_dim_returns_float():
    # Use a float-valued path (endpoint.timeout) for the float-kind test —
    # phases.profiling.concurrency is int-typed in BenchmarkConfig.
    cfg = AdaptiveSearchSweep(
        search_space=[
            SearchSpaceDimension(path="endpoint.timeout", lo=1.0, hi=100.0, kind="real")
        ],
        objectives=[
            Objective(
                metric="output_token_throughput",
                stat="avg",
                direction=OptimizationDirection.MAXIMIZE,
            )
        ],
        max_iterations=5,
        n_initial_points=2,
        random_seed=42,
        sla_filters=[
            SLAFilter(
                metric_tag="time_to_first_token",
                stat="p95",
                op="lt",
                threshold=200.0,
            )
        ],
        optuna_sampler="tpe",
    )
    planner = OptunaSearchPlanner(_base_config(), cfg)
    proposal = planner.ask()
    assert proposal is not None
    _, variation = proposal
    proposed = variation.values["endpoint.timeout"]
    assert isinstance(proposed, float)
    assert 1.0 <= proposed <= 100.0


# ----------------------------------------------------------------------------
# 3. tell() consumes the pending trial.
# ----------------------------------------------------------------------------


def test_tell_consumes_pending_trial():
    planner = OptunaSearchPlanner(_base_config(), _cfg())
    _, variation = planner.ask()
    assert planner._pending_trial is not None
    planner.tell(variation, [_make_result(variation, throughput=10.0, ttft_p95=50.0)])
    assert planner._pending_trial is None


def test_tell_without_matching_ask_raises():
    planner = OptunaSearchPlanner(_base_config(), _cfg())
    fake_variation = SweepVariation(
        index=0, label="x", values={"phases.profiling.concurrency": 1}
    )
    with pytest.raises(RuntimeError, match="without matching ask"):
        planner.tell(fake_variation, [])


# ----------------------------------------------------------------------------
# 4. Convergence: max_iterations.
# ----------------------------------------------------------------------------


def test_is_converged_on_max_iterations_exhausted():
    planner = OptunaSearchPlanner(
        _base_config(),
        _cfg(max_iterations=5, n_initial_points=1, plateau_window=20),
    )
    assert not planner.is_converged()
    for _ in range(5):
        proposal = planner.ask()
        assert proposal is not None
        _, v = proposal
        planner.tell(v, [_make_result(v, throughput=10.0, ttft_p95=50.0)])
    assert planner.is_converged()
    assert planner.convergence_reason() == "max_iterations"
    assert planner.ask() is None


# ----------------------------------------------------------------------------
# 5. Convergence: improvement_patience.
# ----------------------------------------------------------------------------


def test_is_converged_on_improvement_patience():
    """Same objective every iteration → no improvement → patience fires."""
    cfg = _cfg(
        max_iterations=20,
        n_initial_points=1,
        improvement_patience=3,
        plateau_window=20,  # disable CV so patience is the only signal
        plateau_threshold=1e-9,
    )
    planner = OptunaSearchPlanner(_base_config(), cfg)
    # First iteration sets the best.
    _, v = planner.ask()
    planner.tell(v, [_make_result(v, throughput=100.0, ttft_p95=50.0)])
    # Three subsequent iterations all worse-than-best.
    for _ in range(3):
        _, v = planner.ask()
        planner.tell(v, [_make_result(v, throughput=50.0, ttft_p95=50.0)])
    assert planner.is_converged()
    assert planner.convergence_reason() == "improvement_patience"


# ----------------------------------------------------------------------------
# 6. Convergence: plateau_cv.
# ----------------------------------------------------------------------------


def test_is_converged_on_plateau_cv():
    cfg = _cfg(
        max_iterations=20,
        n_initial_points=1,
        improvement_patience=99,  # disable patience so CV is the only signal
        plateau_window=3,
        plateau_threshold=0.05,
    )
    planner = OptunaSearchPlanner(_base_config(), cfg)
    for _ in range(3):
        _, v = planner.ask()
        planner.tell(v, [_make_result(v, throughput=100.0, ttft_p95=50.0)])
    assert planner.is_converged()
    assert planner.convergence_reason() == "plateau_cv"


# ----------------------------------------------------------------------------
# 7. boundary_summary() for 1D search-space.
# ----------------------------------------------------------------------------


def test_boundary_summary_1d_search_space_populates_both_blocks():
    """Mixed feasibility iterations → boundary_summary reports both bounds."""
    planner = OptunaSearchPlanner(_base_config(), _cfg(max_iterations=5))
    # Force three iterations with synthesized variation values + verdicts:
    # We let optuna ask for the actual proposal but inject a deterministic
    # ttft so we control feasibility.
    seen = []
    for ttft in (50.0, 50.0, 500.0):  # last one violates p95<200
        _, v = planner.ask()
        seen.append(v.values["phases.profiling.concurrency"])
        planner.tell(v, [_make_result(v, throughput=10.0, ttft_p95=ttft)])
    summary = planner.boundary_summary()
    assert summary is not None
    assert summary["swept_dim_path"] == "phases.profiling.concurrency"
    # At least one feasible iteration → feasible_max set.
    assert summary["feasible_max"] is not None
    # The infeasible one we forced → infeasible_min set.
    assert summary["infeasible_min"] is not None


# ----------------------------------------------------------------------------
# 8. boundary_summary() for 2D search-space.
# ----------------------------------------------------------------------------


def test_boundary_summary_returns_none_for_multi_dim():
    extra = SearchSpaceDimension(
        path="phases.profiling.requests", lo=1, hi=100, kind="int"
    )
    planner = OptunaSearchPlanner(
        _base_config(), _cfg(max_iterations=3, extra_dims=[extra])
    )
    _, v = planner.ask()
    planner.tell(v, [_make_result(v, throughput=10.0, ttft_p95=50.0)])
    assert planner.boundary_summary() is None


# ----------------------------------------------------------------------------
# 9. Sampler selection (parametrized).
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sampler_name,expected_attr",
    [
        param("gp", "GPSampler", id="gp"),
        param("tpe", "TPESampler", id="tpe"),
        param("botorch", "BoTorchSampler", id="botorch"),
    ],
)  # fmt: skip
def test_sampler_selection_constructs_expected_class(sampler_name, expected_attr):
    if sampler_name == "gp":
        # GPSampler eagerly imports torch in build_sampler; skip when absent.
        pytest.importorskip("torch")
    if sampler_name == "botorch":
        pytest.importorskip("optuna_integration")
        # BoTorchSampler raises ImportError on construction if botorch is
        # missing; skip the whole case in that environment rather than fail.
        try:
            from optuna_integration import BoTorchSampler  # noqa: F401
            from optuna_integration import BoTorchSampler as _Probe

            _Probe()  # validates botorch is importable
        except ImportError:
            pytest.skip("botorch not installed")

    planner = OptunaSearchPlanner(
        _base_config(), _cfg(max_iterations=3, optuna_sampler=sampler_name)
    )
    sampler_cls_name = type(planner._study.sampler).__name__
    assert sampler_cls_name == expected_attr


def test_implicit_botorch_sampler_without_botorch_falls_back_to_tpe(monkeypatch):
    """Schema-default BoTorch falls back because the user did not request it."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "optuna_integration" or name.startswith("optuna_integration."):
            raise ImportError("simulated missing optuna-integration")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    cfg = AdaptiveSearchSweep(
        search_space=[
            SearchSpaceDimension(
                path="phases.profiling.concurrency", lo=1, hi=100, kind="int"
            )
        ],
        objectives=[
            Objective(
                metric="output_token_throughput",
                stat="avg",
                direction=OptimizationDirection.MAXIMIZE,
            )
        ],
        max_iterations=5,
        n_initial_points=2,
        random_seed=42,
        sla_filters=[],
        optuna_acquisition="qlognei",
    )
    with pytest.warns(RuntimeWarning, match="falling back to optuna_sampler='tpe'"):
        planner = OptunaSearchPlanner(_base_config(), cfg)

    assert planner._cfg.optuna_sampler == "tpe"
    assert planner._cfg.optuna_acquisition is None
    assert type(planner._study.sampler).__name__ == "TPESampler"


def test_explicit_botorch_sampler_without_botorch_raises_clear_error(monkeypatch):
    """Explicit BoTorch sampler requests fail instead of silently downgrading."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "optuna_integration" or name.startswith("optuna_integration."):
            raise ImportError("simulated missing optuna-integration")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    cfg = _cfg(optuna_sampler="botorch", optuna_acquisition="qlognei")
    with pytest.raises(ImportError, match=r"BoTorch sampler requires"):
        OptunaSearchPlanner(_base_config(), cfg)


def test_gp_sampler_without_torch_raises_clear_error(monkeypatch):
    """GPSampler branch eagerly raises ImportError when torch is unimportable.

    Regression for: prior to this guard, ``build_sampler`` deferred the
    torch import to Optuna's post-startup GP fit phase, so a user with the
    core Optuna install (no torch) would crash mid-search after
    ``n_initial_points`` random trials. Eager validation surfaces the
    failure at planner construction with an actionable message.

    Simulated by patching ``sys.modules['torch'] = None`` so the import
    fails even when torch is installed in the test environment.
    """
    import sys

    from aiperf.orchestrator.search_planner._optuna_helpers import build_sampler

    monkeypatch.setitem(sys.modules, "torch", None)
    cfg = _cfg(optuna_sampler="gp")
    with pytest.raises(ImportError, match=r"torch"):
        build_sampler(cfg)
    # Re-validate the message names the dep-light alternative.
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match=r"--optuna-sampler tpe"):
        OptunaSearchPlanner(_base_config(), cfg)


# ----------------------------------------------------------------------------
# 10. Constraints: feasibility correctly determined.
# ----------------------------------------------------------------------------


def test_iteration_feasible_when_all_slas_pass():
    planner = OptunaSearchPlanner(_base_config(), _cfg())
    _, v = planner.ask()
    # ttft p95 = 50 < threshold 200 → feasible.
    planner.tell(v, [_make_result(v, throughput=10.0, ttft_p95=50.0)])
    assert planner.history()[0].feasible is True


def test_iteration_infeasible_when_any_sla_violates():
    planner = OptunaSearchPlanner(_base_config(), _cfg())
    _, v = planner.ask()
    # ttft p95 = 500 >= threshold 200 → infeasible.
    planner.tell(v, [_make_result(v, throughput=10.0, ttft_p95=500.0)])
    assert planner.history()[0].feasible is False


# ----------------------------------------------------------------------------
# 11. Constraints: unmeasurable metric.
# ----------------------------------------------------------------------------


def test_iteration_infeasible_when_metric_unmeasurable():
    """SLA references a metric absent from summary_metrics → feasible=False."""
    planner = OptunaSearchPlanner(_base_config(), _cfg())
    _, v = planner.ask()
    # ttft_p95=None → time_to_first_token missing entirely.
    planner.tell(v, [_make_result(v, throughput=10.0, ttft_p95=None)])
    assert planner.history()[0].feasible is False


def test_signed_violation_handles_unmeasurable():
    sla = SLAFilter(metric_tag="ttft", stat="p95", op="lt", threshold=200.0)
    assert _signed_violation(None, sla) == _UNMEASURABLE_VIOLATION
    assert _signed_violation(150.0, sla) == pytest.approx(-50.0)
    assert _signed_violation(250.0, sla) == pytest.approx(50.0)


def test_attr_key_disambiguates_overlapping_filters():
    """Two filters on the same metric_tag with different stat must have distinct keys."""
    a = SLAFilter(metric_tag="ttft", stat="p95", op="lt", threshold=200.0)
    b = SLAFilter(metric_tag="ttft", stat="p99", op="lt", threshold=500.0)
    assert _attr_key(a) != _attr_key(b)


# ----------------------------------------------------------------------------
# 12. Failed iteration (all results failed).
# ----------------------------------------------------------------------------


def test_failed_iteration_records_none_and_continues():
    """All results failed → history records objective_value=None; planner advances."""
    planner = OptunaSearchPlanner(_base_config(), _cfg(max_iterations=5))
    _, v = planner.ask()
    failed = RunResult(label="x", success=False, error="boom")
    planner.tell(v, [failed])
    iteration = planner.history()[0]
    assert iteration.objective_value is None
    assert iteration.feasible is False
    # Planner must remain consistent — next ask must still work.
    next_proposal = planner.ask()
    assert next_proposal is not None


# ----------------------------------------------------------------------------
# 13. Random seed reproducibility.
# ----------------------------------------------------------------------------


def test_random_seed_reproducibility_first_5_proposals():
    """Two planners with identical random_seed → identical first 5 ask() proposals."""
    cfg_a = _cfg(max_iterations=10, random_seed=2026)
    cfg_b = _cfg(max_iterations=10, random_seed=2026)
    planner_a = OptunaSearchPlanner(_base_config(), cfg_a)
    planner_b = OptunaSearchPlanner(_base_config(), cfg_b)
    proposals_a = []
    proposals_b = []
    for _ in range(5):
        _, va = planner_a.ask()
        _, vb = planner_b.ask()
        proposals_a.append(va.values["phases.profiling.concurrency"])
        proposals_b.append(vb.values["phases.profiling.concurrency"])
        # Tell deterministic verdicts so internal state advances identically.
        planner_a.tell(va, [_make_result(va, throughput=10.0, ttft_p95=50.0)])
        planner_b.tell(vb, [_make_result(vb, throughput=10.0, ttft_p95=50.0)])
    assert proposals_a == proposals_b


# ---------------------------------------------------------------------------
# Credential preservation across ask() / _mutate_base
# ---------------------------------------------------------------------------


def _base_config_with_credentials() -> BenchmarkConfig:
    """Base config carrying credential-bearing fields the JSON serializers
    would redact (locks in the credential-preservation regression)."""
    return BenchmarkConfig.model_validate(
        {
            "models": ["m"],
            "endpoint": {
                "urls": ["http://x"],
                "type": "chat",
                "api_key": "sk-real-prod-key",
                "headers": {
                    "Authorization": "Api-Key real-secret-value",
                    "X-Trace-Id": "trace-001",
                },
            },
            "datasets": [{"name": "profiling", "type": "synthetic"}],
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


def test_ask_preserves_credentials() -> None:
    """REGRESSION-LOCK: ``ask()`` previously dumped the base config with
    mode="json", firing the EndpointConfig.api_key + .headers redactors
    and baking ``<redacted>`` into every proposal's config. Same fix as
    ``smooth_isotonic`` and ``monotonic``; locked separately here so a
    single-planner revert is caught.
    """
    planner = OptunaSearchPlanner(_base_config_with_credentials(), _cfg())
    cfg, _ = planner.ask()
    assert cfg.endpoint.api_key == "sk-real-prod-key"
    assert cfg.endpoint.headers["Authorization"] == "Api-Key real-secret-value"
    # Non-sensitive header must round-trip too.
    assert cfg.endpoint.headers["X-Trace-Id"] == "trace-001"


def test_ask_preserves_url_userinfo() -> None:
    """REGRESSION-LOCK (PR #982 dynamo-ops): URL userinfo survives
    ``ask()`` via ``context={"include_secrets": True}``. See
    ``smooth_isotonic`` for the full rationale.
    """
    cfg_dict = _base_config_with_credentials().model_dump(
        mode="python", exclude_none=True, context={"include_secrets": True}
    )
    cfg_dict["endpoint"]["urls"] = [
        "http://alice:s3cret@host1.example.com/v1/chat/completions"
    ]
    base = BenchmarkConfig.model_validate(cfg_dict)
    planner = OptunaSearchPlanner(base, _cfg())
    cfg, _ = planner.ask()
    assert cfg.endpoint.urls == [
        "http://alice:s3cret@host1.example.com/v1/chat/completions"
    ]
