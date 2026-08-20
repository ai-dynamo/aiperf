# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.config.resolution.plan import FailurePolicy
from aiperf.config.sweep.multi_run import ConvergenceConfig, MultiRunConfig
from aiperf.kubernetes.crd_models import AIPerfSweepSpec

# Minimal benchmark dict accepted by AIPerfConfig (the type of
# AIPerfWorkloadSpec.benchmark). Tests that focus on axis-combination rules
# don't need a real endpoint, but the typed validator does — so we
# provide the smallest one that round-trips.
_VALID_BENCHMARK = {
    "models": ["test-model"],
    "endpoint": {"url": "http://x"},
    "datasets": [
        {
            "name": "default",
            "type": "synthetic",
            "entries": 1,
            "prompts": {"isl": 8, "osl": 8},
        }
    ],
    "phases": [
        {
            "name": "default",
            "type": "concurrency",
            "kind": "profiling",
            "requests": 1,
            "concurrency": 1,
        }
    ],
}

_VALID_SWEEP = {
    "type": "grid",
    "parameters": {"phases.default.concurrency": [1, 2]},
}


def test_multirun_config_defaults_apply():
    cfg = MultiRunConfig(num_runs=3)
    assert cfg.num_runs == 3
    assert cfg.cooldown_seconds == 0.0
    assert cfg.set_consistent_seed is True
    assert cfg.disable_warmup_after_first is True


def test_multirun_config_accepts_camelcase_alias():
    cfg = MultiRunConfig.model_validate(
        {"numRuns": 5, "cooldownSeconds": 30, "setConsistentSeed": False}
    )
    assert cfg.num_runs == 5
    assert cfg.cooldown_seconds == 30
    assert cfg.set_consistent_seed is False


def test_convergence_config_re_export_is_canonical():
    """`aiperf.kubernetes.sweep_models.ConvergenceConfig` is a re-export of
    the canonical class in `aiperf.config.multi_run`, not a separate class."""
    from aiperf.config.sweep.multi_run import ConvergenceConfig as Canonical

    assert ConvergenceConfig is Canonical


def test_convergence_config_validates_threshold_range():
    with pytest.raises(ValidationError):
        ConvergenceConfig(metric="ttft_p99", threshold=1.0)
    with pytest.raises(ValidationError):
        ConvergenceConfig(metric="ttft_p99", threshold=0.0)


def test_failure_policy_default_continues():
    fp = FailurePolicy()
    assert fp.on_child_failure == "continue"
    assert fp.max_failures == 0


@pytest.mark.parametrize(
    "data",
    [
        param(
            {"sweep": _VALID_SWEEP, "benchmark": _VALID_BENCHMARK},
            id="sweep-only",
        ),
        param(
            {
                "sweep": _VALID_SWEEP,
                "multiRun": {"numRuns": 3},
                "benchmark": _VALID_BENCHMARK,
            },
            id="sweep-with-multirun",
        ),
        param(
            {
                "sweep": _VALID_SWEEP,
                "multiRun": {"cooldownSeconds": 5},
                "benchmark": _VALID_BENCHMARK,
            },
            id="sweep-with-multirun-cooldown",
        ),
    ],
)  # fmt: skip
def test_aiperfsweep_spec_validates(data):
    AIPerfSweepSpec.model_validate(data)


def test_aiperfsweep_rejects_missing_sweep():
    """AIPerfSweep.spec.sweep is required by the kind-specific validator."""
    with pytest.raises(ValidationError, match="sweep is required"):
        AIPerfSweepSpec.model_validate({"benchmark": _VALID_BENCHMARK})


# =============================================================================
# Convergence bounds (regression-locks).
# =============================================================================


def test_convergence_config_min_runs_below_two_rejected():
    """``min_runs`` must be >= 2; convergence needs at least two samples to compute a stat."""
    with pytest.raises(ValidationError):
        ConvergenceConfig(metric="ttft_p99", min_runs=1)


@pytest.mark.parametrize(
    "ttl, ok",
    [
        param(-1, False, id="ttl-minus-one-rejected"),
        param(-100, False, id="ttl-large-negative-rejected"),
        param(0, True, id="ttl-zero-accepted"),
        param(1, True, id="ttl-one-accepted"),
        param(3600, True, id="ttl-one-hour-accepted"),
    ],
)  # fmt: skip
def test_aiperfsweep_spec_ttl_bounds(ttl: int, ok: bool) -> None:
    """``ttlSecondsAfterFinished`` is ``ge=0`` — negatives are rejected."""
    data = {
        "sweep": _VALID_SWEEP,
        "multiRun": {"numRuns": 3},
        "ttlSecondsAfterFinished": ttl,
        "benchmark": _VALID_BENCHMARK,
    }
    if ok:
        spec = AIPerfSweepSpec.model_validate(data)
        assert spec.ttl_seconds_after_finished == ttl
    else:
        with pytest.raises(ValidationError):
            AIPerfSweepSpec.model_validate(data)


# ---------------------------------------------------------------------------
# AIPerfSweepSpec is a flat envelope. Invalid benchmarks and
# wrong-typed deployment fields surface at submit time via Pydantic.
# ---------------------------------------------------------------------------


def test_aiperfsweep_empty_benchmark_rejected_for_missing_endpoint() -> None:
    """Empty benchmark (no endpoint) must fail AIPerfConfig validation with
    the missing-field surfaced — the user needs to know what's wrong."""
    with pytest.raises(ValidationError, match=r"endpoint"):
        AIPerfSweepSpec.model_validate(
            {
                "sweep": _VALID_SWEEP,
                "multiRun": {"numRuns": 3},
                "benchmark": {},
            }
        )


def test_aiperfsweep_wrong_type_image_rejected() -> None:
    """``image`` typed as int instead of str must fail with a message
    naming the field."""
    with pytest.raises(ValidationError, match=r"(?i)image"):
        AIPerfSweepSpec.model_validate(
            {
                "sweep": _VALID_SWEEP,
                "multiRun": {"numRuns": 3},
                "image": 12345,
                "benchmark": _VALID_BENCHMARK,
            }
        )


def test_aiperfsweep_valid_envelope_passes() -> None:
    """Regression-lock so future refactors don't accidentally make
    spec validation a no-op: a valid flat envelope must validate."""
    spec = AIPerfSweepSpec.model_validate(
        {
            "sweep": _VALID_SWEEP,
            "multiRun": {"numRuns": 2},
            "image": "x:latest",
            "benchmark": _VALID_BENCHMARK,
        }
    )
    assert spec.multi_run is not None
    assert spec.multi_run.num_runs == 2


# =============================================================================
# Cluster-side adaptive search (Bayesian Optimization under AIPerfSweep).
# Now lives on `spec.sweep` (AdaptiveSearchSweep variant), not
# `spec.multi_run.adaptive_search`.
# =============================================================================


def test_aiperfsweep_accepts_adaptive_search_sweep_variant():
    """A ``sweep:`` block of ``type: adaptive_search`` validates against the
    AIPerfSweepSpec without an extra ``multi_run`` block."""
    from aiperf.config.sweep import AdaptiveSearchSweep

    spec = AIPerfSweepSpec.model_validate(
        {
            "sweep": {
                "type": "adaptive_search",
                "search_space": [
                    {
                        "path": "phases.profiling.concurrency",
                        "lo": 1,
                        "hi": 1000,
                        "kind": "int",
                    }
                ],
                "objectives": [
                    {
                        "metric": "output_token_throughput",
                        "stat": "avg",
                        "direction": "maximize",
                    }
                ],
                "max_iterations": 10,
            },
            "image": "x:latest",
            "benchmark": _VALID_BENCHMARK,
        }
    )
    assert isinstance(spec.sweep, AdaptiveSearchSweep)
    assert spec.sweep.max_iterations == 10
    assert spec.sweep.objectives[0].metric == "output_token_throughput"


def test_multi_run_no_longer_carries_adaptive_search():
    """Adaptive search lives on ``sweep`` now; passing it under multi_run
    is rejected as an unknown field."""
    with pytest.raises(ValidationError):
        MultiRunConfig.model_validate(
            {"num_runs": 3, "adaptive_search": {"max_iterations": 10}}
        )
