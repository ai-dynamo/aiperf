# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.kubernetes.sweep_models import (
    AIPerfJobTemplate,
    AIPerfSweepSpec,
    ConvergenceConfig,
    FailurePolicy,
    MultiRunConfig,
)

# Minimal benchmark dict accepted by AIPerfConfig (the type of
# AIPerfJobSpec.benchmark). Tests that focus on axis-combination rules
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
            "requests": 1,
            "concurrency": 1,
        }
    ],
}


def test_multirun_config_defaults_apply():
    cfg = MultiRunConfig(trials=3)
    assert cfg.trials == 3
    assert cfg.cooldown_seconds == 0.0
    assert cfg.auto_set_seed is True
    assert cfg.disable_warmup_after_first is True


def test_multirun_config_accepts_camelcase_alias():
    cfg = MultiRunConfig.model_validate(
        {"trials": 5, "cooldownSeconds": 30, "autoSetSeed": False}
    )
    assert cfg.cooldown_seconds == 30
    assert cfg.auto_set_seed is False


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
        param({"sweep": {"type": "grid", "variables": {"x": [1, 2]}},
               "template": {"spec": {"benchmark": _VALID_BENCHMARK}}},
              id="sweep-only"),
        param({"multiRun": {"trials": 3},
               "template": {"spec": {"benchmark": _VALID_BENCHMARK}}},
              id="multirun-only"),
        param({"multiRun": {"cooldownSeconds": 5},
               "convergence": {"metric": "ttft_p99"},
               "template": {"spec": {"benchmark": _VALID_BENCHMARK}}},
              id="convergence-needs-multirun"),
    ],
)  # fmt: skip
def test_aiperfsweep_spec_validates(data):
    AIPerfSweepSpec.model_validate(data)


def test_aiperfsweep_rejects_empty_axes():
    with pytest.raises(ValidationError, match="at least one of"):
        AIPerfSweepSpec.model_validate(
            {"template": {"spec": {"benchmark": _VALID_BENCHMARK}}}
        )


def test_aiperfsweep_rejects_convergence_without_multirun():
    with pytest.raises(ValidationError, match="requires `multiRun`"):
        AIPerfSweepSpec.model_validate(
            {
                "convergence": {"metric": "ttft_p99"},
                "template": {"spec": {"benchmark": _VALID_BENCHMARK}},
            }
        )


def test_aiperfsweep_rejects_convergence_with_explicit_trials():
    with pytest.raises(ValidationError, match="`multiRun.trials` must be unset"):
        AIPerfSweepSpec.model_validate(
            {
                "multiRun": {"trials": 5},
                "convergence": {"metric": "ttft_p99"},
                "template": {"spec": {"benchmark": _VALID_BENCHMARK}},
            }
        )


def test_aiperfsweep_rejects_sweep_in_template_benchmark():
    with pytest.raises(ValidationError, match="not permitted"):
        AIPerfSweepSpec.model_validate(
            {
                "multiRun": {"trials": 3},
                "template": {
                    "spec": {
                        "benchmark": {
                            **_VALID_BENCHMARK,
                            "sweep": {
                                "type": "grid",
                                "variables": {"x": [1, 2]},
                            },
                        }
                    }
                },
            }
        )


def test_aiperfsweep_accepts_sweep_and_convergence_composing():
    spec = AIPerfSweepSpec.model_validate(
        {
            "sweep": {"type": "grid", "variables": {"phases.x.concurrency": [8, 32]}},
            "multiRun": {"cooldownSeconds": 10},
            "convergence": {"metric": "ttft_p99"},
            "template": {"spec": {"benchmark": _VALID_BENCHMARK}},
        }
    )
    assert spec.sweep is not None
    assert spec.convergence is not None


# =============================================================================
# Adversarial regression-locks for the just-landed bug-fixes.
# =============================================================================


def test_convergence_config_min_runs_gt_max_runs_raises():
    """``min_runs`` must be <= ``max_runs``; otherwise convergence can never run."""
    with pytest.raises(ValidationError, match=r"must be <="):
        ConvergenceConfig(metric="ttft_p99", min_runs=10, max_runs=5)


def test_convergence_config_min_runs_eq_max_runs_accepted():
    """Equal min and max is fine — exactly one convergence-check happens."""
    cfg = ConvergenceConfig(metric="ttft_p99", min_runs=5, max_runs=5)
    assert cfg.min_runs == cfg.max_runs == 5


@pytest.mark.parametrize(
    "max_runs, ok",
    [
        param(2, True, id="max-runs-min-bound-accepted"),
        param(20, True, id="max-runs-upper-bound-accepted"),
        param(21, False, id="max-runs-21-rejected"),
        param(50, False, id="max-runs-50-rejected"),
        param(1, False, id="max-runs-below-min-rejected"),
    ],
)  # fmt: skip
def test_convergence_config_max_runs_bounds(max_runs: int, ok: bool) -> None:
    """``ConvergenceConfig.max_runs`` must be ``2 <= n <= 20``.

    Regression-lock: previously had only ``ge=2`` (no upper bound). A
    sweep with ``maxRuns=21`` would pass apiserver/Pydantic validation,
    flow through ``build_plan_from_sweep`` to ``BenchmarkPlan(trials=21)``,
    and crash the sweep-controller pod with a Pydantic ValidationError
    (``BenchmarkPlan.trials`` is bounded ``le=20``). The three bounds —
    CRD ``maxRuns.maximum=20``, ``ConvergenceConfig.max_runs.le=20``,
    and ``BenchmarkPlan.trials.le=20`` — must stay aligned.
    """
    if ok:
        cfg = ConvergenceConfig(metric="ttft_p99", min_runs=2, max_runs=max_runs)
        assert cfg.max_runs == max_runs
    else:
        with pytest.raises(ValidationError):
            ConvergenceConfig(metric="ttft_p99", min_runs=2, max_runs=max_runs)


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
        "multiRun": {"trials": 3},
        "ttlSecondsAfterFinished": ttl,
        "template": {"spec": {"benchmark": _VALID_BENCHMARK}},
    }
    if ok:
        spec = AIPerfSweepSpec.model_validate(data)
        assert spec.ttl_seconds_after_finished == ttl
    else:
        with pytest.raises(ValidationError):
            AIPerfSweepSpec.model_validate(data)


@pytest.mark.parametrize(
    "forbidden_key",
    [
        param("sweep", id="sweep-snake"),
        param("multi_run", id="multirun-snake"),
        param("multiRun", id="multirun-camel"),
        param("convergence", id="convergence"),
    ],
)  # fmt: skip
def test_aiperfsweep_rejects_forbidden_key_under_template_spec(
    forbidden_key: str,
) -> None:
    """Rule-4 broadening: each axis-key under ``template.spec`` is rejected.

    Previously rule 4 only scanned ``template.spec.benchmark``; the
    just-landed fix extends it to scan ``template.spec`` directly so users
    don't accidentally stamp sweep-axis keys onto every child.
    """
    data = {
        "multiRun": {"trials": 3},
        "template": {
            "spec": {
                "benchmark": _VALID_BENCHMARK,
                forbidden_key: {"trials": 1}
                if forbidden_key in ("multi_run", "multiRun")
                else {"metric": "ttft_p99"}
                if forbidden_key == "convergence"
                else {"type": "grid", "variables": {"x": [1, 2]}},
            }
        },
    }
    with pytest.raises(ValidationError, match=rf"template\.spec\.{forbidden_key}"):
        AIPerfSweepSpec.model_validate(data)


def test_aiperfsweep_rejects_convergence_under_template_spec_benchmark() -> None:
    """Rule-4 broadening covers convergence under ``template.spec.benchmark``."""
    with pytest.raises(
        ValidationError, match=r"template\.spec\.benchmark\.convergence"
    ):
        AIPerfSweepSpec.model_validate(
            {
                "multiRun": {"cooldownSeconds": 5},
                "template": {
                    "spec": {
                        "benchmark": {"convergence": {"metric": "ttft_p99"}},
                    }
                },
            }
        )


def test_aiperfsweep_rejects_multirun_camel_under_template_spec_benchmark() -> None:
    """``template.spec.benchmark.multiRun`` (camelCase) is rejected — the rule
    must scan both snake_case and camelCase forms."""
    with pytest.raises(ValidationError, match=r"template\.spec\.benchmark\.multiRun"):
        AIPerfSweepSpec.model_validate(
            {
                "sweep": {"type": "grid", "variables": {"x": [1, 2]}},
                "template": {
                    "spec": {
                        "benchmark": {"multiRun": {"trials": 3}},
                    }
                },
            }
        )


# ---------------------------------------------------------------------------
# template.spec is now typed as AIPerfJobSpec — invalid benchmarks and
# wrong-typed deployment fields surface at submit time via Pydantic field
# validation rather than the previous lazy from_crd_spec round-trip.
# ---------------------------------------------------------------------------


def test_aiperfsweep_empty_benchmark_rejected_for_missing_endpoint() -> None:
    """Empty benchmark (no endpoint) must fail AIPerfConfig validation with
    the missing-field surfaced — the user needs to know what's wrong."""
    with pytest.raises(ValidationError, match=r"endpoint"):
        AIPerfSweepSpec.model_validate(
            {
                "multiRun": {"trials": 3},
                "template": {"spec": {"benchmark": {}}},
            }
        )


def test_aiperfsweep_wrong_type_image_rejected() -> None:
    """``image`` typed as int instead of str must fail with a message
    naming the field."""
    with pytest.raises(ValidationError, match=r"(?i)image"):
        AIPerfSweepSpec.model_validate(
            {
                "multiRun": {"trials": 3},
                "template": {
                    "spec": {
                        "image": 12345,
                        "benchmark": _VALID_BENCHMARK,
                    }
                },
            }
        )


def test_aiperfsweep_valid_endpoint_passes() -> None:
    """Regression-lock so future refactors don't accidentally make
    template.spec validation a no-op: a valid template.spec must still
    validate successfully."""
    spec = AIPerfSweepSpec.model_validate(
        {
            "multiRun": {"trials": 2},
            "template": {
                "spec": {
                    "image": "x:latest",
                    "benchmark": _VALID_BENCHMARK,
                }
            },
        }
    )
    assert spec.multi_run is not None
    assert spec.multi_run.trials == 2


# ---------------------------------------------------------------------------
# AIPerfJobTemplate — typed metadata (ObjectMetaPartial) and typed
# spec (AIPerfJobSpec). Adversarial regression-locks for the Task-2 refactor.
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_template_spec() -> dict:
    """Minimal template.spec dict that validates as AIPerfJobSpec."""
    return {"benchmark": _VALID_BENCHMARK}


def test_aiperf_job_template_metadata_rejects_unknown_keys(valid_template_spec):
    """ObjectMetaPartial rejects fields outside labels/annotations."""
    with pytest.raises(ValidationError, match=r"extra|name"):
        AIPerfJobTemplate.model_validate(
            {
                "metadata": {"name": "should-not-be-here"},
                "spec": valid_template_spec,
            }
        )


def test_aiperf_job_template_metadata_typed_labels_and_annotations(valid_template_spec):
    """Labels and annotations both validate as dict[str, str]."""
    template = AIPerfJobTemplate.model_validate(
        {
            "metadata": {
                "labels": {"team": "perf"},
                "annotations": {"note": "rampA"},
            },
            "spec": valid_template_spec,
        }
    )
    assert template.metadata.labels == {"team": "perf"}
    assert template.metadata.annotations == {"note": "rampA"}


def test_aiperf_job_template_spec_is_typed_aiperf_job_spec(valid_template_spec):
    """template.spec is parsed as AIPerfJobSpec, not a raw dict."""
    template = AIPerfJobTemplate.model_validate({"spec": valid_template_spec})
    from aiperf.operator.models import AIPerfJobSpec

    assert isinstance(template.spec, AIPerfJobSpec)
    assert template.spec.skip_endpoint_check is False
