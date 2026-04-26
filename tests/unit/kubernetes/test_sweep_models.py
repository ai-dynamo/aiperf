# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.kubernetes.sweep_models import (
    AIPerfSweepSpec,
    ConvergenceConfig,
    FailurePolicy,
    MultiRunConfig,
)


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
               "template": {"spec": {"benchmark": {}}}},
              id="sweep-only"),
        param({"multiRun": {"trials": 3},
               "template": {"spec": {"benchmark": {}}}},
              id="multirun-only"),
        param({"multiRun": {"cooldownSeconds": 5},
               "convergence": {"metric": "ttft_p99"},
               "template": {"spec": {"benchmark": {}}}},
              id="convergence-needs-multirun"),
    ],
)  # fmt: skip
def test_aiperfsweep_spec_validates(data):
    AIPerfSweepSpec.model_validate(data)


def test_aiperfsweep_rejects_empty_axes():
    with pytest.raises(ValidationError, match="at least one of"):
        AIPerfSweepSpec.model_validate({"template": {"spec": {"benchmark": {}}}})


def test_aiperfsweep_rejects_convergence_without_multirun():
    with pytest.raises(ValidationError, match="requires `multiRun`"):
        AIPerfSweepSpec.model_validate(
            {
                "convergence": {"metric": "ttft_p99"},
                "template": {"spec": {"benchmark": {}}},
            }
        )


def test_aiperfsweep_rejects_convergence_with_explicit_trials():
    with pytest.raises(ValidationError, match="`multiRun.trials` must be unset"):
        AIPerfSweepSpec.model_validate(
            {
                "multiRun": {"trials": 5},
                "convergence": {"metric": "ttft_p99"},
                "template": {"spec": {"benchmark": {}}},
            }
        )


def test_aiperfsweep_rejects_sweep_in_template_benchmark():
    with pytest.raises(ValidationError, match="not permitted"):
        AIPerfSweepSpec.model_validate(
            {
                "multiRun": {"trials": 3},
                "template": {"spec": {"benchmark": {"sweep": {"type": "grid"}}}},
            }
        )


def test_aiperfsweep_accepts_sweep_and_convergence_composing():
    spec = AIPerfSweepSpec.model_validate(
        {
            "sweep": {"type": "grid", "variables": {"phases.x.concurrency": [8, 32]}},
            "multiRun": {"cooldownSeconds": 10},
            "convergence": {"metric": "ttft_p99"},
            "template": {"spec": {"benchmark": {}}},
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
        "template": {"spec": {"benchmark": {}}},
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
                "benchmark": {},
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
