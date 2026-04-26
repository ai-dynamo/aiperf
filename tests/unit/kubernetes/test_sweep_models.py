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
