# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from aiperf.config.sweep.multi_run import ConvergenceConfig, MultiRunConfig


def test_multi_run_defaults_no_convergence():
    cfg = MultiRunConfig()
    assert cfg.num_runs == 1
    assert cfg.convergence is None
    assert cfg.cooldown_seconds == 0.0


def test_multi_run_with_convergence_nested():
    cfg = MultiRunConfig(
        num_runs=10,
        convergence=ConvergenceConfig(metric="ttft", threshold=0.05, min_runs=3),
    )
    assert cfg.convergence is not None
    assert cfg.convergence.metric == "ttft"
    assert cfg.convergence.min_runs == 3


def test_convergence_min_runs_exceeds_num_runs_raises():
    with pytest.raises(ValidationError, match="must be <= num_runs"):
        MultiRunConfig(
            num_runs=3,
            convergence=ConvergenceConfig(metric="ttft", min_runs=5),
        )


def test_multi_run_rejects_old_flat_convergence_fields():
    with pytest.raises(ValidationError, match=r"convergence_metric"):
        MultiRunConfig(convergence_metric="ttft")


def test_multi_run_rejects_parameter_sweep_fields():
    with pytest.raises(ValidationError, match=r"parameter_sweep_cooldown_seconds"):
        MultiRunConfig(parameter_sweep_cooldown_seconds=10.0)


class TestConvergenceThresholdValidation:
    """`ConvergenceConfig.threshold` is `float | None`, default None.

    None means "use the criterion class's algorithm-specific default."
    When set, Pydantic must still enforce the (0, 1) open interval — a
    threshold of 0 collapses the convergence test to never-fire, and a
    threshold >= 1 makes it always-fire (for the dispersion-style modes)
    or fully degenerate (for the KS-p-value mode where 1 is the max).
    """

    def test_default_threshold_is_none(self):
        cfg = ConvergenceConfig(metric="ttft")
        assert cfg.threshold is None

    def test_explicit_threshold_in_range_accepted(self):
        cfg = ConvergenceConfig(metric="ttft", threshold=0.5)
        assert cfg.threshold == 0.5

    def test_threshold_zero_rejected(self):
        with pytest.raises(ValidationError, match="greater than 0"):
            ConvergenceConfig(metric="ttft", threshold=0.0)

    def test_threshold_negative_rejected(self):
        with pytest.raises(ValidationError, match="greater than 0"):
            ConvergenceConfig(metric="ttft", threshold=-0.01)

    def test_threshold_one_rejected(self):
        with pytest.raises(ValidationError, match="less than 1"):
            ConvergenceConfig(metric="ttft", threshold=1.0)

    def test_threshold_above_one_rejected(self):
        with pytest.raises(ValidationError, match="less than 1"):
            ConvergenceConfig(metric="ttft", threshold=1.5)


class TestConvergenceMinRunsValidation:
    def test_default_min_runs_is_two(self):
        cfg = ConvergenceConfig(metric="ttft")
        assert cfg.min_runs == 2

    def test_min_runs_below_two_rejected(self):
        with pytest.raises(ValidationError, match="greater than or equal to 2"):
            ConvergenceConfig(metric="ttft", min_runs=1)


class TestMultiRunNumRunsValidation:
    """`num_runs` is `ge=1, le=10`. Drift on either bound silently allows
    pathological trial counts (zero = empty execution, thousands = wedge)."""

    def test_default_num_runs_is_one(self):
        cfg = MultiRunConfig()
        assert cfg.num_runs == 1

    def test_num_runs_zero_rejected(self):
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            MultiRunConfig(num_runs=0)

    def test_num_runs_negative_rejected(self):
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            MultiRunConfig(num_runs=-1)

    def test_num_runs_at_cap_accepted(self):
        cfg = MultiRunConfig(num_runs=10)
        assert cfg.num_runs == 10

    def test_num_runs_above_cap_rejected(self):
        with pytest.raises(ValidationError, match="less than or equal to 10"):
            MultiRunConfig(num_runs=11)


class TestMultiRunCooldownValidation:
    """`cooldown_seconds` is `ge=0, le=86400`. The 24h cap surfaces typos
    like `1e18` at config-load time rather than wedging the orchestrator
    inside `asyncio.sleep`."""

    def test_default_cooldown_is_zero(self):
        assert MultiRunConfig().cooldown_seconds == 0.0

    def test_negative_cooldown_rejected(self):
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            MultiRunConfig(cooldown_seconds=-1.0)

    def test_cooldown_at_cap_accepted(self):
        cfg = MultiRunConfig(cooldown_seconds=86400.0)
        assert cfg.cooldown_seconds == 86400.0

    def test_cooldown_above_cap_rejected(self):
        with pytest.raises(ValidationError, match="less than or equal to 86400"):
            MultiRunConfig(cooldown_seconds=86401.0)


class TestMultiRunConfidenceLevelValidation:
    """`confidence_level` is `gt=0, lt=1`. Pre-fix this had ZERO test
    coverage. A drift to `ge=0, le=1` would silently accept 0.0 (always-
    significant) or 1.0 (degenerate Student's t with infinite CI), both of
    which corrupt downstream stats without error."""

    def test_default_is_0_95(self):
        assert MultiRunConfig().confidence_level == 0.95

    def test_zero_rejected(self):
        with pytest.raises(ValidationError, match="greater than 0"):
            MultiRunConfig(confidence_level=0.0)

    def test_one_rejected(self):
        with pytest.raises(ValidationError, match="less than 1"):
            MultiRunConfig(confidence_level=1.0)

    def test_negative_rejected(self):
        with pytest.raises(ValidationError, match="greater than 0"):
            MultiRunConfig(confidence_level=-0.5)

    def test_above_one_rejected(self):
        with pytest.raises(ValidationError, match="less than 1"):
            MultiRunConfig(confidence_level=1.5)

    def test_common_values_accepted(self):
        for value in (0.90, 0.95, 0.99, 0.999):
            cfg = MultiRunConfig(confidence_level=value)
            assert cfg.confidence_level == value


class TestMultiRunBooleanFlagDefaults:
    """Default values are user-visible behavior. Flipping any of these
    silently changes how every multi-run benchmark behaves."""

    def test_set_consistent_seed_default_true(self):
        assert MultiRunConfig().set_consistent_seed is True

    def test_vary_seed_per_trial_default_false(self):
        assert MultiRunConfig().vary_seed_per_trial is False

    def test_disable_warmup_after_first_default_true(self):
        assert MultiRunConfig().disable_warmup_after_first is True


class TestConvergenceMinRunsBoundary:
    def test_min_runs_equal_to_num_runs_accepted(self):
        """Boundary: `min_runs == num_runs` must pass (cross-field validator
        is `<=`, not `<`)."""
        cfg = MultiRunConfig(
            num_runs=5,
            convergence=ConvergenceConfig(metric="ttft", min_runs=5),
        )
        assert cfg.convergence.min_runs == cfg.num_runs == 5
