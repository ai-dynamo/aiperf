# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError
from pytest import param

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

    @pytest.mark.parametrize(
        "threshold, match",
        [
            param(0.0, "greater than 0", id="threshold_zero_rejected"),
            param(-0.01, "greater than 0", id="threshold_negative_rejected"),
            param(1.0, "less than 1", id="threshold_one_rejected"),
            param(1.5, "less than 1", id="threshold_above_one_rejected"),
        ],
    )  # fmt: skip
    def test_threshold_out_of_range_rejected(self, threshold, match):
        with pytest.raises(ValidationError, match=match):
            ConvergenceConfig(metric="ttft", threshold=threshold)


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

    def test_num_runs_at_cap_accepted(self):
        cfg = MultiRunConfig(num_runs=10)
        assert cfg.num_runs == 10

    @pytest.mark.parametrize(
        "num_runs, match",
        [
            param(0, "greater than or equal to 1", id="num_runs_zero_rejected"),
            param(-1, "greater than or equal to 1", id="num_runs_negative_rejected"),
            param(11, "less than or equal to 10", id="num_runs_above_cap_rejected"),
        ],
    )  # fmt: skip
    def test_num_runs_out_of_range_rejected(self, num_runs, match):
        with pytest.raises(ValidationError, match=match):
            MultiRunConfig(num_runs=num_runs)


class TestMultiRunCooldownValidation:
    """`cooldown_seconds` is `ge=0, le=86400`. The 24h cap surfaces typos
    like `1e18` at config-load time rather than wedging the orchestrator
    inside `asyncio.sleep`."""

    def test_default_cooldown_is_zero(self):
        assert MultiRunConfig().cooldown_seconds == 0.0

    def test_cooldown_at_cap_accepted(self):
        cfg = MultiRunConfig(cooldown_seconds=86400.0)
        assert cfg.cooldown_seconds == 86400.0

    @pytest.mark.parametrize(
        "cooldown_seconds, match",
        [
            param(-1.0, "greater than or equal to 0", id="negative_cooldown_rejected"),
            param(86401.0, "less than or equal to 86400", id="cooldown_above_cap_rejected"),
        ],
    )  # fmt: skip
    def test_cooldown_out_of_range_rejected(self, cooldown_seconds, match):
        with pytest.raises(ValidationError, match=match):
            MultiRunConfig(cooldown_seconds=cooldown_seconds)


class TestMultiRunConfidenceLevelValidation:
    """`confidence_level` is `gt=0, lt=1`. Pre-fix this had ZERO test
    coverage. A drift to `ge=0, le=1` would silently accept 0.0 (always-
    significant) or 1.0 (degenerate Student's t with infinite CI), both of
    which corrupt downstream stats without error."""

    def test_default_is_0_95(self):
        assert MultiRunConfig().confidence_level == 0.95

    @pytest.mark.parametrize(
        "confidence_level, match",
        [
            param(0.0, "greater than 0", id="zero_rejected"),
            param(1.0, "less than 1", id="one_rejected"),
            param(-0.5, "greater than 0", id="negative_rejected"),
            param(1.5, "less than 1", id="above_one_rejected"),
        ],
    )  # fmt: skip
    def test_confidence_level_out_of_range_rejected(self, confidence_level, match):
        with pytest.raises(ValidationError, match=match):
            MultiRunConfig(confidence_level=confidence_level)

    def test_common_values_accepted(self):
        for value in (0.90, 0.95, 0.99, 0.999):
            cfg = MultiRunConfig(confidence_level=value)
            assert cfg.confidence_level == value


class TestMultiRunBooleanFlagDefaults:
    """Default values are user-visible behavior. Flipping any of these
    silently changes how every multi-run benchmark behaves."""

    @pytest.mark.parametrize(
        "attr, expected",
        [
            param("set_consistent_seed", True, id="set_consistent_seed_default_true"),
            param("vary_seed_per_trial", False, id="vary_seed_per_trial_default_false"),
            param("disable_warmup_after_first", True, id="disable_warmup_after_first_default_true"),
        ],
    )  # fmt: skip
    def test_boolean_flag_default(self, attr, expected):
        assert getattr(MultiRunConfig(), attr) is expected


class TestConvergenceMinRunsBoundary:
    def test_min_runs_equal_to_num_runs_accepted(self):
        """Boundary: `min_runs == num_runs` must pass (cross-field validator
        is `<=`, not `<`)."""
        cfg = MultiRunConfig(
            num_runs=5,
            convergence=ConvergenceConfig(metric="ttft", min_runs=5),
        )
        assert cfg.convergence.min_runs == cfg.num_runs == 5


class TestRepeatTrialFlagsRequireMultipleRuns:
    """v1 parity: across-trial flags (--set-consistent-seed,
    --profile-run-disable-warmup-after-first, --profile-run-cooldown-seconds)
    have no meaning at a single profiling run and must fail loud, not silently
    build a no-op single-run multi_run block. The port routed them but dropped
    v1's num_profile_runs==1 guard.
    """

    @pytest.mark.parametrize(
        "kwargs, flag_fragment",
        [
            ({"set_consistent_seed": True}, "--set-consistent-seed"),
            (
                {"profile_run_disable_warmup_after_first": True},
                "--profile-run-disable-warmup-after-first",
            ),
            ({"profile_run_cooldown_seconds": 5.0}, "--profile-run-cooldown-seconds"),
        ],
    )
    def test_flag_without_multiple_runs_raises(self, kwargs, flag_fragment):
        from aiperf.config.flags._converter_optionals import build_multi_run
        from aiperf.config.flags.cli_config import CLIConfig

        with pytest.raises(ValueError, match=flag_fragment):
            build_multi_run(CLIConfig(model_names=["m"], **kwargs))

    def test_flag_with_multiple_runs_is_accepted(self):
        from aiperf.config.flags._converter_optionals import build_multi_run
        from aiperf.config.flags.cli_config import CLIConfig

        mr = build_multi_run(
            CLIConfig(model_names=["m"], num_profile_runs=5, set_consistent_seed=True)
        )
        assert mr["num_runs"] == 5
        assert mr["set_consistent_seed"] is True
