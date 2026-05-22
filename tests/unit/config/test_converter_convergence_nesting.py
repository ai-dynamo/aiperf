# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for convergence-field nesting in build_multi_run.

Schema-2.0 (commit 21b58afcf) nested convergence fields under
``MultiRunConfig.convergence`` (a ConvergenceConfig sub-object) instead of
carrying them flat on MultiRunConfig. The v1 -> v2 converter at
``aiperf.config.flags._converter_optionals.build_multi_run`` was missed in
that refactor and continued to emit flat ``convergence_metric`` /
``convergence_mode`` / ``convergence_threshold`` / ``convergence_stat``
keys, which MultiRunConfig (extra='forbid') then rejected at config-load
time. This file locks the nesting in.
"""

from __future__ import annotations

from typing import Any

from aiperf.common.enums import ConvergenceStat
from aiperf.config.flags._converter_optionals import build_multi_run
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.plugin.enums import ConvergenceCriterionType


def _make_user(**sweeping_fields: Any) -> CLIConfig:
    endpoint = CLIConfig(url="http://localhost:8000/test", model_names=["test-model"])
    return CLIConfig(**endpoint.model_dump(exclude_unset=True), **sweeping_fields)


class TestConvergenceNesting:
    """build_multi_run must emit convergence fields nested under 'convergence'."""

    def test_convergence_metric_emits_nested_subdict(self):
        out = build_multi_run(
            _make_user(
                num_profile_runs=5,
                convergence_metric="time_to_first_token",
            )
        )
        assert out is not None
        assert "convergence" in out
        assert out["convergence"]["metric"] == "time_to_first_token"

    def test_no_flat_convergence_keys_emitted(self):
        out = build_multi_run(
            _make_user(
                num_profile_runs=5,
                convergence_metric="time_to_first_token",
                convergence_mode=ConvergenceCriterionType.CV,
                convergence_threshold=0.05,
                convergence_stat=ConvergenceStat.P99,
            )
        )
        assert out is not None
        for forbidden in (
            "convergence_metric",
            "convergence_mode",
            "convergence_threshold",
            "convergence_stat",
        ):
            assert forbidden not in out, (
                f"flat key {forbidden!r} must not appear on multi_run; "
                "Schema-2.0 nests convergence fields under 'convergence'"
            )

    def test_all_convergence_fields_routed_under_nested_block(self):
        out = build_multi_run(
            _make_user(
                num_profile_runs=5,
                convergence_metric="time_to_first_token",
                convergence_mode=ConvergenceCriterionType.CV,
                convergence_threshold=0.05,
                convergence_stat=ConvergenceStat.P99,
            )
        )
        assert out is not None
        conv = out["convergence"]
        assert conv["metric"] == "time_to_first_token"
        assert conv["mode"] == ConvergenceCriterionType.CV
        assert conv["threshold"] == 0.05
        assert conv["stat"] == ConvergenceStat.P99

    def test_no_convergence_block_when_metric_unset(self):
        out = build_multi_run(_make_user(num_profile_runs=5))
        # Either out is None (no fields set) or convergence is absent.
        if out is not None:
            assert "convergence" not in out

    def test_other_convergence_flags_alone_do_not_emit_block(self):
        # convergence_mode/threshold/stat without convergence_metric is a
        # no-op: ConvergenceConfig requires `metric`, and v1's
        # convergence_metric=None means "adaptive disabled".
        out = build_multi_run(
            _make_user(
                num_profile_runs=5,
                convergence_threshold=0.05,
            )
        )
        if out is not None:
            assert "convergence" not in out

    def test_multi_run_config_accepts_converter_output(self):
        """End-to-end: converter output must pass MultiRunConfig validation."""
        from aiperf.config.sweep.multi_run import MultiRunConfig

        out = build_multi_run(
            _make_user(
                num_profile_runs=5,
                convergence_metric="time_to_first_token",
                convergence_mode=ConvergenceCriterionType.CI_WIDTH,
                convergence_threshold=0.20,
            )
        )
        assert out is not None
        # Round-trip through MultiRunConfig (extra='forbid'). Pre-fix this
        # raised: "convergence_metric: Extra inputs are not permitted".
        config = MultiRunConfig.model_validate(out)
        assert config.convergence is not None
        assert config.convergence.metric == "time_to_first_token"
        assert config.convergence.mode == ConvergenceCriterionType.CI_WIDTH
        assert config.convergence.threshold == 0.20


class TestThresholdNoneFallback:
    """When the user does not pass --convergence-threshold, the v2 config gets None.

    Pins the contract that lets each criterion fall through to its own
    algorithm-specific default. Pre-fix, the v1 flag defaulted to 0.10 and
    every criterion received that value regardless of mode.
    """

    def test_unset_v1_threshold_omitted_from_converter_output(self):
        """Converter only emits keys present in `model_fields_set`."""
        out = build_multi_run(
            _make_user(
                num_profile_runs=5,
                convergence_metric="time_to_first_token",
            )
        )
        assert out is not None
        assert "convergence" in out
        assert "threshold" not in out["convergence"], (
            "Unset --convergence-threshold must not synthesize a threshold "
            "key; otherwise per-mode class defaults can never apply."
        )

    def test_unset_v1_threshold_yields_none_in_multi_run_config(self):
        """End-to-end: omitting v1 flag means v2 ConvergenceConfig.threshold is None."""
        from aiperf.config.sweep.multi_run import MultiRunConfig

        out = build_multi_run(
            _make_user(
                num_profile_runs=5,
                convergence_metric="time_to_first_token",
            )
        )
        assert out is not None
        config = MultiRunConfig.model_validate(out)
        assert config.convergence is not None
        assert config.convergence.threshold is None

    def test_explicit_v1_threshold_preserved_through_converter(self):
        """When the user does pass the flag, it must reach the v2 config verbatim."""
        from aiperf.config.sweep.multi_run import MultiRunConfig

        out = build_multi_run(
            _make_user(
                num_profile_runs=5,
                convergence_metric="time_to_first_token",
                convergence_threshold=0.07,
            )
        )
        assert out is not None
        config = MultiRunConfig.model_validate(out)
        assert config.convergence is not None
        assert config.convergence.threshold == 0.07
