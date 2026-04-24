# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for config validators with no existing coverage.

Focuses on:
- Seamless-not-on-first-phase constraint
- Stop condition requirements per phase type
- Weighted model strategy weight validation
- User-centric constraint enforcement (sessions >= users, requests >= users)
- Dataset reference validation across phases
- DurationSpec parsing (numbers, strings, units)
- Grace period requires duration constraint
- validate_config_file warning generation
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aiperf.common.enums import ModelSelectionStrategy
from aiperf.config.config import BenchmarkConfig
from aiperf.config.loader import validate_config_file
from aiperf.config.models import ModelsAdvanced
from aiperf.config.phases import (
    _parse_duration,
)

# ============================================================
# Helpers
# ============================================================


def _base_config(**overrides: object) -> dict:
    """Minimal valid BenchmarkConfig dict with overrides."""
    base: dict = {
        "models": ["test-model"],
        "endpoint": {
            "urls": ["http://localhost:8000/v1/chat/completions"],
        },
        "datasets": {
            "main": {
                "type": "synthetic",
                "entries": 100,
                "prompts": {"isl": 128, "osl": 64},
            },
        },
        "phases": {
            "profiling": {
                "type": "concurrency",
                "concurrency": 8,
                "requests": 100,
            },
        },
    }
    base.update(overrides)
    return base


# ============================================================
# Class 1: TestSeamlessNotOnFirstPhase
# ============================================================


class TestSeamlessNotOnFirstPhase:
    """Verify seamless=True is rejected on the first phase."""

    def test_seamless_on_second_phase_passes(self) -> None:
        cfg = BenchmarkConfig(
            **_base_config(
                phases={
                    "warmup": {
                        "type": "concurrency",
                        "concurrency": 4,
                        "requests": 50,
                        "seamless": False,
                        "exclude_from_results": True,
                    },
                    "profiling": {
                        "type": "concurrency",
                        "concurrency": 8,
                        "requests": 100,
                        "seamless": True,
                    },
                },
            )
        )
        assert cfg.phases["profiling"].seamless is True

    def test_seamless_on_first_phase_raises(self) -> None:
        with pytest.raises(ValidationError, match="seamless"):
            BenchmarkConfig(
                **_base_config(
                    phases={
                        "warmup": {
                            "type": "concurrency",
                            "concurrency": 4,
                            "requests": 50,
                            "seamless": True,
                        },
                        "profiling": {
                            "type": "concurrency",
                            "concurrency": 8,
                            "requests": 100,
                        },
                    },
                )
            )

    def test_no_seamless_anywhere_passes(self) -> None:
        cfg = BenchmarkConfig(
            **_base_config(
                phases={
                    "warmup": {
                        "type": "concurrency",
                        "concurrency": 4,
                        "requests": 50,
                        "exclude_from_results": True,
                    },
                    "profiling": {
                        "type": "concurrency",
                        "concurrency": 8,
                        "requests": 100,
                    },
                },
            )
        )
        assert cfg.phases["warmup"].seamless is False
        assert cfg.phases["profiling"].seamless is False

    def test_single_phase_seamless_raises(self) -> None:
        with pytest.raises(ValidationError, match="seamless"):
            BenchmarkConfig(
                **_base_config(
                    phases={
                        "only_phase": {
                            "type": "concurrency",
                            "concurrency": 8,
                            "requests": 100,
                            "seamless": True,
                        },
                    },
                )
            )


# ============================================================
# Class 2: TestStopConditionRequired
# ============================================================


class TestStopConditionRequired:
    """Verify stop-condition validation per phase type."""

    def test_concurrency_phase_with_duration_passes(self) -> None:
        cfg = BenchmarkConfig(
            **_base_config(
                phases={
                    "profiling": {
                        "type": "concurrency",
                        "concurrency": 8,
                        "duration": 60,
                    },
                },
            )
        )
        assert cfg.phases["profiling"].duration == 60.0

    def test_concurrency_phase_with_requests_passes(self) -> None:
        cfg = BenchmarkConfig(
            **_base_config(
                phases={
                    "profiling": {
                        "type": "concurrency",
                        "concurrency": 8,
                        "requests": 100,
                    },
                },
            )
        )
        assert cfg.phases["profiling"].requests == 100

    def test_concurrency_phase_no_stop_condition_raises(self) -> None:
        with pytest.raises(ValidationError, match="at least one of"):
            BenchmarkConfig(
                **_base_config(
                    phases={
                        "profiling": {
                            "type": "concurrency",
                            "concurrency": 8,
                        },
                    },
                )
            )

    def test_fixed_schedule_phase_no_duration_passes(self) -> None:
        """FixedSchedulePhase opts out of stop condition requirement."""
        cfg = BenchmarkConfig(
            **_base_config(
                datasets={
                    "trace": {
                        "type": "file",
                        "path": "/tmp/trace.jsonl",
                        "format": "mooncake_trace",
                        "sampling": "sequential",
                    },
                },
                phases={
                    "replay": {
                        "type": "fixed_schedule",
                        "dataset": "trace",
                    },
                },
            )
        )
        assert cfg.phases["replay"].requests is None
        assert cfg.phases["replay"].duration is None

    def test_user_centric_with_sessions_passes(self) -> None:
        cfg = BenchmarkConfig(
            **_base_config(
                phases={
                    "profiling": {
                        "type": "user_centric",
                        "rate": 10.0,
                        "users": 5,
                        "sessions": 10,
                    },
                },
            )
        )
        assert cfg.phases["profiling"].sessions == 10

    def test_poisson_phase_no_stop_condition_raises(self) -> None:
        with pytest.raises(ValidationError, match="at least one of"):
            BenchmarkConfig(
                **_base_config(
                    phases={
                        "profiling": {
                            "type": "poisson",
                            "rate": 10.0,
                        },
                    },
                )
            )

    def test_constant_phase_with_both_stop_conditions_passes(self) -> None:
        cfg = BenchmarkConfig(
            **_base_config(
                phases={
                    "profiling": {
                        "type": "constant",
                        "rate": 5.0,
                        "requests": 100,
                        "duration": 60,
                    },
                },
            )
        )
        assert cfg.phases["profiling"].requests == 100
        assert cfg.phases["profiling"].duration == 60.0


# ============================================================
# Class 3: TestWeightedModelStrategy
# ============================================================


class TestWeightedModelStrategy:
    """Verify weighted model strategy weight validation on ModelsAdvanced."""

    def test_weighted_with_correct_weights_passes(self) -> None:
        models = ModelsAdvanced(
            strategy=ModelSelectionStrategy.WEIGHTED,
            items=[
                {"name": "model-a", "weight": 0.5},
                {"name": "model-b", "weight": 0.5},
            ],
        )
        assert models.strategy == ModelSelectionStrategy.WEIGHTED

    def test_weighted_with_missing_weights_raises(self) -> None:
        with pytest.raises(ValidationError, match="weights.*specified"):
            ModelsAdvanced(
                strategy=ModelSelectionStrategy.WEIGHTED,
                items=[
                    {"name": "model-a", "weight": 0.7},
                    {"name": "model-b"},
                ],
            )

    def test_weighted_with_wrong_sum_raises(self) -> None:
        with pytest.raises(ValidationError, match="weights must sum to 1.0"):
            ModelsAdvanced(
                strategy=ModelSelectionStrategy.WEIGHTED,
                items=[
                    {"name": "model-a", "weight": 0.3},
                    {"name": "model-b", "weight": 0.3},
                ],
            )

    def test_weighted_within_tolerance_passes(self) -> None:
        """Weights summing to ~1.0 within [0.99, 1.01] tolerance are accepted."""
        models = ModelsAdvanced(
            strategy=ModelSelectionStrategy.WEIGHTED,
            items=[
                {"name": "model-a", "weight": 0.333},
                {"name": "model-b", "weight": 0.333},
                {"name": "model-c", "weight": 0.334},
            ],
        )
        total = sum(item.weight for item in models.items)
        assert 0.99 <= total <= 1.01

    def test_non_weighted_ignores_weights(self) -> None:
        """Non-weighted strategies skip weight validation entirely."""
        models = ModelsAdvanced(
            strategy=ModelSelectionStrategy.ROUND_ROBIN,
            items=[
                {"name": "model-a", "weight": 0.3},
                {"name": "model-b"},
            ],
        )
        assert models.strategy == ModelSelectionStrategy.ROUND_ROBIN

    def test_weighted_single_model_weight_one_passes(self) -> None:
        models = ModelsAdvanced(
            strategy=ModelSelectionStrategy.WEIGHTED,
            items=[{"name": "only-model", "weight": 1.0}],
        )
        assert len(models.items) == 1


# ============================================================
# Class 4: TestUserCentricConstraints
# ============================================================


class TestUserCentricConstraints:
    """Verify user-centric phase constraints (sessions/requests >= users)."""

    def test_sessions_greater_than_users_passes(self) -> None:
        cfg = BenchmarkConfig(
            **_base_config(
                phases={
                    "profiling": {
                        "type": "user_centric",
                        "rate": 10.0,
                        "users": 5,
                        "sessions": 10,
                    },
                },
            )
        )
        phase = cfg.phases["profiling"]
        assert phase.sessions == 10
        assert phase.users == 5

    def test_sessions_less_than_users_raises(self) -> None:
        with pytest.raises(ValidationError, match="num-sessions.*num-users"):
            BenchmarkConfig(
                **_base_config(
                    phases={
                        "profiling": {
                            "type": "user_centric",
                            "rate": 10.0,
                            "users": 5,
                            "sessions": 3,
                        },
                    },
                )
            )

    def test_sessions_equal_to_users_passes(self) -> None:
        cfg = BenchmarkConfig(
            **_base_config(
                phases={
                    "profiling": {
                        "type": "user_centric",
                        "rate": 10.0,
                        "users": 5,
                        "sessions": 5,
                    },
                },
            )
        )
        assert cfg.phases["profiling"].sessions == cfg.phases["profiling"].users

    def test_requests_greater_than_users_passes(self) -> None:
        cfg = BenchmarkConfig(
            **_base_config(
                phases={
                    "profiling": {
                        "type": "user_centric",
                        "rate": 10.0,
                        "users": 5,
                        "requests": 10,
                    },
                },
            )
        )
        assert cfg.phases["profiling"].requests == 10

    def test_requests_less_than_users_raises(self) -> None:
        with pytest.raises(ValidationError, match="request-count.*num-users"):
            BenchmarkConfig(
                **_base_config(
                    phases={
                        "profiling": {
                            "type": "user_centric",
                            "rate": 10.0,
                            "users": 5,
                            "requests": 3,
                        },
                    },
                )
            )

    def test_requests_equal_to_users_passes(self) -> None:
        cfg = BenchmarkConfig(
            **_base_config(
                phases={
                    "profiling": {
                        "type": "user_centric",
                        "rate": 10.0,
                        "users": 5,
                        "requests": 5,
                    },
                },
            )
        )
        assert cfg.phases["profiling"].requests == cfg.phases["profiling"].users


# ============================================================
# Class 5: TestDatasetReferences
# ============================================================


class TestDatasetReferences:
    """Verify dataset reference validation across phases."""

    def test_valid_dataset_reference_passes(self) -> None:
        cfg = BenchmarkConfig(
            **_base_config(
                phases={
                    "profiling": {
                        "type": "concurrency",
                        "concurrency": 8,
                        "requests": 100,
                        "dataset": "main",
                    },
                },
            )
        )
        assert cfg.phases["profiling"].dataset == "main"

    def test_invalid_dataset_reference_raises(self) -> None:
        with pytest.raises(ValidationError, match="undefined dataset.*missing"):
            BenchmarkConfig(
                **_base_config(
                    phases={
                        "profiling": {
                            "type": "concurrency",
                            "concurrency": 8,
                            "requests": 100,
                            "dataset": "missing",
                        },
                    },
                )
            )

    def test_none_dataset_reference_passes(self) -> None:
        """Phase without explicit dataset falls back to first dataset."""
        cfg = BenchmarkConfig(**_base_config())
        assert cfg.phases["profiling"].dataset is None

    def test_multiple_phases_different_datasets_passes(self) -> None:
        cfg = BenchmarkConfig(
            **_base_config(
                datasets={
                    "train": {
                        "type": "synthetic",
                        "entries": 100,
                        "prompts": {"isl": 128},
                    },
                    "eval": {
                        "type": "synthetic",
                        "entries": 50,
                        "prompts": {"isl": 256},
                    },
                },
                phases={
                    "warmup": {
                        "type": "concurrency",
                        "concurrency": 4,
                        "requests": 50,
                        "dataset": "train",
                        "exclude_from_results": True,
                    },
                    "profiling": {
                        "type": "concurrency",
                        "concurrency": 8,
                        "requests": 100,
                        "dataset": "eval",
                    },
                },
            )
        )
        assert cfg.phases["warmup"].dataset == "train"
        assert cfg.phases["profiling"].dataset == "eval"

    def test_one_valid_one_invalid_reference_raises(self) -> None:
        with pytest.raises(ValidationError, match="undefined dataset.*bad"):
            BenchmarkConfig(
                **_base_config(
                    phases={
                        "warmup": {
                            "type": "concurrency",
                            "concurrency": 4,
                            "requests": 50,
                            "dataset": "main",
                            "exclude_from_results": True,
                        },
                        "profiling": {
                            "type": "concurrency",
                            "concurrency": 8,
                            "requests": 100,
                            "dataset": "bad",
                        },
                    },
                )
            )


# ============================================================
# Class 6: TestDurationSpec
# ============================================================


class TestDurationSpec:
    """Verify DurationSpec parsing from various input formats."""

    def test_float_passthrough(self) -> None:
        assert _parse_duration(60.0) == 60.0

    def test_int_coerced_to_float(self) -> None:
        result = _parse_duration(60)
        assert result == 60.0
        assert isinstance(result, float)

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("30s", 30.0),
            ("30sec", 30.0),
            ("30 s", 30.0),
        ],
    )  # fmt: skip
    def test_seconds_string(self, input_val: str, expected: float) -> None:
        assert _parse_duration(input_val) == expected

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("5m", 300.0),
            ("5min", 300.0),
            ("5 m", 300.0),
        ],
    )  # fmt: skip
    def test_minutes_string(self, input_val: str, expected: float) -> None:
        assert _parse_duration(input_val) == expected

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("2h", 7200.0),
            ("2hr", 7200.0),
            ("2hour", 7200.0),
            ("2 h", 7200.0),
        ],
    )  # fmt: skip
    def test_hours_string(self, input_val: str, expected: float) -> None:
        assert _parse_duration(input_val) == expected

    def test_none_passthrough(self) -> None:
        assert _parse_duration(None) is None

    def test_invalid_unit_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid duration format"):
            _parse_duration("5x")

    def test_zero_duration_allowed(self) -> None:
        assert _parse_duration(0) == 0.0

    @pytest.mark.parametrize(
        "input_val",
        [
            "5M",
            "5Min",
            "5MIN",
            "5S",
            "5SEC",
            "2H",
            "2HR",
            "2HOUR",
        ],
    )  # fmt: skip
    def test_case_insensitive_units(self, input_val: str) -> None:
        result = _parse_duration(input_val)
        assert result is not None
        assert result > 0

    def test_fractional_value_with_unit(self) -> None:
        assert _parse_duration("1.5h") == 5400.0

    def test_bare_number_string_treated_as_seconds(self) -> None:
        assert _parse_duration("120") == 120.0

    def test_invalid_format_no_number_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid duration format"):
            _parse_duration("minutes")

    def test_duration_spec_in_phase_config(self) -> None:
        """DurationSpec integration: string durations work in phase fields."""
        cfg = BenchmarkConfig(
            **_base_config(
                phases={
                    "profiling": {
                        "type": "concurrency",
                        "concurrency": 8,
                        "duration": "5m",
                    },
                },
            )
        )
        assert cfg.phases["profiling"].duration == 300.0


# ============================================================
# Class 7: TestGracePeriodRequiresDuration
# ============================================================


class TestGracePeriodRequiresDuration:
    """Verify grace_period requires duration to be set (Requires constraint)."""

    def test_grace_period_with_duration_passes(self) -> None:
        cfg = BenchmarkConfig(
            **_base_config(
                phases={
                    "profiling": {
                        "type": "concurrency",
                        "concurrency": 8,
                        "duration": 60,
                        "grace_period": 10,
                    },
                },
            )
        )
        assert cfg.phases["profiling"].grace_period == 10.0
        assert cfg.phases["profiling"].duration == 60.0

    def test_grace_period_without_duration_raises(self) -> None:
        with pytest.raises(ValidationError, match="duration"):
            BenchmarkConfig(
                **_base_config(
                    phases={
                        "profiling": {
                            "type": "concurrency",
                            "concurrency": 8,
                            "requests": 100,
                            "grace_period": 10,
                        },
                    },
                )
            )

    def test_no_grace_period_no_duration_passes(self) -> None:
        cfg = BenchmarkConfig(
            **_base_config(
                phases={
                    "profiling": {
                        "type": "concurrency",
                        "concurrency": 8,
                        "requests": 100,
                    },
                },
            )
        )
        assert cfg.phases["profiling"].grace_period is None
        assert cfg.phases["profiling"].duration is None

    def test_grace_period_with_duration_string_passes(self) -> None:
        """Both grace_period and duration accept DurationSpec strings."""
        cfg = BenchmarkConfig(
            **_base_config(
                phases={
                    "profiling": {
                        "type": "concurrency",
                        "concurrency": 8,
                        "duration": "5m",
                        "grace_period": "30s",
                    },
                },
            )
        )
        assert cfg.phases["profiling"].duration == 300.0
        assert cfg.phases["profiling"].grace_period == 30.0

    def test_zero_grace_period_with_duration_passes(self) -> None:
        cfg = BenchmarkConfig(
            **_base_config(
                phases={
                    "profiling": {
                        "type": "concurrency",
                        "concurrency": 8,
                        "duration": 60,
                        "grace_period": 0,
                    },
                },
            )
        )
        assert cfg.phases["profiling"].grace_period == 0.0


# ============================================================
# Class 8: TestValidateConfigFileWarnings
# ============================================================


class TestValidateConfigFileWarnings:
    """Verify validate_config_file produces warnings for suspicious configs."""

    def test_all_phases_excluded_warns(self, tmp_path) -> None:
        yaml_content = """
models:
  - test-model
endpoint:
  urls:
    - http://localhost:8000/v1/chat/completions
datasets:
  main:
    type: synthetic
    entries: 100
    prompts:
      isl: 128
      osl: 64
phases:
  warmup:
    type: concurrency
    concurrency: 4
    requests: 50
    exclude_from_results: true
"""
        config_file = tmp_path / "all_excluded.yaml"
        config_file.write_text(yaml_content)

        warnings = validate_config_file(config_file)
        assert any(
            "exclude_from_results" in w.lower() or "empty" in w.lower()
            for w in warnings
        )

    def test_no_warnings_clean_config(self, tmp_path) -> None:
        yaml_content = """
models:
  - test-model
endpoint:
  urls:
    - http://localhost:8000/v1/chat/completions
datasets:
  main:
    type: synthetic
    entries: 100
    prompts:
      isl: 128
      osl: 64
phases:
  profiling:
    type: concurrency
    concurrency: 8
    requests: 100
"""
        config_file = tmp_path / "clean.yaml"
        config_file.write_text(yaml_content)

        warnings = validate_config_file(config_file)
        assert warnings == []
