# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SteadyStateConfig and v1 UserConfig steady-state validation.

Steady-state cross-field validation lives on the v1->v2 converter
(``aiperf.config.v1._converter_runtime.validate_steady_state``), per the v1
hard rule that AIPerfConfig is the single validation gate. These tests call
``validate_steady_state(user)`` directly to exercise the same rules the old
``UserConfig.validate_steady_state_options`` model-validator covered.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aiperf.config.steady_state import SteadyStateConfig
from aiperf.config.v1 import UserConfig
from aiperf.config.v1._converter_runtime import validate_steady_state
from aiperf.config.v1._endpoint import EndpointConfig
from aiperf.config.v1._output import OutputConfig

# ---------------------------------------------------------------------------
# SteadyStateConfig unit tests
# ---------------------------------------------------------------------------


class TestSteadyStateConfigDefaults:
    def test_defaults(self) -> None:
        config = SteadyStateConfig()
        assert config.enabled is False
        assert config.start_pct is None
        assert config.end_pct is None
        assert config.min_window_pct == 10.0
        assert config.bootstrap_iterations is None

    def test_enabled(self) -> None:
        config = SteadyStateConfig(enabled=True)
        assert config.enabled is True

    def test_user_override(self) -> None:
        config = SteadyStateConfig(start_pct=10.0, end_pct=90.0)
        assert config.start_pct == 10.0
        assert config.end_pct == 90.0


class TestSteadyStateConfigValidation:
    def test_start_pct_bounds(self) -> None:
        with pytest.raises(ValidationError):
            SteadyStateConfig(start_pct=-1.0)  # ge=0.0
        with pytest.raises(ValidationError):
            SteadyStateConfig(start_pct=100.0)  # lt=100.0

    def test_end_pct_bounds(self) -> None:
        with pytest.raises(ValidationError):
            SteadyStateConfig(end_pct=0.0)  # gt=0.0
        with pytest.raises(ValidationError):
            SteadyStateConfig(end_pct=101.0)  # le=100.0

    def test_min_window_pct_bounds(self) -> None:
        with pytest.raises(ValidationError):
            SteadyStateConfig(min_window_pct=0.0)  # gt=0.0
        with pytest.raises(ValidationError):
            SteadyStateConfig(min_window_pct=101.0)  # le=100.0

    def test_bootstrap_iterations_bounds(self) -> None:
        with pytest.raises(ValidationError):
            SteadyStateConfig(bootstrap_iterations=0)  # gt=0
        config = SteadyStateConfig(bootstrap_iterations=50)
        assert config.bootstrap_iterations == 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_endpoint() -> EndpointConfig:
    return EndpointConfig(
        model_names=["test-model"],
        custom_endpoint="test",
    )


def _make_config(**output_kwargs) -> UserConfig:
    output = OutputConfig(**output_kwargs) if output_kwargs else OutputConfig()
    return UserConfig(endpoint=_make_endpoint(), output=output)


# ---------------------------------------------------------------------------
# v1 converter steady-state validation tests
# ---------------------------------------------------------------------------


class TestUserConfigSteadyStateValidation:
    def test_start_pct_without_end_pct_raises(self) -> None:
        user = _make_config(steady_state=SteadyStateConfig(start_pct=10.0))
        with pytest.raises(ValueError, match="must be used together"):
            validate_steady_state(user)

    def test_end_pct_without_start_pct_raises(self) -> None:
        user = _make_config(steady_state=SteadyStateConfig(end_pct=90.0))
        with pytest.raises(ValueError, match="must be used together"):
            validate_steady_state(user)

    def test_start_pct_gte_end_pct_raises(self) -> None:
        user = _make_config(
            steady_state=SteadyStateConfig(start_pct=50.0, end_pct=50.0)
        )
        with pytest.raises(ValueError, match="must be less than"):
            validate_steady_state(user)

    def test_start_pct_greater_than_end_pct_raises(self) -> None:
        user = _make_config(
            steady_state=SteadyStateConfig(start_pct=90.0, end_pct=10.0)
        )
        with pytest.raises(ValueError, match="must be less than"):
            validate_steady_state(user)

    def test_valid_start_end_pct(self) -> None:
        user = _make_config(
            steady_state=SteadyStateConfig(start_pct=10.0, end_pct=90.0)
        )
        validate_steady_state(user)
        assert user.output.steady_state.start_pct == 10.0
        assert user.output.steady_state.end_pct == 90.0

    def test_manual_override_enables_implicitly(self) -> None:
        """Setting start_pct + end_pct should implicitly enable steady-state."""
        user = _make_config(
            steady_state=SteadyStateConfig(enabled=False, start_pct=10.0, end_pct=90.0)
        )
        validate_steady_state(user)
        assert user.output.steady_state.enabled is True

    def test_disabled_by_default(self) -> None:
        user = _make_config()
        validate_steady_state(user)
        assert user.output.steady_state.enabled is False

    def test_explicitly_enabled(self) -> None:
        user = _make_config(steady_state=SteadyStateConfig(enabled=True))
        validate_steady_state(user)
        assert user.output.steady_state.enabled is True


class TestSteadyStateConfigInOutputConfig:
    def test_output_config_has_steady_state(self) -> None:
        output = OutputConfig()
        assert hasattr(output, "steady_state")
        assert isinstance(output.steady_state, SteadyStateConfig)

    def test_output_config_custom_steady_state(self) -> None:
        output = OutputConfig(
            steady_state=SteadyStateConfig(enabled=True, min_window_pct=15.0)
        )
        assert output.steady_state.enabled is True
        assert output.steady_state.min_window_pct == 15.0
