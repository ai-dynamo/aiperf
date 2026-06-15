# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the UserConfig --scenario / --unsafe-override hook (Task 9)."""

import pytest

from aiperf.common.config import (
    EndpointConfig,
    InputConfig,
    LoadGeneratorConfig,
    UserConfig,
)
from aiperf.common.config.prompt_config import CacheBustConfig
from aiperf.common.enums import CacheBustTarget


def _minimal_endpoint() -> EndpointConfig:
    return EndpointConfig(model_names=["test-model"])


def test_user_config_calls_validator_when_scenario_set(monkeypatch):
    called = {"yes": False}

    def fake_validate(cfg, **_kwargs):
        called["yes"] = True
        from aiperf.common.scenario.validator import ValidationOutcome

        return ValidationOutcome(violations=[], submission_valid=True)

    monkeypatch.setattr(
        "aiperf.common.scenario.validator.validate_scenario", fake_validate
    )

    cfg = UserConfig(
        endpoint=_minimal_endpoint(),
        scenario="inferencex-agentx-mvp",
    )
    assert called["yes"] is True
    assert cfg._scenario_outcome is not None
    assert cfg._scenario_outcome.submission_valid is True


def test_user_config_skips_validator_when_scenario_absent(monkeypatch):
    """validate_scenario is still invoked but is a no-op when scenario is None."""
    seen_scenario_values: list[str | None] = []

    real_validate = None
    from aiperf.common.scenario import validator as _validator_mod

    real_validate = _validator_mod.validate_scenario

    def spy(cfg, **kwargs):
        seen_scenario_values.append(cfg.scenario)
        return real_validate(cfg, **kwargs)

    monkeypatch.setattr("aiperf.common.scenario.validator.validate_scenario", spy)

    cfg = UserConfig(endpoint=_minimal_endpoint())
    assert isinstance(cfg, UserConfig)
    assert cfg.scenario is None
    # Validator was called once and saw scenario=None (no-op outcome).
    assert seen_scenario_values == [None]
    assert cfg._scenario_outcome is not None
    assert cfg._scenario_outcome.submission_valid is None


def test_scenario_lock_error_raises_without_unsafe_override(tmp_path):
    """Default config violates inferencex-agentx-mvp invariants → raise.

    pydantic wraps the ScenarioLockError (a ValueError subclass) into
    ValidationError when raised from a model_validator. We assert the
    underlying message text is preserved.
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc_info:
        UserConfig(
            endpoint=_minimal_endpoint(),
            scenario="inferencex-agentx-mvp",
        )
    assert "Scenario invariants violated" in str(exc_info.value)
    # Default UserConfig has no input file, which violates the required-loader
    # invariant. timing_mode, cache_bust.target, and benchmark_duration would
    # also conflict, but the validator auto-injects agentic_replay /
    # FIRST_TURN_PREFIX / the 1800s default duration before the lock check,
    # so none of those surface as violations.
    assert "--input-file (loader)" in str(exc_info.value)


def test_unsafe_override_downgrades_to_warning(caplog):
    """With --unsafe-override, violations log warnings and submission_valid=False."""
    with caplog.at_level("WARNING"):
        cfg = UserConfig(
            endpoint=_minimal_endpoint(),
            scenario="inferencex-agentx-mvp",
            unsafe_override=True,
        )
    assert cfg._scenario_outcome.submission_valid is False
    assert "unsafe_override" in cfg._scenario_outcome.submission_invalid_reasons
    assert any("Scenario violation" in r.getMessage() for r in caplog.records), (
        "expected at least one scenario violation warning"
    )


def test_unsafe_override_alone_is_noop_without_scenario():
    """--unsafe-override without --scenario should not affect validation."""
    cfg = UserConfig(
        endpoint=_minimal_endpoint(),
        unsafe_override=True,
    )
    assert cfg.scenario is None
    assert cfg._scenario_outcome.submission_valid is None
    assert cfg._scenario_outcome.violations == []


class TestExplicitlySetFlags:
    """Verify the underscore flags the scenario validator depends on."""

    def test_use_think_time_only_explicit_flag_when_passed(self):
        cfg = InputConfig(use_think_time_only=True)
        assert cfg._use_think_time_only_explicitly_set is True

    def test_use_think_time_only_explicit_flag_when_omitted(self):
        cfg = InputConfig()
        assert cfg._use_think_time_only_explicitly_set is False

    def test_inter_turn_delay_cap_explicit_flag_when_passed(self):
        cfg = LoadGeneratorConfig(inter_turn_delay_cap_seconds=60.0)
        assert cfg._inter_turn_delay_cap_explicitly_set is True

    def test_inter_turn_delay_cap_explicit_flag_when_omitted(self):
        cfg = LoadGeneratorConfig()
        assert cfg._inter_turn_delay_cap_explicitly_set is False

    def test_trace_idle_gap_cap_explicit_flag_when_passed(self):
        cfg = LoadGeneratorConfig(trace_idle_gap_cap_seconds=60.0)
        assert cfg._trace_idle_gap_cap_explicitly_set is True

    def test_trace_idle_gap_cap_explicit_flag_when_omitted(self):
        cfg = LoadGeneratorConfig()
        assert cfg._trace_idle_gap_cap_explicitly_set is False

    def test_cache_bust_target_explicit_flag_when_passed(self):
        cfg = CacheBustConfig(target=CacheBustTarget.SYSTEM_PREFIX)
        assert cfg._target_explicitly_set is True

    def test_cache_bust_target_explicit_flag_when_omitted(self):
        cfg = CacheBustConfig()
        assert cfg._target_explicitly_set is False
        assert cfg.target == CacheBustTarget.NONE

    def test_extra_inputs_parsed_canonicalizes_dict_input(self):
        cfg = InputConfig(extra={"ignore_eos": True})
        assert cfg.extra_inputs_parsed == {"ignore_eos": True}

    def test_extra_inputs_parsed_canonicalizes_tuple_list(self):
        cfg = InputConfig(extra=[("ignore_eos", True), ("max_tokens", 100)])
        assert cfg.extra_inputs_parsed == {"ignore_eos": True, "max_tokens": 100}

    def test_extra_inputs_parsed_default_empty_dict(self):
        cfg = InputConfig()
        assert cfg.extra_inputs_parsed == {}

    def test_detected_loader_default_none(self):
        cfg = InputConfig()
        assert cfg.detected_loader is None
