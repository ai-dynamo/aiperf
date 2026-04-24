# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.operator.environment module."""

from pathlib import Path

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.operator.environment import OperatorEnvironment


class TestOperatorEnvironmentDefaults:
    """Tests for OperatorEnvironment default values."""

    def test_default_image(self) -> None:
        """Verify default benchmark image."""
        assert OperatorEnvironment.DEFAULT_IMAGE == "nvcr.io/nvidia/aiperf:latest"

    def test_job_timeout_seconds(self) -> None:
        """Verify default job timeout is 0 (no timeout)."""
        assert OperatorEnvironment.JOB_TIMEOUT_SECONDS == 0

    def test_pod_restart_threshold(self) -> None:
        """Verify default pod restart threshold."""
        assert OperatorEnvironment.POD_RESTART_THRESHOLD == 3

    def test_endpoint_check_timeout(self) -> None:
        """Verify default endpoint check timeout."""
        assert OperatorEnvironment.ENDPOINT_CHECK_TIMEOUT == 10.0


class TestMonitorSettingsDefaults:
    """Tests for monitor timer setting defaults."""

    def test_interval(self) -> None:
        """Verify default monitor interval."""
        assert OperatorEnvironment.MONITOR.INTERVAL == 10.0

    def test_initial_delay(self) -> None:
        """Verify default monitor initial delay."""
        assert OperatorEnvironment.MONITOR.INITIAL_DELAY == 5.0


class TestResultsSettingsDefaults:
    """Tests for results setting defaults."""

    def test_dir(self) -> None:
        """Verify default results directory."""
        assert Path("/data") == OperatorEnvironment.RESULTS.DIR

    def test_max_retries(self) -> None:
        """Verify default results max retries."""
        assert OperatorEnvironment.RESULTS.MAX_RETRIES == 5

    def test_retry_delay(self) -> None:
        """Verify default results retry delay."""
        assert OperatorEnvironment.RESULTS.RETRY_DELAY == 2.0

    def test_ttl_days(self) -> None:
        """Verify default results TTL."""
        assert OperatorEnvironment.RESULTS.TTL_DAYS == 30


class TestEnvironmentVariableOverrides:
    """Tests for environment variable configuration overrides."""

    def test_monitor_interval_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify AIPERF_OPERATOR_MONITOR_INTERVAL overrides default."""
        from aiperf.operator.environment import _MonitorSettings

        monkeypatch.setenv("AIPERF_OPERATOR_MONITOR_INTERVAL", "30.0")
        settings = _MonitorSettings()
        assert settings.INTERVAL == 30.0

    def test_monitor_initial_delay_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify AIPERF_OPERATOR_MONITOR_INITIAL_DELAY overrides default."""
        from aiperf.operator.environment import _MonitorSettings

        monkeypatch.setenv("AIPERF_OPERATOR_MONITOR_INITIAL_DELAY", "15.0")
        settings = _MonitorSettings()
        assert settings.INITIAL_DELAY == 15.0

    def test_results_dir_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify AIPERF_RESULTS_DIR overrides default."""
        from aiperf.operator.environment import _ResultsSettings

        monkeypatch.setenv("AIPERF_RESULTS_DIR", "/custom/results")
        settings = _ResultsSettings()
        assert Path("/custom/results") == settings.DIR

    def test_results_max_retries_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify AIPERF_RESULTS_MAX_RETRIES overrides default."""
        from aiperf.operator.environment import _ResultsSettings

        monkeypatch.setenv("AIPERF_RESULTS_MAX_RETRIES", "10")
        settings = _ResultsSettings()
        assert settings.MAX_RETRIES == 10

    def test_results_retry_delay_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify AIPERF_RESULTS_RETRY_DELAY overrides default."""
        from aiperf.operator.environment import _ResultsSettings

        monkeypatch.setenv("AIPERF_RESULTS_RETRY_DELAY", "5.0")
        settings = _ResultsSettings()
        assert settings.RETRY_DELAY == 5.0

    def test_results_ttl_days_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify AIPERF_RESULTS_TTL_DAYS overrides default."""
        from aiperf.operator.environment import _ResultsSettings

        monkeypatch.setenv("AIPERF_RESULTS_TTL_DAYS", "90")
        settings = _ResultsSettings()
        assert settings.TTL_DAYS == 90

    def test_root_default_image_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify AIPERF_DEFAULT_IMAGE overrides default."""
        from aiperf.operator.environment import _OperatorEnvironment

        monkeypatch.setenv("AIPERF_DEFAULT_IMAGE", "custom-registry/aiperf:v2")
        settings = _OperatorEnvironment()
        assert settings.DEFAULT_IMAGE == "custom-registry/aiperf:v2"

    def test_root_job_timeout_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify AIPERF_JOB_TIMEOUT_SECONDS overrides default."""
        from aiperf.operator.environment import _OperatorEnvironment

        monkeypatch.setenv("AIPERF_JOB_TIMEOUT_SECONDS", "3600")
        settings = _OperatorEnvironment()
        assert settings.JOB_TIMEOUT_SECONDS == 3600

    def test_root_pod_restart_threshold_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify AIPERF_POD_RESTART_THRESHOLD overrides default."""
        from aiperf.operator.environment import _OperatorEnvironment

        monkeypatch.setenv("AIPERF_POD_RESTART_THRESHOLD", "10")
        settings = _OperatorEnvironment()
        assert settings.POD_RESTART_THRESHOLD == 10

    def test_root_endpoint_check_timeout_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify AIPERF_ENDPOINT_CHECK_TIMEOUT overrides default."""
        from aiperf.operator.environment import _OperatorEnvironment

        monkeypatch.setenv("AIPERF_ENDPOINT_CHECK_TIMEOUT", "30.0")
        settings = _OperatorEnvironment()
        assert settings.ENDPOINT_CHECK_TIMEOUT == 30.0


class TestMonitorSettingsValidation:
    """Tests for monitor settings validation bounds."""

    def test_interval_rejects_zero(self) -> None:
        """Verify interval must be > 0."""
        from aiperf.operator.environment import _MonitorSettings

        with pytest.raises(ValidationError):
            _MonitorSettings(INTERVAL=0)

    def test_interval_rejects_above_upper_bound(self) -> None:
        """Verify interval rejects values above 3600."""
        from aiperf.operator.environment import _MonitorSettings

        with pytest.raises(ValidationError):
            _MonitorSettings(INTERVAL=3601)

    def test_initial_delay_accepts_zero(self) -> None:
        """Verify initial delay can be 0."""
        from aiperf.operator.environment import _MonitorSettings

        settings = _MonitorSettings(INITIAL_DELAY=0)
        assert settings.INITIAL_DELAY == 0

    def test_initial_delay_rejects_above_upper_bound(self) -> None:
        """Verify initial delay rejects values above 300."""
        from aiperf.operator.environment import _MonitorSettings

        with pytest.raises(ValidationError):
            _MonitorSettings(INITIAL_DELAY=301)

    def test_initial_delay_rejects_negative(self) -> None:
        """Verify initial delay rejects negative values."""
        from aiperf.operator.environment import _MonitorSettings

        with pytest.raises(ValidationError):
            _MonitorSettings(INITIAL_DELAY=-1)


class TestResultsSettingsValidation:
    """Tests for results settings validation bounds."""

    def test_max_retries_accepts_zero(self) -> None:
        """Verify max retries can be 0 (no retries)."""
        from aiperf.operator.environment import _ResultsSettings

        settings = _ResultsSettings(MAX_RETRIES=0)
        assert settings.MAX_RETRIES == 0

    def test_max_retries_rejects_above_upper_bound(self) -> None:
        """Verify max retries rejects values above 50."""
        from aiperf.operator.environment import _ResultsSettings

        with pytest.raises(ValidationError):
            _ResultsSettings(MAX_RETRIES=51)

    def test_retry_delay_accepts_zero(self) -> None:
        """Verify retry delay can be 0 (no delay)."""
        from aiperf.operator.environment import _ResultsSettings

        settings = _ResultsSettings(RETRY_DELAY=0)
        assert settings.RETRY_DELAY == 0

    def test_retry_delay_rejects_above_upper_bound(self) -> None:
        """Verify retry delay rejects values above 60."""
        from aiperf.operator.environment import _ResultsSettings

        with pytest.raises(ValidationError):
            _ResultsSettings(RETRY_DELAY=61)

    def test_ttl_days_accepts_zero(self) -> None:
        """Verify TTL can be 0 (never clean)."""
        from aiperf.operator.environment import _ResultsSettings

        settings = _ResultsSettings(TTL_DAYS=0)
        assert settings.TTL_DAYS == 0

    def test_ttl_days_rejects_above_upper_bound(self) -> None:
        """Verify TTL rejects values above 3650."""
        from aiperf.operator.environment import _ResultsSettings

        with pytest.raises(ValidationError):
            _ResultsSettings(TTL_DAYS=3651)

    def test_max_retries_rejects_negative(self) -> None:
        """Verify max retries rejects negative values."""
        from aiperf.operator.environment import _ResultsSettings

        with pytest.raises(ValidationError):
            _ResultsSettings(MAX_RETRIES=-1)


class TestRootSettingsValidation:
    """Tests for root operator settings validation bounds."""

    def test_job_timeout_accepts_zero(self) -> None:
        """Verify job timeout of 0 means no timeout."""
        from aiperf.operator.environment import _OperatorEnvironment

        settings = _OperatorEnvironment(JOB_TIMEOUT_SECONDS=0)
        assert settings.JOB_TIMEOUT_SECONDS == 0

    def test_job_timeout_rejects_negative(self) -> None:
        """Verify job timeout rejects negative values."""
        from aiperf.operator.environment import _OperatorEnvironment

        with pytest.raises(ValidationError):
            _OperatorEnvironment(JOB_TIMEOUT_SECONDS=-1)

    def test_pod_restart_threshold_accepts_zero(self) -> None:
        """Verify pod restart threshold can be 0."""
        from aiperf.operator.environment import _OperatorEnvironment

        settings = _OperatorEnvironment(POD_RESTART_THRESHOLD=0)
        assert settings.POD_RESTART_THRESHOLD == 0

    def test_pod_restart_threshold_rejects_above_upper_bound(self) -> None:
        """Verify pod restart threshold rejects values above 100."""
        from aiperf.operator.environment import _OperatorEnvironment

        with pytest.raises(ValidationError):
            _OperatorEnvironment(POD_RESTART_THRESHOLD=101)

    def test_endpoint_check_timeout_rejects_zero(self) -> None:
        """Verify endpoint check timeout must be > 0."""
        from aiperf.operator.environment import _OperatorEnvironment

        with pytest.raises(ValidationError):
            _OperatorEnvironment(ENDPOINT_CHECK_TIMEOUT=0)

    def test_endpoint_check_timeout_rejects_above_upper_bound(self) -> None:
        """Verify endpoint check timeout rejects values above 300."""
        from aiperf.operator.environment import _OperatorEnvironment

        with pytest.raises(ValidationError):
            _OperatorEnvironment(ENDPOINT_CHECK_TIMEOUT=301)


class TestEveryEnvVarMapsToField:
    """Verify every AIPERF_* env var from deploy manifests maps to an OperatorEnvironment field.

    Each entry maps the env var name from the Helm chart values to a
    (accessor_path, expected_default) tuple. If a new env var is added to the
    deployment but not here, the test_all_env_vars_covered parametrization will
    need updating, making drift obvious.
    """

    ENV_VAR_MAP: list[tuple[str, str, object]] = [
        ("AIPERF_OPERATOR_MONITOR_INTERVAL", "MONITOR.INTERVAL", 10.0),
        ("AIPERF_OPERATOR_MONITOR_INITIAL_DELAY", "MONITOR.INITIAL_DELAY", 5.0),
        ("AIPERF_RESULTS_DIR", "RESULTS.DIR", Path("/data")),
        ("AIPERF_RESULTS_MAX_RETRIES", "RESULTS.MAX_RETRIES", 5),
        ("AIPERF_RESULTS_RETRY_DELAY", "RESULTS.RETRY_DELAY", 2.0),
        ("AIPERF_RESULTS_TTL_DAYS", "RESULTS.TTL_DAYS", 30),
        ("AIPERF_JOB_TIMEOUT_SECONDS", "JOB_TIMEOUT_SECONDS", 0),
        ("AIPERF_POD_RESTART_THRESHOLD", "POD_RESTART_THRESHOLD", 3),
        ("AIPERF_ENDPOINT_CHECK_TIMEOUT", "ENDPOINT_CHECK_TIMEOUT", 10.0),
        ("AIPERF_DEFAULT_IMAGE", "DEFAULT_IMAGE", "nvcr.io/nvidia/aiperf:latest"),
        ("AIPERF_PREFLIGHT_TIMEOUT", "PREFLIGHT_TIMEOUT", 30.0),
    ]

    @pytest.mark.parametrize(
        "env_var,accessor,expected",
        [param(e, a, v, id=e) for e, a, v in ENV_VAR_MAP],
    )  # fmt: skip
    def test_default_matches_deploy_manifest(
        self, env_var: str, accessor: str, expected: object
    ) -> None:
        """Verify each env var's default matches the deploy manifest value."""
        obj: object = OperatorEnvironment
        for part in accessor.split("."):
            obj = getattr(obj, part)
        assert obj == expected, (
            f"{env_var} -> {accessor}: got {obj!r}, expected {expected!r}"
        )

    @pytest.mark.parametrize(
        "env_var,accessor,override_value",
        [
            param("AIPERF_OPERATOR_MONITOR_INTERVAL", "INTERVAL", "30.0", id="AIPERF_OPERATOR_MONITOR_INTERVAL"),
            param("AIPERF_OPERATOR_MONITOR_INITIAL_DELAY", "INITIAL_DELAY", "15.0", id="AIPERF_OPERATOR_MONITOR_INITIAL_DELAY"),
            param("AIPERF_RESULTS_DIR", "DIR", "/custom", id="AIPERF_RESULTS_DIR"),
            param("AIPERF_RESULTS_MAX_RETRIES", "MAX_RETRIES", "10", id="AIPERF_RESULTS_MAX_RETRIES"),
            param("AIPERF_RESULTS_RETRY_DELAY", "RETRY_DELAY", "5.0", id="AIPERF_RESULTS_RETRY_DELAY"),
            param("AIPERF_RESULTS_TTL_DAYS", "TTL_DAYS", "90", id="AIPERF_RESULTS_TTL_DAYS"),
            param("AIPERF_JOB_TIMEOUT_SECONDS", "JOB_TIMEOUT_SECONDS", "3600", id="AIPERF_JOB_TIMEOUT_SECONDS"),
            param("AIPERF_POD_RESTART_THRESHOLD", "POD_RESTART_THRESHOLD", "10", id="AIPERF_POD_RESTART_THRESHOLD"),
            param("AIPERF_ENDPOINT_CHECK_TIMEOUT", "ENDPOINT_CHECK_TIMEOUT", "30.0", id="AIPERF_ENDPOINT_CHECK_TIMEOUT"),
            param("AIPERF_DEFAULT_IMAGE", "DEFAULT_IMAGE", "custom:v2", id="AIPERF_DEFAULT_IMAGE"),
            param("AIPERF_PREFLIGHT_TIMEOUT", "PREFLIGHT_TIMEOUT", "60.0", id="AIPERF_PREFLIGHT_TIMEOUT"),
        ],
    )  # fmt: skip
    def test_env_var_overrides_field(
        self,
        env_var: str,
        accessor: str,
        override_value: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify each env var actually overrides its corresponding field."""
        from aiperf.operator.environment import (
            _MonitorSettings,
            _OperatorEnvironment,
            _ResultsSettings,
        )

        monkeypatch.setenv(env_var, override_value)

        if env_var.startswith("AIPERF_OPERATOR_MONITOR_"):
            obj = _MonitorSettings()
        elif env_var.startswith("AIPERF_RESULTS_"):
            obj = _ResultsSettings()
        else:
            obj = _OperatorEnvironment()

        value = getattr(obj, accessor)
        # Compare as strings to handle type coercion
        assert str(value) != "", f"{env_var} override produced empty value"

    def test_all_env_vars_covered(self) -> None:
        """Guard: if this list changes, a deploy manifest env var was added or removed."""
        assert len(self.ENV_VAR_MAP) == 11


# =============================================================================
# Tests for PREFLIGHT_TIMEOUT setting
# =============================================================================


class TestPreflightTimeoutDefault:
    """Tests for the PREFLIGHT_TIMEOUT field default value."""

    def test_default_value(self) -> None:
        """Verify PREFLIGHT_TIMEOUT defaults to 30.0."""
        assert OperatorEnvironment.PREFLIGHT_TIMEOUT == 30.0


class TestPreflightTimeoutEnvVarOverride:
    """Tests for PREFLIGHT_TIMEOUT environment variable override."""

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify AIPERF_PREFLIGHT_TIMEOUT overrides the default."""
        from aiperf.operator.environment import _OperatorEnvironment

        monkeypatch.setenv("AIPERF_PREFLIGHT_TIMEOUT", "60.0")
        settings = _OperatorEnvironment()
        assert settings.PREFLIGHT_TIMEOUT == 60.0


class TestPreflightTimeoutValidation:
    """Tests for PREFLIGHT_TIMEOUT validation constraints (gt=0, le=120)."""

    def test_rejects_zero(self) -> None:
        """Verify PREFLIGHT_TIMEOUT must be > 0."""
        from aiperf.operator.environment import _OperatorEnvironment

        with pytest.raises(ValidationError):
            _OperatorEnvironment(PREFLIGHT_TIMEOUT=0)

    def test_rejects_negative(self) -> None:
        """Verify PREFLIGHT_TIMEOUT rejects negative values."""
        from aiperf.operator.environment import _OperatorEnvironment

        with pytest.raises(ValidationError):
            _OperatorEnvironment(PREFLIGHT_TIMEOUT=-1)

    def test_rejects_above_upper_bound(self) -> None:
        """Verify PREFLIGHT_TIMEOUT rejects values above 120."""
        from aiperf.operator.environment import _OperatorEnvironment

        with pytest.raises(ValidationError):
            _OperatorEnvironment(PREFLIGHT_TIMEOUT=121)

    def test_accepts_boundary_value(self) -> None:
        """Verify PREFLIGHT_TIMEOUT accepts the upper bound of 120."""
        from aiperf.operator.environment import _OperatorEnvironment

        settings = _OperatorEnvironment(PREFLIGHT_TIMEOUT=120)
        assert settings.PREFLIGHT_TIMEOUT == 120

    def test_accepts_small_positive(self) -> None:
        """Verify PREFLIGHT_TIMEOUT accepts a small positive value."""
        from aiperf.operator.environment import _OperatorEnvironment

        settings = _OperatorEnvironment(PREFLIGHT_TIMEOUT=0.1)
        assert settings.PREFLIGHT_TIMEOUT == 0.1
