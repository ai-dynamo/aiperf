# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
from pytest import param

from aiperf.common.environment import (
    _APIServerSettings,
    _CompressionSettings,
    _Environment,
    _SearchPlannerSettings,
    _ServiceSettings,
)


class TestServiceSettingsUvloopWindows:
    """Test suite for automatic uvloop disabling on Windows."""

    @pytest.mark.parametrize(
        "is_windows,expected_disable_uvloop",
        [
            param(True, True, id="windows_auto_disabled"),
            param(False, False, id="linux_enabled"),
            param(False, False, id="macos_enabled"),
        ],
    )
    def test_platform_uvloop_detection(self, is_windows, expected_disable_uvloop):
        """Test that uvloop is automatically disabled on Windows and enabled elsewhere."""
        with patch("aiperf.common.environment.IS_WINDOWS", is_windows):
            settings = _ServiceSettings()
            assert settings.DISABLE_UVLOOP is expected_disable_uvloop

    @pytest.mark.parametrize(
        "is_windows,manual_setting,expected_result",
        [
            param(True, False, True, id="windows_override_attempt"),
            param(True, True, True, id="windows_manual_disable"),
            param(False, True, True, id="linux_manual_disable"),
            param(False, False, False, id="linux_default_enabled"),
            param(False, True, True, id="macos_manual_disable"),
            param(False, False, False, id="macos_default_enabled"),
        ],
    )
    def test_manual_uvloop_settings(self, is_windows, manual_setting, expected_result):
        """Test manual DISABLE_UVLOOP settings across platforms."""
        with patch("aiperf.common.environment.IS_WINDOWS", is_windows):
            settings = _ServiceSettings(DISABLE_UVLOOP=manual_setting)
            assert settings.DISABLE_UVLOOP is expected_result


class TestWindowsTcpPortWindowValidation:
    """Pins the port-window-fits-in-TCP-range validator. Without it,
    ``build_socket_address`` would silently emit invalid URLs like
    ``tcp://127.0.0.1:84999`` and fail at ZMQ bind time with a
    misleading error.
    """

    def test_valid_window_within_range_is_accepted(self) -> None:
        """The default values (28000 + 20000 = 48000 max port) are well
        under 65535 and must not trip the validator."""
        settings = _ServiceSettings()
        assert (
            settings.WINDOWS_TCP_BASE_PORT + settings.WINDOWS_TCP_PORT_RANGE - 1
            <= 65535
        )

    def test_boundary_window_exactly_fitting_is_accepted(self) -> None:
        """Window where ``base + range - 1 == 65535`` is the largest legal
        config — must NOT raise. Example: base 45536 + range 20000 → max
        port 65535."""
        _ServiceSettings(WINDOWS_TCP_BASE_PORT=45536, WINDOWS_TCP_PORT_RANGE=20000)

    def test_window_overflowing_max_port_raises(self) -> None:
        """When ``base + range - 1 > 65535`` the model_validator must raise
        with a message naming both env-var names so the user knows which
        knob to adjust."""
        with pytest.raises(Exception) as exc:
            _ServiceSettings(WINDOWS_TCP_BASE_PORT=65000, WINDOWS_TCP_PORT_RANGE=20000)
        msg = str(exc.value)
        assert "WINDOWS_TCP_BASE_PORT" in msg
        assert "WINDOWS_TCP_PORT_RANGE" in msg
        assert "65535" in msg


class TestProfileConfigureTimeout:
    """Test suite for profile configure timeout validation."""

    @pytest.mark.parametrize(
        "profile_timeout,dataset_timeout,should_raise",
        [
            param(300.0, 300.0, False, id="equal_timeouts_valid"),
            param(400.0, 300.0, False, id="profile_greater_than_dataset_valid"),
            param(200.0, 300.0, True, id="profile_less_than_dataset_invalid"),
            param(1.0, 1.0, False, id="minimum_equal_timeouts_valid"),
            param(100000.0, 100000.0, False, id="maximum_equal_timeouts_valid"),
            param(100000.0, 1.0, False, id="maximum_difference_valid"),
            param(1.0, 100000.0, True, id="maximum_difference_invalid"),
        ],
    )
    def test_validate_profile_configure_timeout(
        self, profile_timeout, dataset_timeout, should_raise, monkeypatch
    ):
        """Test that profile configure timeout validation enforces timeout >= dataset timeout."""
        # Set environment variables to override the defaults
        monkeypatch.setenv(
            "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT", str(profile_timeout)
        )
        monkeypatch.setenv("AIPERF_DATASET_CONFIGURATION_TIMEOUT", str(dataset_timeout))

        if should_raise:
            with pytest.raises(
                ValueError,
                match=r"AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT.*must be greater than or equal to.*AIPERF_DATASET_CONFIGURATION_TIMEOUT",
            ):
                _Environment()
        else:
            env = _Environment()
            assert (
                env.SERVICE.PROFILE_CONFIGURE_TIMEOUT
                >= env.DATASET.CONFIGURATION_TIMEOUT
            )
            assert profile_timeout == env.SERVICE.PROFILE_CONFIGURE_TIMEOUT
            assert dataset_timeout == env.DATASET.CONFIGURATION_TIMEOUT


class TestAPIServerSettings:
    """Test _APIServerSettings defaults and env var overrides."""

    def test_api_server_settings_no_env_returns_defaults(self) -> None:
        settings = _APIServerSettings()
        assert settings.HOST == "127.0.0.1"
        assert settings.PORT is None
        assert settings.CORS_ORIGINS == []

    def test_api_server_settings_env_port_overrides_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AIPERF_API_SERVER_PORT", "8080")
        settings = _APIServerSettings()
        assert settings.PORT == 8080

    @pytest.mark.parametrize(
        "bad_port",
        [
            param("0", id="zero"),
            param("-1", id="negative"),
            param("70000", id="above_max"),
        ],
    )
    def test_api_server_settings_port_out_of_range_raises(
        self, bad_port: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AIPERF_API_SERVER_PORT", bad_port)
        with pytest.raises(ValueError):
            _APIServerSettings()

    def test_api_server_settings_env_host_overrides_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AIPERF_API_SERVER_HOST", "0.0.0.0")
        settings = _APIServerSettings()
        assert settings.HOST == "0.0.0.0"

    def test_api_server_settings_env_cors_origins_parses_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "AIPERF_API_SERVER_CORS_ORIGINS", '["http://localhost:3000"]'
        )
        settings = _APIServerSettings()
        assert settings.CORS_ORIGINS == ["http://localhost:3000"]

    def test_api_server_settings_shutdown_timeout_default(self) -> None:
        settings = _APIServerSettings()
        assert settings.SHUTDOWN_TIMEOUT == 5.0

    def test_api_server_settings_env_shutdown_timeout_overrides_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AIPERF_API_SERVER_SHUTDOWN_TIMEOUT", "30.0")
        settings = _APIServerSettings()
        assert settings.SHUTDOWN_TIMEOUT == 30.0

    @pytest.mark.parametrize(
        "bad_value",
        [
            param("0.5", id="below_minimum"),
            param("301", id="above_maximum"),
        ],
    )
    def test_api_server_settings_shutdown_timeout_out_of_range_raises(
        self, bad_value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AIPERF_API_SERVER_SHUTDOWN_TIMEOUT", bad_value)
        with pytest.raises(ValueError):
            _APIServerSettings()

    def test_environment_api_server_subsystem_exists(self) -> None:
        env = _Environment()
        assert hasattr(env, "API_SERVER")
        assert isinstance(env.API_SERVER, _APIServerSettings)

    def test_api_server_settings_post_complete_grace_default(self) -> None:
        """POST_COMPLETE_GRACE defaults to 5.0 seconds when no env var is set."""
        settings = _APIServerSettings()
        assert settings.POST_COMPLETE_GRACE == 5.0

    def test_api_server_settings_env_post_complete_grace_overrides_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AIPERF_API_SERVER_POST_COMPLETE_GRACE env var overrides the default."""
        monkeypatch.setenv("AIPERF_API_SERVER_POST_COMPLETE_GRACE", "2.5")
        settings = _APIServerSettings()
        assert settings.POST_COMPLETE_GRACE == 2.5

    def test_api_server_settings_post_complete_grace_zero_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Zero is a valid value (recovers pre-fix immediate-shutdown behavior)."""
        monkeypatch.setenv("AIPERF_API_SERVER_POST_COMPLETE_GRACE", "0")
        settings = _APIServerSettings()
        assert settings.POST_COMPLETE_GRACE == 0.0

    @pytest.mark.parametrize(
        "bad_value",
        [
            param("-0.1", id="below_minimum"),
            param("301", id="above_maximum"),
        ],
    )  # fmt: skip
    def test_api_server_settings_post_complete_grace_out_of_range_raises(
        self, bad_value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Values outside [0.0, 300.0] are rejected at settings construction."""
        monkeypatch.setenv("AIPERF_API_SERVER_POST_COMPLETE_GRACE", bad_value)
        with pytest.raises(ValueError):
            _APIServerSettings()


class TestCompressionSettings:
    """Test _CompressionSettings defaults and validation."""

    def test_compression_settings_defaults_valid(self) -> None:
        settings = _CompressionSettings()
        assert settings.CHUNK_SIZE == 65536
        assert settings.ZSTD_LEVEL == 3
        assert settings.GZIP_LEVEL == 6

    def test_compression_settings_chunk_size_env_override_applied(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("AIPERF_COMPRESSION_CHUNK_SIZE", "131072")
        settings = _CompressionSettings()
        assert settings.CHUNK_SIZE == 131072

    @pytest.mark.parametrize(
        "field,env_var,value",
        [
            param("ZSTD_LEVEL", "AIPERF_COMPRESSION_ZSTD_LEVEL", "10", id="zstd"),
            param("GZIP_LEVEL", "AIPERF_COMPRESSION_GZIP_LEVEL", "9", id="gzip"),
        ],
    )
    def test_compression_settings_level_env_override_applied(
        self, field, env_var, value, monkeypatch
    ) -> None:
        monkeypatch.setenv(env_var, value)
        settings = _CompressionSettings()
        assert getattr(settings, field) == int(value)

    @pytest.mark.parametrize(
        "env_var,bad_value",
        [
            param("AIPERF_COMPRESSION_CHUNK_SIZE", "512", id="chunk_too_small"),
            param("AIPERF_COMPRESSION_CHUNK_SIZE", "2097152", id="chunk_too_large"),
            param("AIPERF_COMPRESSION_ZSTD_LEVEL", "0", id="zstd_too_low"),
            param("AIPERF_COMPRESSION_ZSTD_LEVEL", "23", id="zstd_too_high"),
            param("AIPERF_COMPRESSION_GZIP_LEVEL", "0", id="gzip_too_low"),
            param("AIPERF_COMPRESSION_GZIP_LEVEL", "10", id="gzip_too_high"),
        ],
    )
    def test_compression_settings_out_of_range_raises_value_error(
        self, env_var, bad_value, monkeypatch
    ) -> None:
        monkeypatch.setenv(env_var, bad_value)
        with pytest.raises(ValueError):
            _CompressionSettings()

    def test_environment_compression_subsystem_exists(self) -> None:
        env = _Environment()
        assert hasattr(env, "COMPRESSION")
        assert isinstance(env.COMPRESSION, _CompressionSettings)


class TestSearchPlannerSettings:
    """Test _SearchPlannerSettings defaults and env-var overrides."""

    def test_search_planner_defaults(self) -> None:
        settings = _SearchPlannerSettings()
        assert settings.SLA_PRECISION_DEFAULT == 0.05
        assert settings.DEFAULT_WARMUP_SECONDS == 30.0
        assert settings.FIRST_PROBE_WARMUP_FLOOR == 60.0
        assert settings.REPLICATE_WARMUP_FLOOR == 15.0
        assert settings.SLA_PRECISION_REQUESTS == {
            "tight": 10000,
            "normal": 1000,
            "coarse": 300,
        }

    @pytest.mark.parametrize(
        "field,env_var,value,expected",
        [
            param(
                "SLA_PRECISION_DEFAULT",
                "AIPERF_SEARCH_PLANNER_SLA_PRECISION_DEFAULT",
                "0.10",
                0.10,
                id="precision_default",
            ),
            param(
                "DEFAULT_WARMUP_SECONDS",
                "AIPERF_SEARCH_PLANNER_DEFAULT_WARMUP_SECONDS",
                "45.0",
                45.0,
                id="default_warmup",
            ),
            param(
                "FIRST_PROBE_WARMUP_FLOOR",
                "AIPERF_SEARCH_PLANNER_FIRST_PROBE_WARMUP_FLOOR",
                "120.0",
                120.0,
                id="first_probe_floor",
            ),
            param(
                "REPLICATE_WARMUP_FLOOR",
                "AIPERF_SEARCH_PLANNER_REPLICATE_WARMUP_FLOOR",
                "5.0",
                5.0,
                id="replicate_floor",
            ),
        ],
    )
    def test_search_planner_env_override_applied(
        self, field, env_var, value, expected, monkeypatch
    ) -> None:
        monkeypatch.setenv(env_var, value)
        settings = _SearchPlannerSettings()
        assert getattr(settings, field) == expected

    def test_search_planner_sla_precision_requests_json_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "AIPERF_SEARCH_PLANNER_SLA_PRECISION_REQUESTS",
            '{"tight": 20000, "normal": 2000, "coarse": 500}',
        )
        settings = _SearchPlannerSettings()
        assert settings.SLA_PRECISION_REQUESTS == {
            "tight": 20000,
            "normal": 2000,
            "coarse": 500,
        }

    @pytest.mark.parametrize(
        "env_var,bad_value",
        [
            param(
                "AIPERF_SEARCH_PLANNER_SLA_PRECISION_DEFAULT",
                "0.0",
                id="precision_zero",
            ),
            param(
                "AIPERF_SEARCH_PLANNER_SLA_PRECISION_DEFAULT",
                "1.0",
                id="precision_one",
            ),
            param(
                "AIPERF_SEARCH_PLANNER_DEFAULT_WARMUP_SECONDS",
                "-1.0",
                id="warmup_negative",
            ),
        ],
    )
    def test_search_planner_out_of_range_raises_value_error(
        self, env_var, bad_value, monkeypatch
    ) -> None:
        monkeypatch.setenv(env_var, bad_value)
        with pytest.raises(ValueError):
            _SearchPlannerSettings()

    def test_environment_search_planner_subsystem_exists(self) -> None:
        env = _Environment()
        assert hasattr(env, "SEARCH_PLANNER")
        assert isinstance(env.SEARCH_PLANNER, _SearchPlannerSettings)
