# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
from pytest import param

from aiperf.common.environment import (
    _APIServerSettings,
    _CompressionSettings,
    _Environment,
    _ServiceSettings,
)


class TestServiceSettingsUvloopWindows:
    """Test suite for automatic uvloop disabling on Windows."""

    @pytest.mark.parametrize(
        "platform_name,expected_disable_uvloop",
        [
            param("Windows", True, id="windows_auto_disabled"),
            param("Linux", False, id="linux_enabled"),
            param("Darwin", False, id="macos_enabled"),
        ],
    )
    @patch("aiperf.common.environment.platform.system")
    def test_platform_uvloop_detection(
        self, mock_platform, platform_name, expected_disable_uvloop
    ):
        """Test that uvloop is automatically disabled on Windows and enabled elsewhere."""
        mock_platform.return_value = platform_name

        settings = _ServiceSettings()

        assert settings.DISABLE_UVLOOP is expected_disable_uvloop

    @pytest.mark.parametrize(
        "platform_name,manual_setting,expected_result",
        [
            param("Windows", False, True, id="windows_override_attempt"),
            param("Windows", True, True, id="windows_manual_disable"),
            param("Linux", True, True, id="linux_manual_disable"),
            param("Linux", False, False, id="linux_default_enabled"),
            param("Darwin", True, True, id="macos_manual_disable"),
            param("Darwin", False, False, id="macos_default_enabled"),
        ],
    )
    @patch("aiperf.common.environment.platform.system")
    def test_manual_uvloop_settings(
        self, mock_platform, platform_name, manual_setting, expected_result
    ):
        """Test manual DISABLE_UVLOOP settings across platforms."""
        mock_platform.return_value = platform_name

        settings = _ServiceSettings(DISABLE_UVLOOP=manual_setting)

        assert settings.DISABLE_UVLOOP is expected_result


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
