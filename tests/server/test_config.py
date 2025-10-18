# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for config module."""

import os

import pytest
from aiperf_mock_server.config import (
    MockServerConfig,
    _get_env_key,
    _propagate_config_to_env,
    _serialize_env_value,
    set_server_config,
)
from pydantic import ValidationError


class TestMockServerConfig:
    """Tests for MockServerConfig class."""

    def test_default_config(self):
        config = MockServerConfig()
        assert config.port == 8000
        assert config.host == "127.0.0.1"
        assert config.workers == 1
        assert config.ttft == 20.0
        assert config.itl == 5.0
        assert config.log_level == "INFO"
        assert config.verbose is False
        assert config.error_rate == 0.0

    def test_custom_config(self):
        config = MockServerConfig(
            port=9000,
            host="0.0.0.0",
            ttft=10.0,
            itl=2.0,
            log_level="DEBUG",
        )
        assert config.port == 9000
        assert config.host == "0.0.0.0"
        assert config.ttft == 10.0
        assert config.itl == 2.0
        assert config.log_level == "DEBUG"

    def test_verbose_sets_debug(self):
        config = MockServerConfig(verbose=True)
        assert config.log_level == "DEBUG"

    @pytest.mark.parametrize(
        "field,invalid_values",
        [
            ("port", [0, 70000]),
            ("workers", [0, 50]),
            ("ttft", [-1.0]),
            ("itl", [-1.0]),
            ("error_rate", [-1.0, 101.0]),
            ("dcgm_num_gpus", [0, 10]),
            ("dcgm_initial_load", [-0.1, 1.5]),
        ],
    )
    def test_field_validation(self, field, invalid_values):
        for value in invalid_values:
            with pytest.raises(ValidationError):
                MockServerConfig(**{field: value})

    @pytest.mark.parametrize(
        "log_level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
    def test_valid_log_levels(self, log_level):
        config = MockServerConfig(log_level=log_level)
        assert config.log_level == log_level

    def test_dcgm_defaults(self):
        config = MockServerConfig()
        assert config.dcgm_gpu_name == "h200"
        assert config.dcgm_num_gpus == 2
        assert config.dcgm_initial_load == 0.7
        assert config.dcgm_hostname == "localhost"
        assert config.dcgm_seed is None


class TestConfigHelpers:
    """Tests for config helper functions."""

    @pytest.mark.parametrize(
        "field,expected",
        [
            ("port", "MOCK_SERVER_PORT"),
            ("ttft", "MOCK_SERVER_TTFT"),
            ("log_level", "MOCK_SERVER_LOG_LEVEL"),
        ],
    )
    def test_get_env_key(self, field, expected):
        assert _get_env_key(field) == expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("test", "test"),
            (123, "123"),
            (1.5, "1.5"),
            (True, "True"),
        ],
    )
    def test_serialize_env_value(self, value, expected):
        assert _serialize_env_value(value) == expected

    def test_serialize_env_value_list(self):
        result = _serialize_env_value(["a", "b"])
        assert result == '["a", "b"]'

    def test_serialize_env_value_dict(self):
        result = _serialize_env_value({"key": "value"})
        assert "key" in result
        assert "value" in result

    def test_propagate_config_to_env(self, monkeypatch):
        monkeypatch.delenv("MOCK_SERVER_PORT", raising=False)
        monkeypatch.delenv("MOCK_SERVER_TTFT", raising=False)

        config = MockServerConfig(port=9000, ttft=15.0)
        _propagate_config_to_env(config)

        assert os.environ["MOCK_SERVER_PORT"] == "9000"
        assert os.environ["MOCK_SERVER_TTFT"] == "15.0"

    def test_set_server_config(self, monkeypatch):
        from aiperf_mock_server import config as config_module

        monkeypatch.delenv("MOCK_SERVER_PORT", raising=False)

        new_config = MockServerConfig(port=9999)
        set_server_config(new_config)

        assert config_module.server_config.port == 9999
        assert os.environ["MOCK_SERVER_PORT"] == "9999"


class TestEnvironmentVariables:
    """Tests for environment variable loading."""

    def test_load_from_env(self, monkeypatch):
        monkeypatch.setenv("MOCK_SERVER_PORT", "7000")
        monkeypatch.setenv("MOCK_SERVER_TTFT", "30.0")

        config = MockServerConfig()
        assert config.port == 7000
        assert config.ttft == 30.0

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("mock_server_port", "6000")
        config = MockServerConfig()
        assert config.port == 6000

    def test_env_has_priority_over_defaults(self, monkeypatch):
        # In pydantic-settings, environment variables have higher priority
        # than the values passed to the constructor
        monkeypatch.setenv("MOCK_SERVER_PORT", "5000")
        config = MockServerConfig()
        assert config.port == 5000
