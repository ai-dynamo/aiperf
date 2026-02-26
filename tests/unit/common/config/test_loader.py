# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for config loader functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from pytest import param

from aiperf.common.config.loader import (
    _load_config_file,
    load_service_config,
    load_user_config,
)
from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.config.user_config import UserConfig

if TYPE_CHECKING:
    from collections.abc import Generator

# Minimal valid UserConfig payload (endpoint is required)
_MINIMAL_USER_CONFIG = {
    "endpoint": {
        "model_names": ["test-model"],
        "type": "chat",
        "custom_endpoint": "test",
    }
}


@pytest.fixture
def _no_env_config_files() -> Generator[None, None, None]:
    """Ensure Environment.CONFIG file paths are None for test isolation."""
    with patch("aiperf.common.environment.Environment.CONFIG") as mock_config:
        mock_config.SERVICE_FILE = None
        mock_config.USER_FILE = None
        yield


def _write_user_config_json(path: Path) -> Path:
    """Write a minimal valid UserConfig JSON file and return the path."""
    path.write_text(json.dumps(_MINIMAL_USER_CONFIG))
    return path


def _write_user_config_yaml(path: Path) -> Path:
    """Write a minimal valid UserConfig YAML file and return the path."""
    path.write_text(
        "endpoint:\n"
        "  model_names:\n"
        "    - test-model\n"
        "  type: chat\n"
        "  custom_endpoint: test\n"
    )
    return path


class TestLoadConfigFile:
    """Tests for _load_config_file()."""

    def test_json_file(self, tmp_path: Path) -> None:
        """Test loading a valid JSON config file."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"key": "value"}')

        result = _load_config_file(config_file)

        assert result == {"key": "value"}

    def test_yaml_file(self, tmp_path: Path) -> None:
        """Test loading a valid YAML config file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("key: value\n")

        result = _load_config_file(config_file)

        assert result == {"key": "value"}

    def test_yml_extension(self, tmp_path: Path) -> None:
        """Test loading a valid .yml config file."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("key: value\n")

        result = _load_config_file(config_file)

        assert result == {"key": "value"}

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """Test that a missing file raises FileNotFoundError."""
        missing = tmp_path / "missing.json"

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            _load_config_file(missing)

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        """Test that an unsupported file extension raises ValueError."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("")

        with pytest.raises(ValueError, match="Unsupported config file format"):
            _load_config_file(config_file)

    def test_empty_yaml_returns_empty_dict(self, tmp_path: Path) -> None:
        """Test that an empty YAML file returns an empty dict."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")

        result = _load_config_file(config_file)

        assert result == {}

    def test_empty_json_object_returns_empty_dict(self, tmp_path: Path) -> None:
        """Test that an empty JSON object returns an empty dict."""
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")

        result = _load_config_file(config_file)

        assert result == {}

    def test_non_mapping_json_raises(self, tmp_path: Path) -> None:
        """Test that a JSON array raises ValueError."""
        config_file = tmp_path / "config.json"
        config_file.write_text("[1, 2, 3]")

        with pytest.raises(ValueError, match="must contain a mapping/object"):
            _load_config_file(config_file)

    def test_non_mapping_yaml_raises(self, tmp_path: Path) -> None:
        """Test that a YAML list raises ValueError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("- item1\n- item2\n")

        with pytest.raises(ValueError, match="must contain a mapping/object"):
            _load_config_file(config_file)

    @pytest.mark.parametrize(
        "suffix",
        [
            param(".yaml", id="yaml"),
            param(".yml", id="yml"),
        ],
    )
    def test_case_insensitive_extension(self, tmp_path: Path, suffix: str) -> None:
        """Test that file extension matching is case-insensitive."""
        config_file = tmp_path / f"config{suffix.upper()}"
        config_file.write_text("key: value\n")

        result = _load_config_file(config_file)

        assert result == {"key": "value"}


class TestLoadServiceConfig:
    """Tests for load_service_config()."""

    def test_loads_from_json_file(self, tmp_path: Path) -> None:
        """Test loading ServiceConfig from a JSON file."""
        config_file = tmp_path / "service.json"
        config_file.write_text("{}")

        result = load_service_config(config_file)

        assert isinstance(result, ServiceConfig)

    def test_loads_from_yaml_file(self, tmp_path: Path) -> None:
        """Test loading ServiceConfig from a YAML file."""
        config_file = tmp_path / "service.yaml"
        config_file.write_text("---\n")

        result = load_service_config(config_file)

        assert isinstance(result, ServiceConfig)

    @pytest.mark.usefixtures("_no_env_config_files")
    def test_none_path_returns_default(self) -> None:
        """Test that None path with no env var returns default ServiceConfig."""
        result = load_service_config(None)

        assert isinstance(result, ServiceConfig)

    def test_falls_back_to_env_var(self, tmp_path: Path) -> None:
        """Test that None path falls back to AIPERF_CONFIG_SERVICE_FILE env var."""
        config_file = tmp_path / "service.json"
        config_file.write_text("{}")

        with patch("aiperf.common.environment.Environment.CONFIG") as mock_config:
            mock_config.SERVICE_FILE = config_file
            result = load_service_config(None)

        assert isinstance(result, ServiceConfig)

    def test_explicit_path_takes_precedence_over_env(self, tmp_path: Path) -> None:
        """Test that an explicit path argument takes precedence over env var."""
        explicit_file = tmp_path / "explicit.json"
        explicit_file.write_text('{"log_level": "DEBUG"}')
        env_file = tmp_path / "env.json"
        env_file.write_text('{"log_level": "TRACE"}')

        with patch("aiperf.common.environment.Environment.CONFIG") as mock_config:
            mock_config.SERVICE_FILE = env_file
            result = load_service_config(explicit_file)

        assert result.log_level == "DEBUG"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Test that a missing file raises FileNotFoundError."""
        missing = tmp_path / "missing.json"

        with pytest.raises(FileNotFoundError):
            load_service_config(missing)


class TestLoadUserConfig:
    """Tests for load_user_config()."""

    def test_loads_from_json_file(self, tmp_path: Path) -> None:
        """Test loading UserConfig from a JSON file."""
        config_file = _write_user_config_json(tmp_path / "user.json")

        result = load_user_config(config_file)

        assert isinstance(result, UserConfig)

    def test_loads_from_yaml_file(self, tmp_path: Path) -> None:
        """Test loading UserConfig from a YAML file."""
        config_file = _write_user_config_yaml(tmp_path / "user.yaml")

        result = load_user_config(config_file)

        assert isinstance(result, UserConfig)

    @pytest.mark.usefixtures("_no_env_config_files")
    def test_none_path_no_env_var_raises(self) -> None:
        """Test that None path with no env var raises ValueError."""
        with pytest.raises(ValueError, match="User configuration file is required"):
            load_user_config(None)

    def test_falls_back_to_env_var(self, tmp_path: Path) -> None:
        """Test that None path falls back to AIPERF_CONFIG_USER_FILE env var."""
        config_file = _write_user_config_json(tmp_path / "user.json")

        with patch("aiperf.common.environment.Environment.CONFIG") as mock_config:
            mock_config.USER_FILE = config_file
            result = load_user_config(None)

        assert isinstance(result, UserConfig)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Test that a missing file raises FileNotFoundError."""
        missing = tmp_path / "missing.json"

        with pytest.raises(FileNotFoundError):
            load_user_config(missing)
