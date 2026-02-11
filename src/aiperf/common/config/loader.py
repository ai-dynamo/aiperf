# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loader for AIPerf."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson
from ruamel.yaml import YAML

if TYPE_CHECKING:
    from aiperf.common.config.service_config import ServiceConfig
    from aiperf.common.config.user_config import UserConfig


def _load_config_file(path: Path) -> dict[str, Any]:
    """Load a configuration file (JSON or YAML) and return its contents as a dict."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, "rb") as f:
            data = orjson.loads(f.read())
    elif suffix in (".yaml", ".yml"):
        yaml = YAML(pure=True)
        with open(path) as f:
            data = yaml.load(f)
    else:
        raise ValueError(
            f"Unsupported config file format: {suffix}. Use .json, .yaml, or .yml"
        )

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping/object: {path}")

    return data


def load_service_config(path: Path | None = None) -> ServiceConfig:
    """Load the service configuration from a file or environment variable.

    The configuration file path is resolved in this order:
    1. Explicit path argument (if provided)
    2. AIPERF_CONFIG_SERVICE_FILE environment variable
    3. Default ServiceConfig() if neither is set

    Args:
        path: Optional explicit path to the service config file.

    Returns:
        ServiceConfig instance.

    Raises:
        ValidationError: If the service configuration file is invalid.
        FileNotFoundError: If the file does not exist.
    """
    from aiperf.common.config.service_config import ServiceConfig
    from aiperf.common.environment import Environment

    config_path = path or Environment.CONFIG.SERVICE_FILE

    if config_path is not None:
        data = _load_config_file(config_path)
        return ServiceConfig(**data)

    return ServiceConfig()


def load_user_config(path: Path | None = None) -> UserConfig:
    """Load the user configuration from a file or environment variable.

    The configuration file path is resolved in this order:
    1. Explicit path argument (if provided)
    2. AIPERF_CONFIG_USER_FILE environment variable
    3. Raises error if neither is set (user configuration file is required)

    Args:
        path: Optional explicit path to the user configuration file.

    Returns:
        UserConfig instance.

    Raises:
        ValidationError: If the user configuration file is invalid.
        ValueError: If no configuration file path is provided or set via environment variable.
        FileNotFoundError: If the file does not exist.
    """
    from aiperf.common.config.user_config import UserConfig
    from aiperf.common.environment import Environment

    config_path = path or Environment.CONFIG.USER_FILE

    if config_path is not None:
        data = _load_config_file(config_path)
        return UserConfig(**data)

    raise ValueError(
        "User configuration file is required. Provide --user-config-file <path> or set "
        "AIPERF_CONFIG_USER_FILE=<path> environment variable."
    )
