# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service/runtime environment settings subgroups.

Private module for :mod:`aiperf.common.environment`. Contains the
``_ConfigSettings``, ``_RecordSettings``, ``_ServiceSettings``,
``_TimingSettings``, and ``_WorkerSettings`` classes.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.environment import _RecordSettings as _RecordSettings
from aiperf.common.environment import _ServiceSettings as _ServiceSettings
from aiperf.common.environment import _TimingSettings as _TimingSettings
from aiperf.common.environment import _WorkerSettings as _WorkerSettings

_logger = AIPerfLogger(__name__)


class _ConfigSettings(BaseSettings):
    """Configuration file paths for distributed deployments.

    Controls paths to configuration files loaded by services running in containers.
    These are primarily used by `aiperf service` when running in Kubernetes.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_CONFIG_",
    )

    SERVICE_FILE: Path | None = Field(
        default=None,
        description="Path to service configuration JSON/YAML file. "
        "Default: /etc/aiperf/service_config.json in Kubernetes deployments.",
    )
    USER_FILE: Path | None = Field(
        default=None,
        description="Path to user configuration JSON/YAML file. "
        "Default: /etc/aiperf/user_config.json in Kubernetes deployments.",
    )
