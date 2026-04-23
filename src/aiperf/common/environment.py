# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Environment Configuration Module

Provides a hierarchical, type-safe configuration system using Pydantic BaseSettings.
All settings can be configured via environment variables with the AIPERF_ prefix.

Structure:
    Environment.API_SERVER.*     - API server settings
    Environment.COMPRESSION.*    - Compression settings for streaming file transfers
    Environment.CONFIG.*         - Configuration file paths for distributed deployments
    Environment.DATASET.*        - Dataset management
    Environment.DEV.*            - Development and debugging settings
    Environment.GPU.*            - GPU telemetry collection
    Environment.HTTP.*           - HTTP client socket and connection settings
    Environment.LOGGING.*        - Logging configuration
    Environment.METRICS.*        - Metrics collection and storage
    Environment.RECORD.*         - Record processing
    Environment.SERVER_METRICS.* - Server metrics collection
    Environment.SERVICE.*        - Service lifecycle and communication
    Environment.TIMING.*         - Timing manager settings
    Environment.UI.*             - User interface settings
    Environment.WORKER.*         - Worker management and scaling
    Environment.ZMQ.*            - ZMQ communication settings

Examples:
    # Via environment variables:
    AIPERF_HTTP_SO_RCVBUF=20971520
    AIPERF_WORKER_CPU_UTILIZATION_FACTOR=0.8

    # In code:
    print(f"Buffer: {Environment.HTTP.SO_RCVBUF}")
    print(f"Workers: {Environment.WORKER.CPU_UTILIZATION_FACTOR}")

See also: ``aiperf.kubernetes.environment.K8sEnvironment`` (K8s-specific cluster
defaults) and ``aiperf.operator.environment.OperatorEnvironment`` (operator-process
tunables).

The individual ``_XxxSettings`` subgroup classes live in sibling private
modules (``_env_network``, ``_env_services``, ``_env_data``) to keep this
file focused on the top-level ``_Environment`` model.
"""

# ``platform`` is re-imported here (in addition to the call site inside
# ``_env_services``) because existing tests patch
# ``aiperf.common.environment.platform.system``. Since ``platform`` is a
# singleton module, patching it through either attribute path affects the
# call in ``_ServiceSettings.auto_disable_uvloop_on_windows``.
import platform  # noqa: F401

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self

from aiperf.common._env_data import (
    _DatasetSettings,
    _DeveloperSettings,
    _GPUSettings,
    _MetricsSettings,
    _ServerMetricsSettings,
    _UISettings,
)
from aiperf.common._env_network import (
    _APIServerSettings,
    _CompressionSettings,
    _HTTPSettings,
    _LoggingSettings,
    _ZMQSettings,
)
from aiperf.common._env_services import (
    _ConfigSettings,
    _RecordSettings,
    _ServiceSettings,
    _TimingSettings,
    _WorkerSettings,
)
from aiperf.common.aiperf_logger import AIPerfLogger

_logger = AIPerfLogger(__name__)

__all__ = ["Environment"]


class _Environment(BaseSettings):
    """
    Root environment configuration with nested subsystem settings.

    This is a singleton instance that loads configuration from environment variables
    with the AIPERF_ prefix. Settings are organized into logical subsystems for
    better discoverability and maintainability.

    All nested settings can be configured via environment variables using the pattern:
    AIPERF_{SUBSYSTEM}_{SETTING_NAME}

    Example:
        AIPERF_HTTP_CONNECTION_LIMIT=5000
        AIPERF_WORKER_CPU_UTILIZATION_FACTOR=0.8
        AIPERF_ZMQ_RCVTIMEO=600000
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    # Nested subsystem settings (alphabetically ordered)
    API_SERVER: _APIServerSettings = Field(
        default_factory=_APIServerSettings,
        description="API server settings",
    )
    COMPRESSION: _CompressionSettings = Field(
        default_factory=_CompressionSettings,
        description="Compression settings for streaming file transfers",
    )
    CONFIG: _ConfigSettings = Field(
        default_factory=_ConfigSettings,
        description="Configuration file paths for distributed deployments",
    )
    DATASET: _DatasetSettings = Field(
        default_factory=_DatasetSettings,
        description="Dataset loading and configuration settings",
    )
    DEV: _DeveloperSettings = Field(
        default_factory=_DeveloperSettings,
        description="Development and debugging settings",
    )
    GPU: _GPUSettings = Field(
        default_factory=_GPUSettings,
        description="GPU telemetry collection settings",
    )
    HTTP: _HTTPSettings = Field(
        default_factory=_HTTPSettings,
        description="HTTP client socket and connection settings",
    )
    LOGGING: _LoggingSettings = Field(
        default_factory=_LoggingSettings,
        description="Logging system settings",
    )
    METRICS: _MetricsSettings = Field(
        default_factory=_MetricsSettings,
        description="Metrics collection and storage settings",
    )
    RECORD: _RecordSettings = Field(
        default_factory=_RecordSettings,
        description="Record processing and export settings",
    )
    SERVER_METRICS: _ServerMetricsSettings = Field(
        default_factory=_ServerMetricsSettings,
        description="Server metrics collection settings",
    )
    SERVICE: _ServiceSettings = Field(
        default_factory=_ServiceSettings,
        description="Service lifecycle and communication settings",
    )
    TIMING: _TimingSettings = Field(
        default_factory=_TimingSettings,
        description="Timing manager settings",
    )
    UI: _UISettings = Field(
        default_factory=_UISettings,
        description="User interface and dashboard settings",
    )
    WORKER: _WorkerSettings = Field(
        default_factory=_WorkerSettings,
        description="Worker management and scaling settings",
    )
    ZMQ: _ZMQSettings = Field(
        default_factory=_ZMQSettings,
        description="ZMQ communication settings",
    )

    @model_validator(mode="after")
    def validate_dev_mode(self) -> Self:
        """Validate that developer mode is enabled for features that require it."""
        if self.DEV.SHOW_INTERNAL_METRICS and not self.DEV.MODE:
            _logger.warning(
                "Developer mode is not enabled, disabling AIPERF_DEV_SHOW_INTERNAL_METRICS"
            )
            self.DEV.SHOW_INTERNAL_METRICS = False

        if self.DEV.SHOW_EXPERIMENTAL_METRICS and not self.DEV.MODE:
            _logger.warning(
                "Developer mode is not enabled, disabling AIPERF_DEV_SHOW_EXPERIMENTAL_METRICS"
            )
            self.DEV.SHOW_EXPERIMENTAL_METRICS = False

        return self

    @model_validator(mode="after")
    def validate_profile_configure_timeout(self) -> Self:
        """Validate that the profile configure timeout is at least as long as the dataset configuration timeout."""
        if self.SERVICE.PROFILE_CONFIGURE_TIMEOUT < self.DATASET.CONFIGURATION_TIMEOUT:
            raise ValueError(
                f"AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT: {self.SERVICE.PROFILE_CONFIGURE_TIMEOUT} must be greater than or equal to AIPERF_DATASET_CONFIGURATION_TIMEOUT: {self.DATASET.CONFIGURATION_TIMEOUT}"
            )
        return self


# Global singleton instance
Environment: _Environment = _Environment()
