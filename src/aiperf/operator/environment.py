# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator environment configuration.

All settings can be configured via environment variables with the AIPERF_OPERATOR_ prefix,
or AIPERF_ for shared settings (results dir, default image).

Examples:
    AIPERF_OPERATOR_MONITOR_INTERVAL=10.0
    AIPERF_RESULTS_DIR=/data
    AIPERF_DEFAULT_IMAGE=nvcr.io/nvidia/aiperf:latest

See also: ``aiperf.kubernetes.environment.K8sEnvironment`` (cluster defaults
baked into pod manifests) and ``aiperf.common.environment.Environment``
(shared AIPerf runtime).
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "OperatorEnvironment",
]


class _MonitorSettings(BaseSettings):
    """Timer settings for the kopf monitor handler."""

    model_config = SettingsConfigDict(env_prefix="AIPERF_OPERATOR_MONITOR_")

    INTERVAL: float = Field(
        default=10.0,
        gt=0,
        le=3600,
        description="Seconds between progress checks",
    )
    INITIAL_DELAY: float = Field(
        default=5.0,
        ge=0,
        le=300,
        description="Seconds before first progress check after job creation",
    )


class _ResultsSettings(BaseSettings):
    """Results fetching and storage settings."""

    model_config = SettingsConfigDict(env_prefix="AIPERF_RESULTS_")

    DIR: Path = Field(
        default=Path("/data"),
        description="Base directory for storing benchmark results (mounted PVC)",
    )
    MAX_RETRIES: int = Field(
        default=5,
        ge=0,
        le=50,
        description="Max retries when fetching results from controller",
    )
    RETRY_DELAY: float = Field(
        default=2.0,
        ge=0,
        le=60,
        description="Seconds between result fetch retries",
    )
    TTL_DAYS: int = Field(
        default=30,
        ge=0,
        le=3650,
        description="Days to keep results before cleanup (0 = never clean)",
    )
    COMPRESS_ON_DISK: bool = Field(
        default=True,
        description="Store downloaded result files as zstd-compressed (.zst) on disk",
    )


class _OperatorEnvironment(BaseSettings):
    """Root operator environment configuration.

    Loads from environment variables. Nested settings use their own prefixes.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    DEFAULT_IMAGE: str = Field(
        default="nvcr.io/nvidia/aiperf:latest",
        description="Default container image for benchmark jobs",
    )
    JOB_TIMEOUT_SECONDS: float = Field(
        default=0,
        ge=0,
        description="Job timeout in seconds (0 = no timeout)",
    )
    POD_RESTART_THRESHOLD: int = Field(
        default=3,
        ge=0,
        le=100,
        description="Pod restart count before emitting a warning event",
    )
    ENDPOINT_CHECK_TIMEOUT: float = Field(
        default=10.0,
        gt=0,
        le=300,
        description="Seconds to wait for endpoint health check",
    )
    PREFLIGHT_TIMEOUT: float = Field(
        default=30.0,
        gt=0,
        le=120,
        description="Seconds to wait for all pre-flight checks to complete",
    )
    CONFIGMAP_PROPAGATION_DELAY_SECONDS: float = Field(
        default=10.0,
        ge=0,
        le=60,
        description="Seconds to wait after creating the benchmark ConfigMap before creating "
        "the JobSet. Allows kubelet caches on worker nodes to sync the ConfigMap before "
        "pods start mounting it, preventing FailedMount races on first deployment with "
        "a freshly pulled image.",
    )

    MONITOR: _MonitorSettings = Field(
        default_factory=_MonitorSettings,
        description="Monitor timer settings",
    )
    RESULTS: _ResultsSettings = Field(
        default_factory=_ResultsSettings,
        description="Results fetching and storage settings",
    )


OperatorEnvironment = _OperatorEnvironment()
