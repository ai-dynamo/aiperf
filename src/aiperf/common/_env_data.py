# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data/observability environment settings subgroups.

Private module for :mod:`aiperf.common.environment`. Contains the
``_DatasetSettings``, ``_DeveloperSettings``, ``_GPUSettings``,
``_MetricsSettings``, ``_ServerMetricsSettings``, and ``_UISettings``
classes.
"""

from pathlib import Path
from typing import Annotated

from pydantic import BeforeValidator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from aiperf.config.parsing import (
    parse_service_types,
    parse_str_or_csv_list,
)
from aiperf.plugin.enums import ServiceType


class _DatasetSettings(BaseSettings):
    """Dataset loading and configuration.

    Controls timeouts and behavior for dataset loading operations,
    as well as memory-mapped dataset storage settings.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_DATASET_",
    )

    CONFIGURATION_TIMEOUT: float = Field(
        ge=1.0,
        le=100000.0,
        default=300.0,
        description="Timeout in seconds for dataset configuration operations",
    )
    MMAP_BASE_PATH: Path | None = Field(
        default=None,
        description="Base path for memory-mapped dataset files. If None, uses system temp directory. "
        "Set to a shared filesystem path for Kubernetes mounted volumes. "
        "Example: AIPERF_DATASET_MMAP_BASE_PATH=/mnt/shared-pvc "
        "creates files at /mnt/shared-pvc/aiperf_mmap_{benchmark_id}/",
    )
    DOWNLOAD_MAX_RETRIES: int = Field(
        ge=0,
        le=20,
        default=3,
        description="Maximum number of retries for dataset download in Kubernetes worker pods",
    )
    DOWNLOAD_RETRY_DELAY: float = Field(
        ge=0.1,
        le=60.0,
        default=2.0,
        description="Initial delay in seconds between dataset download retries (doubles each retry)",
    )
    PUBLIC_DATASET_TIMEOUT: float = Field(
        ge=1.0,
        le=100000.0,
        default=300.0,
        description="Timeout in seconds for public dataset loading operations",
    )
    MEDIA_DOWNLOAD_TIMEOUT: float = Field(
        ge=1.0,
        le=100000.0,
        default=60.0,
        description="Timeout in seconds per media URL download when inline encoding is required",
    )
    MEDIA_DOWNLOAD_MAX_CONCURRENCY: int = Field(
        ge=1,
        le=100,
        default=10,
        description="Maximum number of concurrent media URL downloads",
    )


class _DeveloperSettings(BaseSettings):
    """Development and debugging configuration.

    Controls developer-focused features like debug logging, profiling, and internal metrics.
    These settings are typically disabled in production environments.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_DEV_",
    )

    DEBUG_SERVICES: Annotated[
        set[ServiceType] | None,
        BeforeValidator(parse_service_types),
    ] = Field(
        default=None,
        description="List of services to enable DEBUG logging for (comma-separated or multiple flags)",
    )
    ENABLE_YAPPI: bool = Field(
        default=False,
        description="Enable yappi profiling (Yet Another Python Profiler) for performance analysis. "
        "Requires 'uv add yappi snakeviz'",
    )
    MEMORY_PROFILE_ENABLED: bool = Field(
        default=False,
        description="Enable memory profiling using tracemalloc. "
        "Logs memory usage and top allocators periodically.",
    )
    MEMORY_PROFILE_INTERVAL: float = Field(
        ge=1.0,
        le=3600.0,
        default=10.0,
        description="Interval in seconds between memory profile snapshots when profiling is enabled.",
    )
    MEMORY_PROFILE_TOP_N: int = Field(
        ge=1,
        le=100,
        default=10,
        description="Number of top memory allocators to log in each snapshot.",
    )
    MODE: bool = Field(
        default=False,
        description="Enable AIPerf Developer mode for internal metrics and debugging",
    )
    SHOW_EXPERIMENTAL_METRICS: bool = Field(
        default=False,
        description="[Developer use only] Show experimental metrics in output (requires DEV_MODE)",
    )
    SHOW_INTERNAL_METRICS: bool = Field(
        default=False,
        description="[Developer use only] Show internal and hidden metrics in output (requires DEV_MODE)",
    )
    TRACE_SERVICES: Annotated[
        set[ServiceType] | None,
        BeforeValidator(parse_service_types),
    ] = Field(
        default=None,
        description="List of services to enable TRACE logging for (comma-separated or multiple flags)",
    )


class _GPUSettings(BaseSettings):
    """GPU telemetry collection configuration.

    Controls GPU metrics collection frequency, endpoint detection, and shutdown behavior.
    Metrics are collected from DCGM endpoints at the specified interval.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_GPU_",
        env_parse_enums=True,
    )

    COLLECTION_INTERVAL: float = Field(
        ge=0.01,
        le=300.0,
        default=0.333,
        description="GPU telemetry metrics collection interval in seconds (default: 333ms, ~3Hz)",
    )
    DEFAULT_DCGM_ENDPOINTS: Annotated[
        str | list[str],
        BeforeValidator(parse_str_or_csv_list),
    ] = Field(
        default=["http://localhost:9400/metrics", "http://localhost:9401/metrics"],
        description="Default DCGM endpoint URLs to check for GPU telemetry (comma-separated string or JSON array)",
    )
    EXPORT_BATCH_SIZE: int = Field(
        ge=1,
        le=1000000,
        default=100,
        description="Batch size for telemetry record export results processor",
    )
    EXPORT_FLUSH_INTERVAL: float = Field(
        ge=0.1,
        le=300.0,
        default=2.0,
        description="Maximum seconds telemetry JSONL records may remain buffered before being flushed to disk",
    )
    REACHABILITY_TIMEOUT: int = Field(
        ge=1,
        le=300,
        default=10,
        description="Timeout in seconds for checking GPU telemetry endpoint reachability during init",
    )
    SHUTDOWN_DELAY: float = Field(
        ge=1.0,
        le=300.0,
        default=5.0,
        description="Delay in seconds before shutting down GPU telemetry service to allow command response transmission",
    )
    THREAD_JOIN_TIMEOUT: float = Field(
        ge=1.0,
        le=300.0,
        default=5.0,
        description="Timeout in seconds for joining GPU telemetry collection threads during shutdown",
    )


class _MetricsSettings(BaseSettings):
    """Metrics collection and storage configuration.

    Controls metrics storage allocation and collection behavior.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_METRICS_",
    )

    ARRAY_INITIAL_CAPACITY: int = Field(
        ge=100,
        le=1000000,
        default=10000,
        description="Initial array capacity for metric storage dictionaries to minimize reallocation",
    )
    USAGE_PCT_DIFF_THRESHOLD: float = Field(
        ge=0.0,
        le=100.0,
        default=10.0,
        description="Percentage difference threshold for flagging discrepancies between API usage and client token counts (default: 10%)",
    )
    OSL_MISMATCH_PCT_THRESHOLD: float = Field(
        ge=0.0,
        le=100.0,
        default=5.0,
        description="Percentage difference threshold for flagging discrepancies between requested and actual output sequence length (default: 5%)",
    )
    OSL_MISMATCH_MAX_TOKEN_THRESHOLD: int = Field(
        ge=1,
        default=50,
        description="Maximum absolute token threshold for OSL mismatch. The effective threshold is min(requested_osl * pct_threshold, this value). Makes threshold tighter for large OSL values (default: 50 tokens)",
    )


class _ServerMetricsSettings(BaseSettings):
    """Server metrics collection configuration.

    Controls server metrics collection frequency, endpoint detection, and shutdown behavior.
    Metrics are collected from Prometheus-compatible endpoints at the specified interval.
    Use `--no-server-metrics` CLI flag to disable collection.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_SERVER_METRICS_",
        env_parse_enums=True,
    )

    COLLECTION_FLUSH_PERIOD: float = Field(
        ge=0.0,
        le=30.0,
        default=2.0,
        description="Time in seconds to continue collecting metrics after profiling completes, "
        "allowing server-side metrics to flush/finalize before shutting down (default: 2.0s)",
    )
    COLLECTION_INTERVAL: float = Field(
        ge=0.001,
        le=300.0,
        default=0.333,
        description="Server metrics collection interval in seconds (default: 333ms, ~3Hz)",
    )
    EXPORT_BATCH_SIZE: int = Field(
        ge=1,
        le=1000000,
        default=100,
        description="Batch size for server metrics jsonl writer export results processor",
    )
    EXPORT_FLUSH_INTERVAL: float = Field(
        ge=0.1,
        le=300.0,
        default=2.0,
        description="Maximum seconds server metrics JSONL records may remain buffered before being flushed to disk",
    )
    REACHABILITY_TIMEOUT: int = Field(
        ge=1,
        le=300,
        default=10,
        description="Timeout in seconds for checking server metrics endpoint reachability during init",
    )
    SHUTDOWN_DELAY: float = Field(
        ge=1.0,
        le=300.0,
        default=5.0,
        description="Delay in seconds before shutting down server metrics service to allow command response transmission",
    )


class _UISettings(BaseSettings):
    """User interface and dashboard configuration.

    Controls refresh rates, update thresholds, and notification behavior for the
    various UI modes (dashboard, tqdm, etc.).
    """

    model_config = SettingsConfigDict(
        env_prefix="AIPERF_UI_",
    )

    LOG_REFRESH_INTERVAL: float = Field(
        ge=0.01,
        le=100000.0,
        default=0.1,
        description="Log viewer refresh interval in seconds (default: 10 FPS)",
    )
    MIN_UPDATE_PERCENT: float = Field(
        ge=0.01,
        le=100.0,
        default=1.0,
        description="Minimum percentage difference from last update to trigger a UI update (for non-dashboard UIs)",
    )
    NOTIFICATION_TIMEOUT: int = Field(
        ge=1,
        le=100000,
        default=3,
        description="Duration in seconds to display UI notifications before auto-dismissing",
    )
    REALTIME_METRICS_INTERVAL: float = Field(
        ge=1.0,
        le=1000.0,
        default=5.0,
        description="Interval in seconds between real-time metrics messages",
    )
    REALTIME_METRICS_ENABLED: bool = Field(
        default=False,
        description="Enable real-time metrics collection and reporting despite UI type",
    )
    SPINNER_REFRESH_RATE: float = Field(
        ge=0.1,
        le=100.0,
        default=0.1,
        description="Progress spinner refresh rate in seconds (default: 10 FPS)",
    )
    STATUS_LOG_INTERVAL: float = Field(
        ge=1.0,
        le=3600.0,
        default=30.0,
        description="Interval in seconds between periodic status log messages when using --ui none",
    )
