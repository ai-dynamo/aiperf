# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Pydantic Models

GPU Telemetry - Live or replayed GPU metrics collection configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from pydantic import ConfigDict, Field, model_validator

from aiperf.common.enums import GPUTelemetryMode
from aiperf.config.base import BaseConfig
from aiperf.plugin.enums import GPUTelemetryCollectorType

__all__ = [
    "GpuTelemetryConfig",
]


class GpuTelemetryConfig(BaseConfig):
    """
    GPU telemetry configuration for live or replayed GPU metrics collection.

    Collects GPU metrics through DCGM exporter endpoints by default; the
    collector field can switch collection to local PyNVML, and mode controls
    summary vs. realtime dashboard display.

    Accepts shorthand forms:
        - String URL: "http://localhost:9400/metrics"
          → GpuTelemetryConfig(enabled=True, urls=["http://localhost:9400/metrics"])
        - Singular url field: {url: "..."}
          → GpuTelemetryConfig(urls=["..."])
    """

    # x-kubernetes-preserve-unknown-fields lets apiserver accept the
    # string-URL shorthand (collapsed to the full object form by
    # normalize_before_validation) which a Kubernetes structural schema
    # cannot express as a string|object union.
    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        json_schema_extra={"x-kubernetes-preserve-unknown-fields": True},
    )

    enabled: Annotated[
        bool,
        Field(
            default=True,
            description="Enable GPU telemetry collection. Set to false to disable.",
        ),
    ]

    urls: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="DCGM exporter endpoint URLs. "
            "Example: http://localhost:9400/metrics",
        ),
    ]

    metrics_file: Annotated[
        Path | None,
        Field(
            default=None,
            description="Path to CSV file with pre-recorded GPU metrics. "
            "Alternative to live DCGM collection.",
        ),
    ]

    collector: Annotated[
        GPUTelemetryCollectorType,
        Field(
            default=GPUTelemetryCollectorType.DCGM,
            description="GPU telemetry collector backend. Use 'dcgm' for DCGM exporter endpoints or 'pynvml' for local PyNVML collection.",
        ),
    ]

    mode: Annotated[
        GPUTelemetryMode,
        Field(
            default=GPUTelemetryMode.SUMMARY,
            description="GPU telemetry display mode. Summary emits aggregate console output; realtime_dashboard enables live dashboard updates.",
        ),
    ]

    @model_validator(mode="before")
    @classmethod
    def normalize_before_validation(cls, data: Any) -> Any:
        """Normalize shorthand forms before validation.

        Handles:
            - String URL → full config dict with that URL
            - url → urls (singular to plural)
        """
        # String URL → full config with that URL
        if isinstance(data, str):
            return {"enabled": True, "urls": [data]}

        if not isinstance(data, dict):
            return data

        # url → urls (singular to plural)
        if "url" in data and "urls" not in data:
            url = data.pop("url")
            data["urls"] = [url] if isinstance(url, str) else url

        return data
