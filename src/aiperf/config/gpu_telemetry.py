# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Pydantic Models

GPU Telemetry - Live or replayed GPU metrics collection configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Self

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

    Collects GPU metrics through DCGM exporter endpoints by default. The
    ``collector`` field can switch to a local backend (``pynvml`` for NVIDIA,
    ``amdsmi`` for AMD ROCm); ``mode`` controls summary vs. realtime dashboard
    display.

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
            description="GPU telemetry collector backend. Use 'dcgm' for DCGM "
            "exporter endpoints or a local collector (e.g. 'pynvml' for NVIDIA, "
            "'amdsmi' for AMD ROCm) for on-host metrics collection.",
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

    @model_validator(mode="after")
    def validate_collector_compatibility(self) -> Self:
        """Enforce local-collector invariants driven by plugin metadata.

        - Local collectors (``is_local: true`` in plugin metadata) cannot be
          paired with DCGM URLs — they collect on the local host and the two
          modes are mutually exclusive.
        - For local collectors, defer to the collector class's
          ``validate_environment`` classmethod so native-binding probes live
          alongside the implementation that needs them.
        """
        if not self.enabled:
            return self

        from aiperf.plugin import plugins
        from aiperf.plugin.enums import PluginType

        meta = plugins.get_gpu_telemetry_collector_metadata(self.collector)
        if not meta.is_local:
            return self

        if self.urls:
            raise ValueError(
                f"Cannot use {self.collector} with DCGM URLs. Use either "
                f"'{self.collector}' for local GPU monitoring or URLs for "
                "DCGM endpoints, not both."
            )

        collector_cls = plugins.get_class(
            PluginType.GPU_TELEMETRY_COLLECTOR, str(self.collector)
        )
        try:
            collector_cls.validate_environment()
        except RuntimeError as e:
            raise ValueError(str(e)) from e

        return self
