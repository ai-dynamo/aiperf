# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Pydantic Models

OpenTelemetry - Metrics streaming configuration.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import ConfigDict, Field

from aiperf.config.base import BaseConfig

__all__ = [
    "OTelConfig",
]


class OTelConfig(BaseConfig):
    """OpenTelemetry metrics streaming configuration."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    metrics_url: Annotated[
        str | None,
        Field(default=None, description="OTLP/HTTP metrics endpoint URL."),
    ]
    stream_metrics_enabled: Annotated[
        bool,
        Field(default=True, description="Stream metric records to OTel."),
    ]
    stream_timing_enabled: Annotated[
        bool,
        Field(default=True, description="Stream timing records to OTel."),
    ]
    custom_resource_attributes: Annotated[
        dict[str, str],
        Field(default_factory=dict, description="Custom OTel resource attributes."),
    ]
    gen_ai_provider: Annotated[
        str | None,
        Field(default=None, description="GenAI semantic convention provider override."),
    ]

    @property
    def collector_enabled(self) -> bool:
        """Whether OTel metrics streaming is enabled."""
        return self.metrics_url is not None

    @property
    def stream(self) -> str:
        """Human-readable stream selection for diagnostics."""
        if self.stream_metrics_enabled and self.stream_timing_enabled:
            return "default"
        if self.stream_metrics_enabled:
            return "metrics"
        if self.stream_timing_enabled:
            return "timing"
        return "none"
