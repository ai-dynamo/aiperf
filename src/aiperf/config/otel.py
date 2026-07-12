# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Pydantic Models

OpenTelemetry - Metrics streaming configuration.
"""

from __future__ import annotations

from typing import Annotated
from urllib.parse import urlparse, urlunparse

from pydantic import BeforeValidator, ConfigDict, Field

from aiperf.config.base import BaseConfig

__all__ = [
    "OTelConfig",
    "normalize_otel_metrics_url",
]


def normalize_otel_metrics_url(
    value: str | None,
    *,
    field_name: str = "metrics_url",
) -> str | None:
    """Normalize every Config-v2 input path to an OTLP/HTTP metrics URL."""
    if value is None:
        return None
    normalized_url = value.strip()
    if not normalized_url:
        raise ValueError(f"{field_name} cannot be empty.")
    if "://" not in normalized_url:
        normalized_url = f"http://{normalized_url}"

    parsed = urlparse(normalized_url)
    if not parsed.scheme or not parsed.netloc or not parsed.hostname:
        raise ValueError(
            f"Invalid {field_name} value: {value!r}. "
            "Expected host[:port] or a full URL."
        )
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError(
            f"Invalid {field_name} value: {value!r}. "
            f"Only http and https schemes are supported (got {parsed.scheme!r}). "
            "OTLP/gRPC is not supported; use the OTLP/HTTP exporter endpoint."
        )

    path = parsed.path.rstrip("/")
    if path.endswith("/v1/metrics"):
        normalized_path = path
    elif not path:
        normalized_path = "/v1/metrics"
    else:
        normalized_path = f"{path}/v1/metrics"
    return urlunparse(parsed._replace(path=normalized_path))


class OTelConfig(BaseConfig):
    """OpenTelemetry metrics streaming configuration."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    metrics_url: Annotated[
        str | None,
        BeforeValidator(normalize_otel_metrics_url),
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
