# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OTel GenAI semantic convention mapping for AIPerf metrics.

Single source of truth for translating AIPerf internal metric names, units,
and attributes onto the OTel GenAI semantic conventions
(https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/).

This module produces pure data (strings, floats, tuples). It does NOT import
opentelemetry.* or mlflow.* — it must remain import-safe in environments
without the optional aiperf[otel] or aiperf[mlflow] extras.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from aiperf.common.config.user_config import UserConfig
    from aiperf.common.messages.inference_messages import MetricRecordsData


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GenAISemconvEmission:
    """Payload produced by translate() when a spec equivalent exists."""

    spec_metric_name: str
    """Spec-defined metric name, e.g. 'gen_ai.client.operation.duration'."""

    instrument_kind: str
    """OTel instrument kind — 'histogram' for all current spec metrics."""

    unit: str
    """Spec-defined unit, e.g. 's', '{token}'."""

    value: float
    """Already unit-converted value (e.g. ns -> s)."""

    attributes: Mapping[str, str | int]
    """Spec-required + aiperf-specific attributes merged."""

    explicit_bucket_boundaries: tuple[float, ...]
    """Spec-defined histogram bucket boundaries."""


# ---------------------------------------------------------------------------
# Unit converters
# ---------------------------------------------------------------------------


def _ns_to_s(ns: float) -> float:
    """Convert nanoseconds to seconds."""
    return ns * 1e-9


def _identity(v: float) -> float:
    return v


UNIT_CONVERTERS: dict[str, Callable[[float], float]] = {
    "request_latency": _ns_to_s,
    "time_to_first_token": _ns_to_s,
    "inter_token_latency": _ns_to_s,
    "input_token_count": _identity,
    "output_token_count": _identity,
}
"""Per-aiperf-metric converter. Duration metrics convert ns -> s; token counts are identity."""


def convert_metric_value(metric_name: str, value: float) -> float:
    """Convert an AIPerf metric value to its spec unit (e.g. ns -> s).

    Returns the value unchanged if no converter is registered for the metric.
    """
    converter = UNIT_CONVERTERS.get(metric_name, _identity)
    return converter(value)


# ---------------------------------------------------------------------------
# Bucket boundaries (from OTel GenAI spec)
# ---------------------------------------------------------------------------

_DURATION_BUCKET_BOUNDARIES: tuple[float, ...] = (
    0.01,
    0.02,
    0.04,
    0.08,
    0.16,
    0.32,
    0.64,
    1.28,
    2.56,
    5.12,
    10.24,
    20.48,
    40.96,
    81.92,
)

_TTFT_BUCKET_BOUNDARIES: tuple[float, ...] = (
    0.001,
    0.005,
    0.01,
    0.02,
    0.04,
    0.06,
    0.08,
    0.1,
    0.12,
    0.14,
    0.16,
    0.18,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.75,
    1.0,
    2.0,
    5.0,
)

_TIME_PER_OUTPUT_CHUNK_BUCKET_BOUNDARIES: tuple[float, ...] = (
    0.001,
    0.005,
    0.01,
    0.02,
    0.04,
    0.06,
    0.08,
    0.1,
    0.12,
    0.14,
    0.16,
    0.18,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.75,
    1.0,
    2.0,
    5.0,
)

_TOKEN_USAGE_BUCKET_BOUNDARIES: tuple[float, ...] = (
    1,
    4,
    16,
    64,
    256,
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
    4194304,
    16777216,
)


# ---------------------------------------------------------------------------
# Metric name mapping table
# ---------------------------------------------------------------------------

METRIC_NAME_MAP: dict[str, tuple[str, str, tuple[float, ...]]] = {
    "request_latency": (
        "gen_ai.client.operation.duration",
        "s",
        _DURATION_BUCKET_BOUNDARIES,
    ),
    "time_to_first_token": (
        "gen_ai.client.operation.time_to_first_chunk",
        "s",
        _TTFT_BUCKET_BOUNDARIES,
    ),
    "inter_token_latency": (
        "gen_ai.client.operation.time_per_output_chunk",
        "s",
        _TIME_PER_OUTPUT_CHUNK_BUCKET_BOUNDARIES,
    ),
    "input_token_count": (
        "gen_ai.client.token.usage",
        "{token}",
        _TOKEN_USAGE_BUCKET_BOUNDARIES,
    ),
    "output_token_count": (
        "gen_ai.client.token.usage",
        "{token}",
        _TOKEN_USAGE_BUCKET_BOUNDARIES,
    ),
}
"""Maps aiperf metric name -> (spec_metric_name, spec_unit, explicit_bucket_boundaries).

Token usage metrics (input/output) both map to gen_ai.client.token.usage and are
discriminated by the gen_ai.token.type attribute (see TOKEN_USAGE_SPECIAL_CASE).
"""


TOKEN_USAGE_SPECIAL_CASE: Mapping[str, str] = {
    "input_token_count": "input",
    "output_token_count": "output",
}
"""Two aiperf metrics merge into one spec histogram using gen_ai.token.type attribute."""


# ---------------------------------------------------------------------------
# Operation name mapping
# ---------------------------------------------------------------------------

_OPERATION_NAME_MAP: dict[str, str] = {
    "chat": "chat",
    "completions": "text_completion",
    "embeddings": "embeddings",
}


def _map_operation_name(endpoint_type: str) -> str:
    """Map AIPerf endpoint.type to gen_ai.operation.name.

    Falls back to 'chat' for unknown endpoint types (documented behaviour).
    """
    return _OPERATION_NAME_MAP.get(endpoint_type.lower(), "chat")


# ---------------------------------------------------------------------------
# Error type classification
# ---------------------------------------------------------------------------

_ERROR_TYPE_MAP: dict[str, str] = {
    "timeout": "timeout",
    "asyncio.TimeoutError": "timeout",
    "TimeoutError": "timeout",
    "cancelled": "cancelled",
}


def _classify_error_type(error: Any | None) -> str | None:
    """Classify an ErrorDetails into a spec error.type value.

    Returns None when there is no error.
    """
    if error is None:
        return None

    code = getattr(error, "code", None)
    error_type = getattr(error, "type", None)
    cause_chain = getattr(error, "cause_chain", None) or []

    # HTTP status code classification. Guard on isinstance(code, int) because
    # the input type is Any — a non-integer `.code` (e.g. a string "timeout")
    # would raise TypeError on the range comparison, crashing the fanout.
    # bool is a subclass of int but is meaningless as a status code.
    if isinstance(code, int) and not isinstance(code, bool):
        if 500 <= code <= 599:
            return "http_5xx"
        if 400 <= code <= 499:
            return "http_4xx"

    # Check error type string
    if error_type and error_type in _ERROR_TYPE_MAP:
        return _ERROR_TYPE_MAP[error_type]

    # Check cause chain for timeout indicators
    for cause in cause_chain:
        cause_str = str(cause).lower()
        if "timeout" in cause_str:
            return "timeout"
        if "cancel" in cause_str:
            return "cancelled"

    # Check message for parse errors
    message = getattr(error, "message", "") or ""
    if "json" in message.lower() and (
        "parse" in message.lower() or "decode" in message.lower()
    ):
        return "parse_error"

    return "_OTHER"


# ---------------------------------------------------------------------------
# Attribute builders
# ---------------------------------------------------------------------------


def _build_duration_attributes(
    record: MetricRecordsData, user_config: UserConfig
) -> dict[str, Any]:
    """Build spec-required attributes for duration/latency histograms."""
    attrs: dict[str, Any] = {}
    attrs["gen_ai.operation.name"] = _map_operation_name(str(user_config.endpoint.type))
    attrs["gen_ai.provider.name"] = infer_provider_name(user_config)
    if user_config.endpoint.model_names:
        attrs["gen_ai.request.model"] = user_config.endpoint.model_names[0]
    error_type = _classify_error_type(record.error)
    if error_type is not None:
        attrs["error.type"] = error_type
    return attrs


def _build_token_usage_attributes(
    record: MetricRecordsData, user_config: UserConfig, *, token_type: str
) -> dict[str, Any]:
    """Build spec-required attributes for token usage histograms."""
    attrs: dict[str, Any] = {}
    attrs["gen_ai.operation.name"] = _map_operation_name(str(user_config.endpoint.type))
    attrs["gen_ai.provider.name"] = infer_provider_name(user_config)
    if user_config.endpoint.model_names:
        attrs["gen_ai.request.model"] = user_config.endpoint.model_names[0]
    attrs["gen_ai.token.type"] = token_type
    return attrs


ATTRIBUTE_BUILDERS: dict[str, Callable[..., dict[str, Any]]] = {
    "request_latency": _build_duration_attributes,
    "time_to_first_token": _build_duration_attributes,
    "inter_token_latency": _build_duration_attributes,
    "input_token_count": lambda record, user_config: _build_token_usage_attributes(
        record, user_config, token_type="input"
    ),
    "output_token_count": lambda record, user_config: _build_token_usage_attributes(
        record, user_config, token_type="output"
    ),
}
"""Per-metric attribute builder. Each returns the spec-required attributes for that metric."""


# ---------------------------------------------------------------------------
# Provider inference
# ---------------------------------------------------------------------------

_PROVIDER_HOST_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"api\.openai\.com$"), "openai"),
    (re.compile(r"api\.anthropic\.com$"), "anthropic"),
    (re.compile(r"api\.deepseek\.com$"), "deepseek"),
    (re.compile(r"api\.mistral\.ai$"), "mistral_ai"),
    (re.compile(r"api\.cohere\.(ai|com)$"), "cohere"),
    (re.compile(r"api\.x\.ai$"), "x_ai"),
    (re.compile(r"api\.groq\.com$"), "groq"),
    (re.compile(r"api\.perplexity\.ai$"), "perplexity"),
    (re.compile(r"generativelanguage\.googleapis\.com$"), "gcp.gemini"),
    (re.compile(r".*-aiplatform\.googleapis\.com$"), "gcp.vertex_ai"),
    (re.compile(r"bedrock-runtime\..*\.amazonaws\.com$"), "aws.bedrock"),
    (re.compile(r".*\.openai\.azure\.com$"), "azure.ai.openai"),
    (re.compile(r".*\.services\.ai\.azure\.com$"), "azure.ai.inference"),
    (re.compile(r".*\.ibm\.com$"), "ibm.watsonx.ai"),
)


def _extract_host(url: str) -> str | None:
    """Extract the host portion from a URL string, handling various formats."""
    url = url.strip().lower()
    if not url:
        return None

    if "://" in url:
        try:
            parsed = urlparse(url)
            return parsed.hostname
        except ValueError:
            return None

    # Bare host or host:port
    host = url.split(":")[0].split("/")[0]
    return host if host else None


def infer_provider_name(user_config: UserConfig) -> str:
    """Determine gen_ai.provider.name attribute value.

    Precedence:
    (a) explicit --gen-ai-provider override wins,
    (b) else auto-infer from URL host,
    (c) else '_OTHER'.
    """
    # Path (a): explicit override
    explicit = getattr(user_config, "gen_ai_provider", None)
    if explicit:
        return explicit

    # Path (b): URL host inference
    urls = getattr(user_config.endpoint, "urls", None)
    if urls:
        for url in urls:
            host = _extract_host(url)
            if host is None:
                continue
            for pattern, provider in _PROVIDER_HOST_PATTERNS:
                if pattern.search(host):
                    return provider

    # Path (c): fallback
    return "_OTHER"


# ---------------------------------------------------------------------------
# Cross-metric attributes (for timing strategy)
# ---------------------------------------------------------------------------


def cross_metric_attributes(user_config: UserConfig) -> dict[str, str]:
    """Return the three Required GenAI spec attributes for timing-strategy callers.

    These are merged into the attribute dict of aiperf.timing.* metrics so
    dashboards can join timing metrics with the spec-named request metrics.
    """
    attrs: dict[str, str] = {}
    attrs["gen_ai.operation.name"] = _map_operation_name(str(user_config.endpoint.type))
    attrs["gen_ai.provider.name"] = infer_provider_name(user_config)
    if user_config.endpoint.model_names:
        attrs["gen_ai.request.model"] = user_config.endpoint.model_names[0]
    return attrs


# ---------------------------------------------------------------------------
# Main translation entry point
# ---------------------------------------------------------------------------


def translate(
    aiperf_metric_name: str,
    aiperf_value: float,
    record: MetricRecordsData,
    *,
    user_config: UserConfig,
) -> GenAISemconvEmission | None:
    """Translate an AIPerf metric to its GenAI spec equivalent.

    Returns GenAISemconvEmission if the metric has a spec equivalent, None otherwise.
    When None is returned, the caller should emit the metric under its original
    aiperf.* name unchanged.
    """
    mapping = METRIC_NAME_MAP.get(aiperf_metric_name)
    if mapping is None:
        return None

    spec_metric_name, unit, bucket_boundaries = mapping

    # Convert value
    converter = UNIT_CONVERTERS[aiperf_metric_name]
    converted_value = converter(aiperf_value)

    # Build attributes
    builder = ATTRIBUTE_BUILDERS[aiperf_metric_name]
    attributes = builder(record, user_config)

    return GenAISemconvEmission(
        spec_metric_name=spec_metric_name,
        instrument_kind="histogram",
        unit=unit,
        value=converted_value,
        attributes=attributes,
        explicit_bucket_boundaries=bucket_boundaries,
    )
