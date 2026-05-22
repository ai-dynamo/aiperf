# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model endpoint information.

This module contains the pydantic models that encapsulate the information needed to
send requests to an inference server, primarily around the model, endpoint, and
additional request payload information.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field, field_serializer

from aiperf.common.enums import (
    ConnectionReuseStrategy,
    ModelSelectionStrategy,
    RequestContentType,
)
from aiperf.common.models import AIPerfBaseModel
from aiperf.config.endpoint import EndpointDefaults, TemplateConfig
from aiperf.plugin.enums import EndpointType, TransportType

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class ModelInfo(AIPerfBaseModel):
    """Information about a model."""

    name: str = Field(
        ...,
        min_length=1,
        description="The name of the model. This is used to identify the model.",
    )
    version: str | None = Field(
        default=None,
        description="The version of the model.",
    )


class ModelListInfo(AIPerfBaseModel):
    """Information about a list of models."""

    models: list[ModelInfo] = Field(
        ...,
        min_length=1,
        description="The models to use for the endpoint.",
    )
    model_selection_strategy: ModelSelectionStrategy = Field(
        ...,
        description="The strategy to use for selecting the model to use for the endpoint.",
    )


class EndpointInfo(AIPerfBaseModel):
    """Information about an endpoint."""

    type: EndpointType = Field(
        default=EndpointDefaults.TYPE,
        description="The type of request payload to use for the endpoint.",
    )
    base_urls: list[str] = Field(
        default=[EndpointDefaults.URL],
        min_length=1,
        description="URL(s) of the endpoint. Multiple URLs enable load balancing across servers.",
    )
    custom_endpoint: str | None = Field(
        default=None,
        description="Custom endpoint to use for the models.",
    )
    url_params: dict[str, Any] | None = Field(
        default=None, description="Custom URL parameters to use for the endpoint."
    )
    streaming: bool = Field(
        default=EndpointDefaults.STREAMING,
        description="Whether the endpoint supports streaming.",
    )
    headers: list[tuple[str, str]] = Field(
        default=[],
        description="Custom URL headers to use for the endpoint.",
    )

    @field_serializer("headers", when_used="json")
    def _redact_headers(self, value: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Redact credential-bearing header values in exported JSON artifacts.

        Defense-in-depth alongside EndpointConfig._redact_headers: any future
        exporter that dumps EndpointInfo to JSON inherits the same redaction.
        """
        from aiperf.common.redact import redact_header_tuples

        return redact_header_tuples(value)

    api_key: str | None = Field(
        default=EndpointDefaults.API_KEY,
        description="API key to use for the endpoint.",
        repr=False,
        exclude=True,
    )
    ssl_options: dict[str, Any] | None = Field(
        default=None,
        description="SSL options to use for the endpoint.",
    )
    timeout: float = Field(
        default=EndpointDefaults.TIMEOUT,
        gt=0,
        description="The timeout in seconds for each request to the endpoint.",
    )
    extra: list[tuple[str, Any]] = Field(
        default=[],
        description="Additional inputs to include with every request. "
        "You can repeat this flag for multiple inputs. Inputs should be in an 'input_name:value' format. "
        "Alternatively, a string representing a json formatted dict can be provided.",
    )
    use_legacy_max_tokens: bool = Field(
        default=EndpointDefaults.USE_LEGACY_MAX_TOKENS,
        description="Use the legacy 'max_tokens' field instead of 'max_completion_tokens' in request payloads.",
    )
    use_server_token_count: bool = Field(
        default=EndpointDefaults.USE_SERVER_TOKEN_COUNT,
        description="Use server-reported token counts from API usage fields instead of client-side tokenization.",
    )
    connection_reuse_strategy: ConnectionReuseStrategy = Field(
        default=EndpointDefaults.CONNECTION_REUSE_STRATEGY,
        description="Transport connection reuse strategy.",
    )
    download_video_content: bool = Field(
        default=EndpointDefaults.DOWNLOAD_VIDEO_CONTENT,
        description="For video generation endpoints, download the video content after generation completes.",
    )
    request_content_type: RequestContentType | None = Field(
        default=EndpointDefaults.REQUEST_CONTENT_TYPE,
        description="Content type for request body serialization. None means application/json.",
    )
    session_header: str | None = Field(
        default=None,
        description="HTTP header name to use for the per-session affinity identifier. "
        "When set, replaces the default `X-Correlation-ID` header name with this value.",
    )
    collect_trace_chunks: bool = Field(
        default=False,
        description="Collect per-chunk trace data (timestamps and sizes) for HTTP trace export. "
        "When False, only aggregate metrics are tracked (counts, totals, first/last timestamps).",
    )
    template: TemplateConfig | None = Field(
        default=None,
        description="Custom template configuration for template endpoints. "
        "Provides the Jinja2 request body and JMESPath response_field used by TemplateEndpoint.",
    )

    @property
    def base_url(self) -> str:
        """Return the first URL for backward compatibility."""
        return self.base_urls[0]

    def get_url(self, index: int | None = None) -> str:
        """Get a URL by index with wrap-around.

        Args:
            index: Index into the URLs list. If None, returns the first URL.

        Returns:
            The URL at the given index (with modulo wrap-around).
        """
        if index is None:
            return self.base_urls[0]
        return self.base_urls[index % len(self.base_urls)]


class ModelEndpointInfo(AIPerfBaseModel):
    """Information about a model endpoint."""

    models: ModelListInfo = Field(
        ...,
        description="The models to use for the endpoint.",
    )
    endpoint: EndpointInfo = Field(
        ...,
        description="The endpoint to use for the models.",
    )
    transport: TransportType | None = Field(
        default=None,
        description="The transport to use for the endpoint. If not provided, it will be auto-detected from the URL.",
    )

    @classmethod
    def from_run(cls, run: BenchmarkRun) -> ModelEndpointInfo:
        """Create a ModelEndpointInfo from a BenchmarkRun."""
        cfg = run.cfg
        ep = cfg.endpoint
        models_advanced = cfg.models
        return cls(
            models=ModelListInfo(
                models=[ModelInfo(name=item.name) for item in models_advanced.items],
                model_selection_strategy=models_advanced.strategy,
            ),
            endpoint=EndpointInfo(
                type=ep.type,
                custom_endpoint=getattr(ep, "path", None),
                streaming=ep.streaming,
                base_urls=list(ep.urls),
                headers=list((getattr(ep, "headers", {}) or {}).items()),
                extra=list((getattr(ep, "extra", {}) or {}).items()),
                timeout=ep.timeout,
                api_key=ep.api_key,
                use_legacy_max_tokens=ep.use_legacy_max_tokens,
                use_server_token_count=ep.use_server_token_count,
                connection_reuse_strategy=ep.connection_reuse,
                download_video_content=ep.download_video_content,
                request_content_type=ep.request_content_type,
                collect_trace_chunks=False,
                template=getattr(ep, "template", None),
                session_header=getattr(ep, "session_header", None),
            ),
            transport=ep.transport,
        )

    @property
    def primary_model(self) -> ModelInfo:
        """Get the primary model."""
        return self.models.models[0]

    @property
    def primary_model_name(self) -> str:
        """Get the primary model name."""
        return self.primary_model.name
