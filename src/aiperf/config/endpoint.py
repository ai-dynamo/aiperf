# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Pydantic Models

Endpoint - Server connection and API configuration
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import (
    ConfigDict,
    Field,
    field_serializer,
    model_validator,
)
from typing_extensions import Self

from aiperf.common.enums import (
    ConnectionReuseStrategy,
    RequestContentType,
)
from aiperf.config._base import BaseConfig
from aiperf.plugin.enums import (
    EndpointType,
    TransportType,
    URLSelectionStrategy,
)

__all__ = [
    "EndpointConfig",
    "TemplateConfig",
]


class TemplateConfig(BaseConfig):
    """
    Configuration for custom template-based endpoints.

    When endpoint type is "template", this configures how requests
    are formatted and responses are parsed.
    """

    model_config = ConfigDict(extra="forbid")

    body: Annotated[
        str,
        Field(
            description="Jinja2 template string for request body. "
            "Variables: {{prompt}}, {{max_tokens}}, {{model}}, {{messages}}, etc.",
        ),
    ]

    response_field: Annotated[
        str,
        Field(
            default="text",
            description="JSON path to extract response text from API response. "
            "Use dot notation for nested fields: 'choices.0.message.content'.",
        ),
    ]


class EndpointConfig(BaseConfig):
    """
    Endpoint configuration for connecting to inference servers.

    This section configures how AIPerf connects to and communicates
    with the target inference server(s). It supports multiple URLs
    for load-balanced deployments and various API types.
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    urls: Annotated[
        list[str],
        Field(
            min_length=1,
            description="List of server URLs to benchmark. "
            "Requests distributed according to url_strategy. "
            "Example: ['http://localhost:8000/v1/chat/completions']",
        ),
    ]

    url_strategy: Annotated[
        URLSelectionStrategy,
        Field(
            default=URLSelectionStrategy.ROUND_ROBIN,
            description="Strategy for distributing requests across multiple URLs. "
            "round_robin cycles through URLs in order.",
        ),
    ]

    type: Annotated[
        EndpointType,
        Field(
            default=EndpointType.CHAT,
            description="API endpoint type determining request/response format. "
            "chat: OpenAI chat completions, completions: OpenAI completions, "
            "embeddings: vector embeddings, rankings: reranking, "
            "template: custom format.",
        ),
    ]

    path: Annotated[
        str | None,
        Field(
            default=None,
            description="Override default endpoint path. "
            "Use for servers with non-standard API paths. "
            "Example: '/custom/v2/generate'",
        ),
    ]

    api_key: Annotated[
        str | None,
        Field(
            default=None,
            description="API authentication key. "
            "Supports environment variable substitution: ${OPENAI_API_KEY}. "
            "Can also use ${VAR:default} syntax for defaults.",
        ),
    ]

    @field_serializer("api_key", when_used="json")
    def _redact_api_key(self, value: str | None) -> str | None:
        """Never serialize the raw API key into exported JSON artifacts."""
        from aiperf.common.redact import REDACTED_VALUE

        if value is None:
            return None
        return REDACTED_VALUE

    timeout: Annotated[
        float,
        Field(
            ge=0.0,
            default=600.0,
            description="Request timeout in seconds (0 = no timeout). "
            "Requests exceeding this duration are marked as failed. "
            "Should exceed expected max response time.",
        ),
    ]

    ready_check_timeout: Annotated[
        float,
        Field(
            ge=0.0,
            default=0.0,
            description="Seconds to wait for endpoint readiness before benchmarking "
            "(0 = skip). Sends a real inference request to verify the model "
            "is loaded and can generate output, not just a /health check.",
        ),
    ]

    streaming: Annotated[
        bool,
        Field(
            default=False,
            description="Enable streaming (Server-Sent Events) responses. "
            "Required for accurate TTFT (time to first token) measurement. "
            "Server must support streaming for this to work.",
        ),
    ]

    transport: Annotated[
        TransportType | None,
        Field(
            default=None,
            description="HTTP transport protocol (http/https). "
            "Auto-detected from URL scheme if not specified. "
            "Explicit setting overrides auto-detection.",
        ),
    ]

    connection_reuse: Annotated[
        ConnectionReuseStrategy,
        Field(
            default=ConnectionReuseStrategy.POOLED,
            description="HTTP connection management strategy. "
            "pooled: shared connection pool (fastest), "
            "never: new connection per request (includes TCP overhead), "
            "sticky_sessions: dedicated connection per session.",
        ),
    ]

    use_legacy_max_tokens: Annotated[
        bool,
        Field(
            default=False,
            description="Use 'max_tokens' field instead of 'max_completion_tokens'. "
            "Enable for compatibility with older OpenAI API versions.",
        ),
    ]

    use_server_token_count: Annotated[
        bool,
        Field(
            default=False,
            description="Use server-reported token counts from response usage field. "
            "When true, trusts usage.prompt_tokens and usage.completion_tokens. "
            "When false, counts tokens locally using configured tokenizer.",
        ),
    ]

    template: Annotated[
        TemplateConfig | None,
        Field(
            default=None,
            description="Custom template configuration for template endpoint type. "
            "Only used when type='template'. "
            "Defines request body format and response parsing.",
        ),
    ]

    headers: Annotated[
        dict[str, str],
        Field(
            default_factory=dict,
            description="Custom HTTP headers to include in all requests. "
            "Useful for authentication, tracing, or routing. "
            "Values support environment variable substitution.",
        ),
    ]

    extra: Annotated[
        dict[str, Any],
        Field(
            default_factory=dict,
            description="Additional fields to include in request body. "
            "Merged into every request. "
            "Common fields: temperature, top_p, top_k, stop.",
        ),
    ]

    download_video_content: Annotated[
        bool,
        Field(
            default=False,
            description="For video generation endpoints, download the video content "
            "after generation completes. Adds a content download step to the "
            "async polling flow.",
        ),
    ]

    request_content_type: Annotated[
        RequestContentType | None,
        Field(
            default=None,
            description=(
                "Content type for request body serialization. Default is "
                "'application/json'. Set to 'multipart/form-data' for servers that "
                "require form-encoded requests (e.g. vLLM video generation)."
            ),
        ),
    ]

    @model_validator(mode="before")
    @classmethod
    def normalize_before_validation(cls, data: Any) -> Any:
        """Normalize endpoint config before validation.

        Handles:
            - url → urls (singular to plural, wrapped in list)
            - Auto-set type to 'template' when template field is provided
            - Disable streaming when endpoint type does not support it
        """
        if not isinstance(data, dict):
            return data

        # url → urls (singular to plural)
        if "url" in data and "urls" not in data:
            url = data.pop("url")
            data["urls"] = [url] if isinstance(url, str) else url

        # Auto-detect template type
        if "template" in data and data["template"] is not None and "type" not in data:
            data["type"] = EndpointType.TEMPLATE

        # Disable streaming when the endpoint type does not support it
        if data.get("streaming"):
            try:
                from aiperf.plugin import plugins

                endpoint_type = data.get("type", EndpointType.CHAT)
                metadata = plugins.get_endpoint_metadata(endpoint_type)
                if not metadata.supports_streaming:
                    import warnings

                    warnings.warn(
                        f"Streaming is not supported for endpoint type '{endpoint_type}'. "
                        "Streaming will be disabled.",
                        UserWarning,
                        stacklevel=2,
                    )
                    data["streaming"] = False
            except ImportError:
                pass

        return data

    @model_validator(mode="after")
    def _validate_template_required(self) -> Self:
        if self.type == EndpointType.TEMPLATE and self.template is None:
            raise ValueError("template is required when endpoint type is 'template'")
        return self

    @model_validator(mode="after")
    def _validate_request_content_type(self) -> Self:
        """multipart/form-data only on endpoints that declare requires_form_data."""
        if (
            self.request_content_type is None
            or self.request_content_type == RequestContentType.APPLICATION_JSON
        ):
            return self

        from aiperf.plugin import plugins

        metadata = plugins.get_endpoint_metadata(self.type)
        if not getattr(metadata, "requires_form_data", False):
            raise ValueError(
                f"request_content_type={self.request_content_type} is only supported for "
                f"endpoint types that accept form-data (e.g. video_generation); "
                f"endpoint type {self.type} does not."
            )
        return self
