# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Pydantic Models

Endpoint - Server connection and API configuration
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlparse

from pydantic import (
    AfterValidator,
    ConfigDict,
    Field,
    field_serializer,
    model_validator,
)

from aiperf.common.enums import (
    ConnectionReuseStrategy,
    ModelSelectionStrategy,
    RequestContentType,
)
from aiperf.config.base import BaseConfig
from aiperf.config.loader.parsing import normalize_http_urls
from aiperf.plugin.enums import (
    EndpointType,
    TransportType,
    URLSelectionStrategy,
)

__all__ = [
    "EndpointConfig",
    "EndpointDefaults",
    "TemplateConfig",
]


@dataclass(frozen=True)
class EndpointDefaults:
    MODEL_SELECTION_STRATEGY = ModelSelectionStrategy.ROUND_ROBIN
    CUSTOM_ENDPOINT = None
    TYPE = EndpointType.CHAT
    STREAMING = False
    URL = "http://localhost:8000"
    URL_STRATEGY = URLSelectionStrategy.ROUND_ROBIN
    TIMEOUT = 6 * 60 * 60  # 6 hours, match vLLM benchmark default
    API_KEY = None
    USE_LEGACY_MAX_TOKENS = False
    USE_SERVER_TOKEN_COUNT = False
    CONNECTION_REUSE_STRATEGY = ConnectionReuseStrategy.POOLED
    DOWNLOAD_VIDEO_CONTENT = False
    REQUEST_CONTENT_TYPE = None
    # Readiness probe defaults. Timeout 0 disables the probe (the default);
    # any positive value enables it. Interval is only consulted when the
    # probe is enabled but is validated positive so mis-configuration
    # (e.g. --wait-for-model-interval 0) is rejected at config-load time.
    WAIT_FOR_MODEL_TIMEOUT = 0.0
    WAIT_FOR_MODEL_INTERVAL = 5.0
    WAIT_FOR_MODEL_MODE = "inference"


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
            "URLs without a scheme have ``http://`` prepended automatically. "
            "Example: ['http://localhost:8000/v1/chat/completions']",
            json_schema_extra={"x-kubernetes-preserve-unknown-fields": True},
        ),
        AfterValidator(normalize_http_urls),
    ]

    @field_serializer("urls")
    def _redact_urls(self, value: list[str], info: Any) -> list[str]:
        """Redact URL userinfo in serialized config artifacts by default."""
        context = getattr(info, "context", None)
        if isinstance(context, dict) and context.get("include_secrets"):
            return value
        from aiperf.common.redact import redact_url

        return [redact_url(url) for url in value]

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
            "template: custom format, and others — see `aiperf plugins` "
            "for the full list.",
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
            repr=False,
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
            default=EndpointDefaults.TIMEOUT,
            description="Request timeout in seconds (0 = no timeout). "
            "Requests exceeding this duration are marked as failed. "
            "Should exceed expected max response time.",
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
            description="Transport plugin name. Currently only 'http' (aiohttp-based "
            "HTTP/1.1) is shipped. Auto-detected from URL when unset; explicit "
            "setting overrides auto-detection.",
        ),
    ]

    connection_reuse: Annotated[
        ConnectionReuseStrategy,
        Field(
            default=ConnectionReuseStrategy.POOLED,
            description="HTTP connection management strategy. "
            "pooled: shared connection pool (fastest), "
            "never: new connection per request (includes TCP overhead), "
            "sticky-user-sessions: dedicated connection per user session.",
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

    @field_serializer("headers", when_used="json")
    def _redact_headers(self, value: dict[str, str]) -> dict[str, str]:
        """Redact credential-bearing header values in exported JSON artifacts.

        Mirrors the api_key serializer above: profile_export_aiperf.json and
        server_metrics_export.json otherwise leak Authorization / X-API-Key /
        api-key etc. verbatim into on-disk artifacts.
        """
        from aiperf.common.redact import redact_headers

        return redact_headers(value) or {}

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

    session_header: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "HTTP header name used to carry the per-session affinity identifier. "
                "When set, replaces the default `X-Correlation-ID` header. Useful "
                "when the inference server expects a custom session-affinity header "
                "(e.g. `--session-header X-Session-ID`)."
            ),
        ),
    ]

    wait_for_model_timeout: Annotated[
        float,
        Field(
            default=0.0,
            description="Enable a pre-flight readiness probe by setting this to a positive value (seconds). "
            "aiperf applies this timeout to each URL/model probe before starting the benchmark, "
            "aborting with a non-zero exit if any probe times out. For multiple URLs or models, "
            "worst-case wall-clock time can be roughly this timeout multiplied by the number of "
            "URL/model probes. The probe strategy is controlled by `--wait-for-model-mode`, which "
            "defaults to sending a 1-token inference request. 0 (default) disables the probe. "
            "Eliminates the need for external shell-based readiness loops in containers and Kubernetes recipes.",
            ge=0.0,
        ),
    ]

    wait_for_model_interval: Annotated[
        float,
        Field(
            default=5.0,
            description="Seconds between readiness probe attempts. "
            "Only consulted when `--wait-for-model-timeout` is positive.",
            gt=0.0,
        ),
    ]

    wait_for_model_mode: Annotated[
        Literal["models", "inference", "both"],
        Field(
            default="inference",
            description="Strategy for the readiness probe. "
            "'inference' (default): POST a 1-token inference request to the configured endpoint; "
            "this is the strongest signal — it proves the full stack (frontend, scheduler, worker, "
            "forward pass) is live. Any HTTP status < 500 counts as ready. "
            "'models': GET `/v1/models` and verify the model id appears in `data[]` "
            "(cheaper, no tokens consumed; falls back to a plain GET on the base URL on 404). "
            "'both': run 'models' first, then 'inference'. "
            "Only consulted when `--wait-for-model-timeout` is positive.",
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

        # url → urls (singular to plural). Both keys can be present after a
        # YAML+CLI deep-merge (resolve_config) where the YAML used the
        # singular shorthand and a CLI flag overlaid the plural form: in that
        # case the CLI wins and we drop the YAML shorthand. Without this,
        # AIPerfConfig.endpoint.model_validate fails with `url: Extra inputs
        # are not permitted` because both keys reach the validator.
        if "url" in data:
            url = data.pop("url")
            if "urls" not in data:
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
    def _validate_endpoint_boundaries(self) -> Self:
        for url in self.urls:
            # Reject leading/trailing whitespace explicitly so that a malformed
            # URL like "  http://h  " or "http://h\n" produces a clear
            # config-time error instead of an InvalidUrlClientError at request
            # time. Internal whitespace (spaces in path, embedded newlines) is
            # also rejected because aiohttp will refuse the request anyway.
            if url != url.strip():
                raise ValueError(
                    f"URL {url!r} has leading or trailing whitespace. "
                    f"Strip whitespace before passing the URL."
                )
            if any(ch.isspace() for ch in url):
                raise ValueError(
                    f"URL {url!r} contains whitespace. URLs must not contain "
                    f"spaces, tabs, or newlines."
                )
            parsed = urlparse(url)
            # Reject anything that lacks a scheme, a netloc, or a hostname.
            # ``http://:18765`` parses as scheme=http, netloc=':18765', hostname=None
            # — looks valid by netloc alone, but aiohttp rejects it with
            # InvalidUrlClientError at request time. Catch it here.
            if not parsed.scheme or not parsed.netloc or not parsed.hostname:
                raise ValueError(
                    f"URL {url!r} is missing scheme or host. "
                    f"Expected 'http://host:port' or 'https://host:port'."
                )
            if parsed.scheme.lower() not in ("http", "https"):
                raise ValueError(
                    f"URL {url!r} has unsupported scheme {parsed.scheme!r}. "
                    f"Expected 'http' or 'https'."
                )
            # Validate the port if one is present. urlparse.port raises
            # ValueError on access for non-numeric or out-of-range ports
            # (``:abc``, ``:99999``, ``:-1``); catch and re-raise with a
            # clear, URL-aware message instead of bubbling the obscure
            # "Port out of range 0-65535".
            try:
                port = parsed.port
            except ValueError as exc:
                raise ValueError(
                    f"URL {url!r} has an invalid port. "
                    f"Ports must be integers in the range 1..65535."
                ) from exc
            if port is not None and not (1 <= port <= 65535):
                raise ValueError(
                    f"URL {url!r} has port {port} outside the valid range 1..65535."
                )
        if self.path is not None and not self.path.startswith("/"):
            raise ValueError("endpoint.path must start with a leading slash")
        return self

    @model_validator(mode="after")
    def _validate_template_required(self) -> Self:
        if self.type == EndpointType.TEMPLATE and self.template is None:
            raise ValueError("template is required when endpoint type is 'template'")
        return self

    @model_validator(mode="after")
    def _validate_wait_for_model_coherent(self) -> Self:
        """Reject configurations where probe sub-options are set to non-default
        values without enabling the probe itself (timeout > 0). Catches typos like
        `--wait-for-model-interval 1` without a timeout value.

        Defaults (`interval=5.0`, `mode="inference"`) are tolerated even when
        they appear in ``model_fields_set`` (e.g. from a round-tripped config),
        so only explicit non-default values trigger the error.
        """
        if self.wait_for_model_timeout > 0:
            return self
        non_default = []
        if (
            "wait_for_model_interval" in self.model_fields_set
            and self.wait_for_model_interval != 5.0
        ):
            non_default.append("--wait-for-model-interval")
        if (
            "wait_for_model_mode" in self.model_fields_set
            and self.wait_for_model_mode != "inference"
        ):
            non_default.append("--wait-for-model-mode")
        if non_default:
            shown = ", ".join(non_default)
            raise ValueError(
                f"{shown} has no effect unless --wait-for-model-timeout is set "
                f"to a positive value. Set --wait-for-model-timeout to enable "
                f"the readiness probe."
            )
        return self

    @model_validator(mode="after")
    def _validate_request_content_type(self) -> Self:
        """Auto-select multipart for endpoints that declare requires_form_data."""
        from aiperf.plugin import plugins

        metadata = plugins.get_endpoint_metadata(self.type)
        requires_form_data = getattr(metadata, "requires_form_data", False)

        if self.request_content_type is None:
            if requires_form_data:
                self.request_content_type = RequestContentType.MULTIPART_FORM_DATA
            return self

        if self.request_content_type == RequestContentType.APPLICATION_JSON:
            if requires_form_data:
                raise ValueError(
                    f"endpoint type {self.type} requires multipart/form-data; "
                    "application/json is not supported."
                )
            return self

        if not requires_form_data:
            raise ValueError(
                f"request_content_type={self.request_content_type} is only supported for "
                f"endpoint types that accept form-data (e.g. image_edit, "
                f"video_generation); endpoint type {self.type} does not."
            )
        return self
