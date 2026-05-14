# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BeforeValidator,
    Field,
    SerializationInfo,
    field_serializer,
    model_validator,
)
from typing_extensions import Self

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.common.config.config_defaults import EndpointDefaults
from aiperf.common.config.config_validators import (
    normalize_http_urls,
    parse_str_or_list,
)
from aiperf.common.config.groups import Groups
from aiperf.common.enums import (
    ConnectionReuseStrategy,
    ModelSelectionStrategy,
    RequestContentType,
)
from aiperf.common.redact import REDACTED_VALUE, redact_url
from aiperf.plugin.enums import (
    EndpointType,
    TransportType,
    URLSelectionStrategy,
)

_logger = AIPerfLogger(__name__)


class EndpointConfig(BaseConfig):
    """
    A configuration class for defining endpoint related settings.
    """

    @model_validator(mode="after")
    def validate_streaming(self) -> Self:
        """Validate that streaming is supported for the endpoint type."""
        if not self.streaming:
            return self

        # Lazy import to avoid circular dependency
        from aiperf.plugin import plugins

        metadata = plugins.get_endpoint_metadata(self.type)
        if not metadata.supports_streaming:
            _logger.warning(
                f"Streaming is not supported for --endpoint-type {self.type}, setting streaming to False"
            )
            self.streaming = False
        return self

    @model_validator(mode="after")
    def validate_wait_for_model_coherent(self) -> Self:
        """Reject configurations where probe sub-options are set without
        enabling the probe itself (timeout > 0). Catches typos like
        `--wait-for-model-interval 1` without a timeout value.
        """
        if self.wait_for_model_timeout > 0:
            return self
        dependent = {"wait_for_model_interval", "wait_for_model_mode"}
        set_without_enable = sorted(dependent & self.model_fields_set)
        if set_without_enable:
            flag_names = {
                "wait_for_model_interval": "--wait-for-model-interval",
                "wait_for_model_mode": "--wait-for-model-mode",
            }
            shown = ", ".join(flag_names[f] for f in set_without_enable)
            raise ValueError(
                f"{shown} has no effect unless --wait-for-model-timeout is set "
                f"to a positive value. Set --wait-for-model-timeout to enable "
                f"the readiness probe."
            )
        return self

    model_names: Annotated[
        list[str],
        Field(
            ...,  # This must be set by the user
            description="Model name(s) to be benchmarked. Can be a comma-separated list or a single model name.",
        ),
        BeforeValidator(parse_str_or_list),
        CLIParameter(
            name=(
                "--model-names",
                "--model",  # GenAI-Perf
                "-m",  # GenAI-Perf
            ),
            group=Groups.ENDPOINT,
        ),
    ]

    model_selection_strategy: Annotated[
        ModelSelectionStrategy,
        Field(
            description="When multiple models are specified, this is how a specific model should be assigned to a prompt.\n"
            "round_robin: nth prompt in the list gets assigned to n-mod len(models).\n"
            "random: assignment is uniformly random",
        ),
        CLIParameter(
            name=(
                "--model-selection-strategy",  # GenAI-Perf
            ),
            group=Groups.ENDPOINT,
        ),
    ] = EndpointDefaults.MODEL_SELECTION_STRATEGY

    custom_endpoint: Annotated[
        str | None,
        Field(
            description="Set a custom API endpoint path (e.g., `/v1/custom`, `/my-api/chat`). "
            "By default, endpoints follow OpenAI-compatible paths like `/v1/chat/completions`. "
            "Use this option to override the default path for non-standard API implementations.",
        ),
        CLIParameter(
            name=(
                "--custom-endpoint",
                "--endpoint",  # GenAI-Perf
            ),
            group=Groups.ENDPOINT,
        ),
    ] = EndpointDefaults.CUSTOM_ENDPOINT

    type: Annotated[
        EndpointType,
        Field(
            description="The API endpoint type to benchmark. Determines request/response format and supported features. "
            "Common types: `chat` (multi-modal conversations), `embeddings` (vector generation), `completions` (text completion). "
            "See enum documentation for all supported endpoint types.",
        ),
        CLIParameter(
            name=(
                "--endpoint-type",  # GenAI-Perf
            ),
            group=Groups.ENDPOINT,
        ),
    ] = EndpointDefaults.TYPE

    streaming: Annotated[
        bool,
        Field(
            description="Enable streaming responses. When enabled, the server streams tokens incrementally "
            "as they are generated. Automatically disabled if the selected endpoint type does not support streaming. "
            "Enables measurement of time-to-first-token (TTFT) and inter-token latency (ITL) metrics.",
        ),
        CLIParameter(
            name=(
                "--streaming",  # GenAI-Perf
            ),
            group=Groups.ENDPOINT,
        ),
    ] = EndpointDefaults.STREAMING

    urls: Annotated[
        list[str],
        Field(
            description="Base URL(s) of the API server(s) to benchmark. Multiple URLs can be specified for load balancing "
            "across multiple instances (e.g., `--url http://server1:8000 --url http://server2:8000`). "
            "The endpoint path is automatically appended based on `--endpoint-type` (e.g., `/v1/chat/completions` for `chat`). "
            "URLs that do not include a scheme (no `://`) have `http://` prepended automatically.",
            min_length=1,
            # Run the validator chain on the default too — without this, a
            # bare `--wait-for-model-timeout 30` (no `--url`) would send the
            # un-normalized default to aiohttp and reproduce the original bug.
            validate_default=True,
        ),
        BeforeValidator(parse_str_or_list),
        AfterValidator(normalize_http_urls),
        CLIParameter(
            name=(
                "--url",  # GenAI-Perf
                "-u",  # GenAI-Perf
            ),
            consume_multiple=True,
            group=Groups.ENDPOINT,
        ),
    ] = [EndpointDefaults.URL]

    url_selection_strategy: Annotated[
        URLSelectionStrategy,
        Field(
            description="Strategy for selecting URLs when multiple `--url` values are provided. "
            "'round_robin' (default): distribute requests evenly across URLs in sequential order.",
        ),
        CLIParameter(
            name=("--url-strategy",),
            group=Groups.ENDPOINT,
        ),
    ] = EndpointDefaults.URL_STRATEGY

    @property
    def url(self) -> str:
        """Return the first URL for backward compatibility."""
        return self.urls[0]

    timeout_seconds: Annotated[
        float,
        Field(
            description="Maximum time in seconds to wait for each HTTP request to complete, including connection establishment, "
            "request transmission, and response receipt. Applies to both streaming and non-streaming requests. "
            "Requests exceeding this timeout are cancelled and recorded as failures.",
        ),
        CLIParameter(
            name=("--request-timeout-seconds"),
            group=Groups.ENDPOINT,
        ),
    ] = EndpointDefaults.TIMEOUT

    api_key: Annotated[
        str | None,
        Field(
            description="API authentication key for the endpoint. When provided, automatically included in request headers as "
            "`Authorization: Bearer <api_key>`.",
            repr=False,
        ),
        CLIParameter(
            name=("--api-key"),
            group=Groups.ENDPOINT,
        ),
    ] = EndpointDefaults.API_KEY

    wait_for_model_timeout: Annotated[
        float,
        Field(
            description="Enable a pre-flight readiness probe by setting this to a positive value (seconds). "
            "aiperf applies this timeout to each URL/model probe before starting the benchmark, "
            "aborting with a non-zero exit if any probe times out. For multiple URLs or models, "
            "worst-case wall-clock time can be roughly this timeout multiplied by the number of "
            "URL/model probes. The probe strategy is controlled by `--wait-for-model-mode`, which "
            "defaults to sending a 1-token inference request. 0 (default) disables the probe. "
            "Eliminates the need for external shell-based readiness loops in containers and Kubernetes recipes.",
            ge=0.0,
        ),
        CLIParameter(
            name=("--wait-for-model-timeout",),
            group=Groups.ENDPOINT,
        ),
    ] = EndpointDefaults.WAIT_FOR_MODEL_TIMEOUT

    wait_for_model_interval: Annotated[
        float,
        Field(
            description="Seconds between readiness probe attempts. "
            "Only consulted when `--wait-for-model-timeout` is positive.",
            gt=0.0,
        ),
        CLIParameter(
            name=("--wait-for-model-interval",),
            group=Groups.ENDPOINT,
        ),
    ] = EndpointDefaults.WAIT_FOR_MODEL_INTERVAL

    wait_for_model_mode: Annotated[
        Literal["models", "inference", "both"],
        Field(
            description="Strategy for the readiness probe. "
            "'inference' (default): POST a 1-token inference request to the configured endpoint; "
            "this is the strongest signal — it proves the full stack (frontend, scheduler, worker, "
            "forward pass) is live. Any HTTP status < 500 counts as ready. "
            "'models': GET `/v1/models` and verify the model id appears in `data[]` "
            "(cheaper, no tokens consumed; falls back to a plain GET on the base URL on 404). "
            "'both': run 'models' first, then 'inference'. "
            "Only consulted when `--wait-for-model-timeout` is positive.",
        ),
        CLIParameter(
            name=("--wait-for-model-mode",),
            group=Groups.ENDPOINT,
        ),
    ] = EndpointDefaults.WAIT_FOR_MODEL_MODE

    transport: Annotated[
        TransportType | None,
        Field(
            description="Transport protocol to use for API requests. If not specified, auto-detected from the URL scheme "
            "(`http`/`https` → `TransportType.HTTP`). Currently supports `http` transport using aiohttp with connection pooling, "
            "TCP optimization, and Server-Sent Events (SSE) for streaming. Explicit override rarely needed.",
        ),
        CLIParameter(
            name=("--transport", "--transport-type"),
            group=Groups.ENDPOINT,
        ),
    ] = None

    use_legacy_max_tokens: Annotated[
        bool,
        Field(
            description="Use the legacy 'max_tokens' field instead of 'max_completion_tokens' in request payloads. "
            "The OpenAI API now prefers 'max_completion_tokens', but some older APIs or implementations may require 'max_tokens'.",
        ),
        CLIParameter(
            name=("--use-legacy-max-tokens",),
            group=Groups.ENDPOINT,
        ),
    ] = EndpointDefaults.USE_LEGACY_MAX_TOKENS

    use_server_token_count: Annotated[
        bool,
        Field(
            description=(
                "Use server-reported token counts from API usage fields instead of "
                "client-side tokenization. When enabled, tokenizers are still loaded "
                "(needed for dataset generation) but tokenizer.encode() is not called "
                "for computing metrics. Token count fields will be None if the server "
                "does not provide usage information. For OpenAI-compatible streaming "
                "endpoints (chat/completions), stream_options.include_usage is automatically "
                "configured when this flag is enabled."
            ),
        ),
        CLIParameter(
            name=("--use-server-token-count",),
            group=Groups.ENDPOINT,
        ),
    ] = EndpointDefaults.USE_SERVER_TOKEN_COUNT

    connection_reuse_strategy: Annotated[
        ConnectionReuseStrategy,
        Field(
            description=(
                "Transport connection reuse strategy. "
                "'pooled' (default): connections are pooled and reused across all requests. "
                "'never': new connection for each request, closed after response. "
                "'sticky-user-sessions': connection persists across turns of a multi-turn "
                "conversation, closed on final turn (enables sticky load balancing)."
            ),
        ),
        CLIParameter(
            name=("--connection-reuse-strategy",),
            group=Groups.ENDPOINT,
        ),
    ] = EndpointDefaults.CONNECTION_REUSE_STRATEGY

    download_video_content: Annotated[
        bool,
        Field(
            description=(
                "For video generation endpoints, download the video content after generation completes. "
                "When enabled, request latency includes the video download time. "
                "When disabled (default), only generation time is measured."
            ),
        ),
        CLIParameter(
            name=("--download-video-content",),
            group=Groups.ENDPOINT,
        ),
    ] = EndpointDefaults.DOWNLOAD_VIDEO_CONTENT

    request_content_type: Annotated[
        RequestContentType | None,
        Field(
            description=(
                "Content type for request body serialization. By default, requests are sent as "
                "'application/json'. Set to 'multipart/form-data' for servers that require form-encoded "
                "requests (e.g., vLLM video generation endpoints)."
            ),
        ),
        CLIParameter(
            name=("--request-content-type",),
            group=Groups.ENDPOINT,
        ),
    ] = EndpointDefaults.REQUEST_CONTENT_TYPE

    @model_validator(mode="after")
    def validate_request_content_type(self) -> Self:
        """Auto-default to multipart for requires_form_data endpoints, and
        reject content-type/endpoint combinations that the transport cannot
        serialize (JSON on multipart-only endpoints, or multipart on JSON-only
        endpoints).
        """
        from aiperf.plugin import plugins

        metadata = plugins.get_endpoint_metadata(self.type)

        if self.request_content_type is None:
            if metadata.requires_form_data:
                self.request_content_type = RequestContentType.MULTIPART_FORM_DATA
            return self

        if (
            self.request_content_type == RequestContentType.APPLICATION_JSON
            and metadata.requires_form_data
        ):
            raise ValueError(
                f"--endpoint-type {self.type} requires multipart/form-data; "
                f"application/json is not supported on this endpoint. "
                f"Omit --request-content-type to use the auto-default."
            )

        if self.request_content_type == RequestContentType.APPLICATION_JSON:
            return self

        if not metadata.requires_form_data:
            raise ValueError(
                f"--request-content-type {self.request_content_type} is only supported for "
                f"endpoint types that support form-data encoding (e.g., image_edit, "
                f"video_generation), but --endpoint-type {self.type} does not support it."
            )
        return self

    @field_serializer("api_key")
    @classmethod
    def _redact_api_key(cls, v: str | None, info: SerializationInfo) -> str | None:
        """Redact api_key during serialization unless context explicitly allows it."""
        if info.context and info.context.get("include_secrets"):
            return v
        return REDACTED_VALUE if v else v

    @field_serializer("urls")
    @classmethod
    def _redact_urls(cls, v: list[str], info: SerializationInfo) -> list[str]:
        """Strip userinfo (user:password@) from URLs during serialization.

        ``profile_export_aiperf.json`` is written to the artifact directory and
        uploaded as an MLflow run artifact (see ``MLflowDataExporter`` /
        ``_collect_artifact_files``). Without this serializer, a URL like
        ``http://alice:s3cret@host:8000`` round-trips verbatim into both the
        on-disk export and the MLflow artifact tree.

        The ``include_secrets`` context lets callers (runtime code that needs
        to actually connect) bypass redaction; the HTTP client reads
        ``endpoint.urls`` directly from the in-memory model, not from a
        serialized form, so userinfo stays available where it's needed.
        """
        if info.context and info.context.get("include_secrets"):
            return v
        return [redact_url(url) for url in v]
