# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""v1 EndpointConfig - validator-free CLI input DTO.

Mirrors the field shape from the pre-v1 EndpointConfig but strips ALL
@field_validator / @model_validator decorators. Domain validation (e.g.
"streaming requires an endpoint type that supports streaming") lives on
AIPerfConfig downstream of the converter.

BeforeValidator metadata (input-shape coercers like `parse_str_or_list`) IS
preserved - those run during cyclopts input parsing, not as domain validators.
"""

from typing import Annotated, Literal

from pydantic import (
    BeforeValidator,
    Field,
)

from aiperf.common.enums import (
    ConnectionReuseStrategy,
    ModelSelectionStrategy,
    RequestContentType,
)
from aiperf.config._base import BaseConfig
from aiperf.config.cli_parameter import CLIParameter, Groups
from aiperf.config.defaults import EndpointDefaults
from aiperf.config.parsing import parse_str_or_list
from aiperf.plugin.enums import (
    EndpointType,
    TransportType,
    URLSelectionStrategy,
)


class EndpointConfig(BaseConfig):
    """A configuration class for defining endpoint related settings."""

    _CLI_GROUP = Groups.ENDPOINT

    model_names: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Model name(s) to be benchmarked. Can be a comma-separated list or a single model name.",
        ),
        BeforeValidator(parse_str_or_list),
        CLIParameter(
            name=(
                "--model-names",
                "--model",  # GenAI-Perf
                "-m",  # GenAI-Perf
            ),
            group=_CLI_GROUP,
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
            name=("--model-selection-strategy",),  # GenAI-Perf
            group=_CLI_GROUP,
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
            group=_CLI_GROUP,
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
            name=("--endpoint-type",),  # GenAI-Perf
            group=_CLI_GROUP,
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
            name=("--streaming",),  # GenAI-Perf
            group=_CLI_GROUP,
        ),
    ] = EndpointDefaults.STREAMING

    urls: Annotated[
        list[str],
        Field(
            description="Base URL(s) of the API server(s) to benchmark. Multiple URLs can be specified for load balancing "
            "across multiple instances (e.g., `--url http://server1:8000 --url http://server2:8000`). "
            "The endpoint path is automatically appended based on `--endpoint-type` (e.g., `/v1/chat/completions` for `chat`).",
            min_length=1,
        ),
        BeforeValidator(parse_str_or_list),
        CLIParameter(
            name=(
                "--url",  # GenAI-Perf
                "-u",  # GenAI-Perf
            ),
            consume_multiple=True,
            group=_CLI_GROUP,
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
            group=_CLI_GROUP,
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
            name=("--request-timeout-seconds",),
            group=_CLI_GROUP,
        ),
    ] = EndpointDefaults.TIMEOUT

    ready_check_timeout: Annotated[
        float,
        Field(
            description=(
                "Seconds to wait for endpoint readiness before benchmarking "
                "(0 = skip). Sends a real inference request to verify the model "
                "is loaded and can generate output."
            ),
        ),
        CLIParameter(
            name=("--ready-check-timeout",),
            group=_CLI_GROUP,
        ),
    ] = 0.0

    ready_check_mode: Annotated[
        Literal["models", "inference", "both"],
        Field(
            description=(
                "How readiness probes the endpoint: 'models' checks /v1/models, "
                "'inference' sends a canned one-token inference request, and "
                "'both' runs the models check before inference."
            ),
        ),
        CLIParameter(
            name=("--ready-check-mode",),
            group=_CLI_GROUP,
        ),
    ] = "inference"

    ready_check_interval: Annotated[
        float,
        Field(
            gt=0.0,
            description="Seconds between endpoint readiness probe attempts.",
        ),
        CLIParameter(
            name=("--ready-check-interval",),
            group=_CLI_GROUP,
        ),
    ] = 5.0

    api_key: Annotated[
        str | None,
        Field(
            description="API authentication key for the endpoint. When provided, automatically included in request headers as "
            "`Authorization: Bearer <api_key>`.",
            repr=False,
        ),
        CLIParameter(
            name=("--api-key",),
            group=_CLI_GROUP,
        ),
    ] = EndpointDefaults.API_KEY

    transport: Annotated[
        TransportType | None,
        Field(
            description="Transport protocol to use for API requests. If not specified, auto-detected from the URL scheme "
            "(`http`/`https` -> `TransportType.HTTP`). Currently supports `http` transport using aiohttp with connection pooling, "
            "TCP optimization, and Server-Sent Events (SSE) for streaming. Explicit override rarely needed.",
        ),
        CLIParameter(
            name=("--transport", "--transport-type"),
            group=_CLI_GROUP,
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
            group=_CLI_GROUP,
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
            group=_CLI_GROUP,
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
            group=_CLI_GROUP,
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
            group=_CLI_GROUP,
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
            group=_CLI_GROUP,
        ),
    ] = EndpointDefaults.REQUEST_CONTENT_TYPE
