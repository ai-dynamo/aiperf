# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import BeforeValidator, Field, model_validator
from typing_extensions import Self

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.common.config.config_defaults import EndpointDefaults
from aiperf.common.config.config_validators import parse_str_or_list
from aiperf.common.config.groups import Groups
from aiperf.common.enums import (
    ConnectionReuseStrategy,
    ModelSelectionStrategy,
)
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

    _CLI_GROUP = Groups.ENDPOINT

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
            name=(
                "--model-selection-strategy",  # GenAI-Perf
            ),
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
            name=(
                "--endpoint-type",  # GenAI-Perf
            ),
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
            name=(
                "--streaming",  # GenAI-Perf
            ),
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
            name=("--request-timeout-seconds"),
            group=_CLI_GROUP,
        ),
    ] = EndpointDefaults.TIMEOUT

    api_key: Annotated[
        str | None,
        Field(
            description="API authentication key for the endpoint. When provided, automatically included in request headers as "
            "`Authorization: Bearer <api_key>`.",
        ),
        CLIParameter(
            name=("--api-key"),
            group=_CLI_GROUP,
        ),
    ] = EndpointDefaults.API_KEY

    transport: Annotated[
        TransportType | None,
        Field(
            description="Transport protocol to use for API requests. If not specified, auto-detected from the URL scheme "
            "(`http`/`https` → `TransportType.HTTP`). Currently supports `http` transport using aiohttp with connection pooling, "
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
