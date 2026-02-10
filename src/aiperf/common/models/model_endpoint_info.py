# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model endpoint information.

This module contains the pydantic models that encapsulate the information needed to
send requests to an inference server, primarily around the model, endpoint, and
additional request payload information.
"""

from typing import Any

from pydantic import Field

from aiperf.common.config import EndpointDefaults, UserConfig
from aiperf.common.enums import ConnectionReuseStrategy, ModelSelectionStrategy
from aiperf.common.models import AIPerfBaseModel
from aiperf.plugin.enums import EndpointType, TransportType


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

    @classmethod
    def from_user_config(cls, user_config: UserConfig) -> "ModelListInfo":
        """Create a ModelListInfo from a UserConfig."""
        return cls(
            models=[
                ModelInfo(name=model) for model in user_config.endpoint.model_names
            ],
            model_selection_strategy=user_config.endpoint.model_selection_strategy,
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
    api_key: str | None = Field(
        default=EndpointDefaults.API_KEY,
        description="API key to use for the endpoint.",
    )
    ssl_options: dict[str, Any] | None = Field(
        default=None,
        description="SSL options to use for the endpoint.",
    )
    timeout: float = Field(
        default=EndpointDefaults.TIMEOUT,
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

    @classmethod
    def from_user_config(cls, user_config: UserConfig) -> "EndpointInfo":
        """Create an EndpointInfo from a UserConfig."""
        return cls(
            type=user_config.endpoint.type,
            custom_endpoint=user_config.endpoint.custom_endpoint,
            streaming=user_config.endpoint.streaming,
            base_urls=user_config.endpoint.urls,
            headers=user_config.input.headers,
            extra=user_config.input.extra,
            timeout=user_config.endpoint.timeout_seconds,
            api_key=user_config.endpoint.api_key,
            use_legacy_max_tokens=user_config.endpoint.use_legacy_max_tokens,
            use_server_token_count=user_config.endpoint.use_server_token_count,
            connection_reuse_strategy=user_config.endpoint.connection_reuse_strategy,
            download_video_content=user_config.endpoint.download_video_content,
        )


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
    def from_user_config(cls, user_config: UserConfig) -> "ModelEndpointInfo":
        """Create a ModelEndpointInfo from a UserConfig."""
        return cls(
            models=ModelListInfo.from_user_config(user_config),
            endpoint=EndpointInfo.from_user_config(user_config),
            transport=user_config.endpoint.transport,
        )

    @property
    def primary_model(self) -> ModelInfo:
        """Get the primary model."""
        return self.models.models[0]

    @property
    def primary_model_name(self) -> str:
        """Get the primary model name."""
        return self.primary_model.name
