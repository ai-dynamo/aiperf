# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model endpoint information.

This module contains the pydantic models that encapsulate the information needed to
send requests to an inference server, primarily around the model, endpoint, and
additional request payload information.
"""

from typing import Any

from pydantic import Field

from aiperf.common.config import EndpointDefaults, UserConfig
from aiperf.common.enums import EndpointType, ModelSelectionStrategy
from aiperf.common.models import AIPerfBaseModel


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
        default=EndpointType.OPENAI_CHAT_COMPLETIONS,
        description="The type of request payload to use for the endpoint.",
    )
    base_url: str | None = Field(
        default=None,
        description="URL of the endpoint.",
    )
    custom_endpoint: str | None = Field(
        default=None,
        description="Custom endpoint to use for the models.",
    )
    url_params: dict[str, Any] | None = Field(
        default=None, description="Custom URL parameters to use for the endpoint."
    )
    streaming: bool = Field(
        default=False,
        description="Whether the endpoint supports streaming.",
    )
    headers: dict[str, str] | None = Field(
        default=None,
        description="Custom URL headers to use for the endpoint.",
    )
    api_key: str | None = Field(
        default=None,
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
    extra: dict[str, Any] | None = Field(
        default=None,
        description="Additional inputs to include with every request. "
        "You can repeat this flag for multiple inputs. Inputs should be in an 'input_name:value' format. "
        "Alternatively, a string representing a json formatted dict can be provided.",
    )

    @classmethod
    def from_user_config(cls, user_config: UserConfig) -> "EndpointInfo":
        """Create an HttpEndpointInfo from a UserConfig."""
        return cls(
            type=EndpointType(user_config.endpoint.type),
            custom_endpoint=user_config.endpoint.custom_endpoint,
            streaming=user_config.endpoint.streaming,
            base_url=user_config.endpoint.url,
            headers=user_config.input.headers,
            extra=user_config.input.extra,
            timeout=user_config.endpoint.timeout_seconds,
            api_key=user_config.endpoint.api_key,
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

    @classmethod
    def from_user_config(cls, user_config: UserConfig) -> "ModelEndpointInfo":
        """Create a ModelEndpointInfo from a UserConfig."""
        return cls(
            models=ModelListInfo.from_user_config(user_config),
            endpoint=EndpointInfo.from_user_config(user_config),
        )

    @property
    def url(self) -> str:
        """Get the full URL for the endpoint."""
        url = self.endpoint.base_url.rstrip("/") if self.endpoint.base_url else ""
        if self.endpoint.custom_endpoint:
            url += "/" + self.endpoint.custom_endpoint.lstrip("/")
        elif path := self.endpoint.type.endpoint_path:
            url += "/" + path.lstrip("/")
        return url

    @property
    def primary_model(self) -> ModelInfo:
        """Get the primary model."""
        return self.models.models[0]

    @property
    def primary_model_name(self) -> str:
        """Get the primary model name."""
        return self.primary_model.name
