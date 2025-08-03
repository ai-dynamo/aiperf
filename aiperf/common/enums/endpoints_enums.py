# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field
from typing_extensions import Self

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class EndpointTypeInfo(BaseModel):
    """Pydantic model for endpoint-specific metadata. This model is used to store additional info on each EndpointType enum value."""

    tag: str = Field(
        ..., min_length=1, description="The string value for the endpoint type."
    )
    supports_streaming: bool = Field(
        ..., description="True if the endpoint supports streaming, False otherwise."
    )
    produces_tokens: bool = Field(
        ..., description="True if the endpoint produces tokens, False otherwise."
    )
    endpoint_path: str | None = Field(
        None,
        description="The default URL path for the endpoint. If None, the endpoint does not have a specific path.",
    )
    metrics_title: str | None = Field(
        None,
        description="The title string for the metrics table. If None, the default title is used.",
    )


class EndpointType(CaseInsensitiveStrEnum):
    """Endpoint types supported by AIPerf.

    These are the full definitions of the endpoints that are supported by AIPerf.
    Each enum value contains additional metadata about the endpoint, such as whether it supports streaming,
    produces tokens, and the default endpoint path. This is stored as an attribute on the enum value, and can be accessed
    via the `info` property. The enum values can still be used as strings for user input and comparison (via the `tag` field).
    """

    OPENAI_CHAT_COMPLETIONS = EndpointTypeInfo(
        tag="chat",
        supports_streaming=True,
        produces_tokens=True,
        endpoint_path="/v1/chat/completions",
        metrics_title="LLM Metrics",
    )
    OPENAI_COMPLETIONS = EndpointTypeInfo(
        tag="completions",
        supports_streaming=True,
        produces_tokens=True,
        endpoint_path="/v1/completions",
        metrics_title="LLM Metrics",
    )
    OPENAI_EMBEDDINGS = EndpointTypeInfo(
        tag="embeddings",
        supports_streaming=False,
        produces_tokens=False,
        endpoint_path="/v1/embeddings",
        metrics_title="Embeddings Metrics",
    )
    OPENAI_RESPONSES = EndpointTypeInfo(
        tag="responses",
        supports_streaming=True,
        produces_tokens=True,
        endpoint_path="/v1/responses",
        metrics_title="LLM Metrics",
    )

    # Override the __new__ method to store the Pydantic `EndpointTypeInfo` model as an attribute. This is a python feature that
    # allows us to modify the behavior of the enum class's constructor. We use this to ensure the the enums still look like
    # a regular string enum, but also have the additional information stored as an attribute.
    def __new__(cls, endpoint_info: EndpointTypeInfo) -> Self:
        obj = str.__new__(cls, endpoint_info.tag)
        # Ensure string value is set for comparison. This is the how enums work internally.
        obj._value_ = endpoint_info.tag
        # Store the Pydantic model as an attribute
        obj._info: EndpointTypeInfo = endpoint_info  # type: ignore
        return obj

    @property
    def info(self) -> EndpointTypeInfo:
        """Get the endpoint info for the endpoint type."""
        # This is the Pydantic model that was stored as an attribute in the __new__ method.
        return self._info  # type: ignore

    @property
    def supports_streaming(self) -> bool:
        """Return True if the endpoint supports streaming. This is used for validation of user input."""
        return self.info.supports_streaming

    @property
    def produces_tokens(self) -> bool:
        """Return True if the endpoint produces tokens. This is used to determine what metrics are applicable to the endpoint."""
        return self.info.produces_tokens

    @property
    def endpoint_path(self) -> str | None:
        """Get the default endpoint path for the endpoint type. If None, the endpoint does not have a specific path."""
        return self.info.endpoint_path

    @property
    def metrics_title(self) -> str:
        """Get the metrics table title string for the endpoint type. If None, the default title is used."""
        return self.info.metrics_title or "LLM Metrics"
