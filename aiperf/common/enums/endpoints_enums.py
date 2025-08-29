# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import cached_property

from pydantic import Field

from aiperf.common.enums.base_enums import (
    BasePydanticBackedStrEnum,
    BasePydanticEnumInfo,
    CaseInsensitiveStrEnum,
)


class EndpointServiceKind(CaseInsensitiveStrEnum):
    """Endpoint service kind."""

    OPENAI = "openai"


class EndpointTypeInfo(BasePydanticEnumInfo):
    """Pydantic model for endpoint-specific metadata. This model is used to store additional info on each EndpointType enum value.

    For documentation on the fields, see the :class:`EndpointType` @property functions.
    """

    service_kind: EndpointServiceKind = Field(...)
    supports_streaming: bool = Field(...)
    produces_tokens: bool = Field(...)
    supports_audio: bool = Field(default=False)
    supports_images: bool = Field(default=False)
    endpoint_path: str | None = Field(default=None)
    metrics_title: str | None = Field(default=None)


class EndpointType(BasePydanticBackedStrEnum):
    """Endpoint types supported by AIPerf.

    These are the full definitions of the endpoints that are supported by AIPerf.
    Each enum value contains additional metadata about the endpoint, such as whether it supports streaming,
    produces tokens, and the default endpoint path. This is stored as an attribute on the enum value, and can be accessed
    via the `info` property. The enum values can still be used as strings for user input and comparison (via the `tag` field).
    """

    OPENAI_CHAT_COMPLETIONS = EndpointTypeInfo(
        tag="chat",
        service_kind=EndpointServiceKind.OPENAI,
        supports_streaming=True,
        produces_tokens=True,
        supports_audio=True,
        supports_images=True,
        endpoint_path="/v1/chat/completions",
        metrics_title="LLM Metrics",
    )
    OPENAI_COMPLETIONS = EndpointTypeInfo(
        tag="completions",
        service_kind=EndpointServiceKind.OPENAI,
        supports_streaming=True,
        produces_tokens=True,
        endpoint_path="/v1/completions",
        metrics_title="LLM Metrics",
    )
    OPENAI_EMBEDDINGS = EndpointTypeInfo(
        tag="embeddings",
        service_kind=EndpointServiceKind.OPENAI,
        supports_streaming=False,
        produces_tokens=False,
        endpoint_path="/v1/embeddings",
        metrics_title="Embeddings Metrics",
    )
    RANKINGS = EndpointTypeInfo(
        tag="rankings",
        service_kind=EndpointServiceKind.OPENAI,
        supports_streaming=False,
        produces_tokens=False,
        endpoint_path="/v1/ranking",
        metrics_title="Rankings Metrics",
    )
    OPENAI_RESPONSES = EndpointTypeInfo(
        tag="responses",
        service_kind=EndpointServiceKind.OPENAI,
        supports_streaming=True,
        produces_tokens=True,
        supports_audio=False,  # Not yet supported by OpenAI
        supports_images=True,
        endpoint_path="/v1/responses",
        metrics_title="LLM Metrics",
    )

    @cached_property
    def info(self) -> EndpointTypeInfo:
        """Get the endpoint info for the endpoint type."""
        return self._info  # type: ignore

    @property
    def service_kind(self) -> EndpointServiceKind:
        """Get the service kind for the endpoint type."""
        return self.info.service_kind

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
    def supports_audio(self) -> bool:
        """Return True if the endpoint supports audio input.
        This is used to determine what metrics are applicable to the endpoint, as well as what inputs can be used."""
        return self.info.supports_audio

    @property
    def supports_images(self) -> bool:
        """Return True if the endpoint supports image input.
        This is used to determine what metrics are applicable to the endpoint, as well as what inputs can be used."""
        return self.info.supports_images

    @property
    def metrics_title(self) -> str:
        """Get the metrics table title string for the endpoint type. If None, the default title is used."""
        return self.info.metrics_title or "Metrics"
