# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pydantic import Field

from aiperf.common.models import AIPerfBaseModel
from aiperf.common.types import TransportTypeT


class EndpointMetadata(AIPerfBaseModel):
    """Endpoint metadata for discovery and documentation."""

    endpoint_path: str | None = Field(
        default=None, description="API path (e.g., /v1/chat/completions)."
    )
    streaming_path: str | None = Field(
        default=None,
        description="Streaming API path if different from the endpoint path (e.g., /generate_stream).",
    )
    service_kind: str = Field(
        default="openai",
        description="The service kind of the endpoint (used for artifact naming).",
    )
    supports_streaming: bool = Field(
        default=False, description="Whether endpoint supports streaming responses."
    )
    tokenizes_input: bool = Field(
        default=False, description="Whether endpoint tokenizes text inputs."
    )
    produces_tokens: bool = Field(
        default=False, description="Whether endpoint produces token-based output."
    )
    supports_audio: bool = Field(
        default=False, description="Whether endpoint accepts audio input."
    )
    supports_images: bool = Field(
        default=False, description="Whether endpoint accepts image input."
    )
    supports_videos: bool = Field(
        default=False, description="Whether endpoint accepts video input."
    )
    produces_images: bool = Field(
        default=False, description="Whether endpoint produces image-based outputs."
    )
    metrics_title: str | None = Field(
        default=None, description="Display title for metrics dashboard."
    )


class TransportMetadata(AIPerfBaseModel):
    """Transport metadata for discovery and documentation."""

    transport_type: TransportTypeT = Field(
        description="Transport type identifier for this transport"
    )
    url_schemes: list[str] = Field(
        default_factory=list,
        description="URL schemes this transport handles (for auto-detection and validation).",
    )
