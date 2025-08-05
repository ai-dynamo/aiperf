# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from aiperf.clients.model_endpoint_info import ModelEndpointInfo
from aiperf.common.enums import EndpointType
from aiperf.common.factories import RequestConverterFactory
from aiperf.common.mixins.aiperf_logger_mixin import AIPerfLoggerMixin
from aiperf.common.models import Turn


@RequestConverterFactory.register(EndpointType.OPENAI_EMBEDDINGS)
class OpenAIEmbeddingsRequestConverter(AIPerfLoggerMixin):
    """Request converter for OpenAI embeddings requests."""

    async def format_payload(
        self,
        model_endpoint: ModelEndpointInfo,
        turn: Turn,
    ) -> dict[str, Any]:
        """Format payload for an embeddings request."""

        prompts = [
            content for text in turn.texts for content in text.contents if content
        ]

        extra = model_endpoint.endpoint.extra or {}

        payload = {
            "model": turn.model or model_endpoint.primary_model_name,
            "input": prompts,
        }

        if extra:
            payload.update(extra)

        self.trace(lambda: f"Formatted payload: {payload}")
        return payload
