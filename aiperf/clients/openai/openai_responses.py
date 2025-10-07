# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from aiperf.clients.model_endpoint_info import ModelEndpointInfo
from aiperf.common.enums import EndpointType
from aiperf.common.factories import RequestConverterFactory
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import Turn


# TODO: Not fully implemented yet.
@RequestConverterFactory.register(EndpointType.RESPONSES)
class OpenAIResponsesRequestConverter(AIPerfLoggerMixin):
    """Request converter for OpenAI Responses requests."""

    async def format_payload(
        self,
        model_endpoint: ModelEndpointInfo,
        turns: list[Turn],
    ) -> dict[str, Any]:
        """Format payload for a responses request."""

        # This converter does not support mutli-turn responses. Only the last turn is used.
        # When this is fully supported, we can remove this limitation.
        turn = turns[-1]

        # TODO: Add support for image and audio inputs.
        prompts = [
            content for text in turn.texts for content in text.contents if content
        ]

        extra = model_endpoint.endpoint.extra or []

        payload = {
            "input": prompts,
            "model": turn.model or model_endpoint.primary_model_name,
            "stream": model_endpoint.endpoint.streaming,
        }
        if turn.max_tokens:
            payload["max_output_tokens"] = turn.max_tokens

        if extra:
            payload.update(extra)

        self.debug(lambda: f"Formatted payload: {payload}")
        return payload
