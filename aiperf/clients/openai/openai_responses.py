# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from aiperf.clients.client_interfaces import (
    RequestConverterFactory,
    RequestConverterProtocol,
)
from aiperf.clients.model_endpoint_info import ModelEndpointInfo
from aiperf.common.dataset_models import Turn
from aiperf.common.enums import EndpointType


# TODO: Not fully implemented yet.
@RequestConverterFactory.register(EndpointType.OPENAI_RESPONSES)
class OpenAIResponsesRequestConverter(RequestConverterProtocol[dict[str, Any]]):
    """Request converter for OpenAI Responses requests."""

    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    async def format_payload(
        self,
        model_endpoint: ModelEndpointInfo,
        turn: Turn,
    ) -> dict[str, Any]:
        """Format payload for a responses request."""

        # TODO: Add support for image and audio inputs.
        prompts = [content for text in turn.text for content in text.content if content]

        extra = model_endpoint.endpoint.extra or {}

        payload = {
            "input": prompts,
            "model": model_endpoint.primary_model_name,
            # TODO: How do we handle max_output_tokens? Should be provided by OSL logic
            "max_output_tokens": extra.pop("max_output_tokens", None),
            "stream": model_endpoint.endpoint.streaming,
        }

        if extra:
            payload.update(extra)

        self.logger.debug("Formatted payload: %s", payload)
        return payload
