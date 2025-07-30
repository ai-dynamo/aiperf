# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from aiperf.clients.client_interfaces import (
    RequestConverterFactory,
    RequestConverterProtocol,
)
from aiperf.clients.model_endpoint_info import ModelEndpointInfo
from aiperf.common.enums import EndpointType
from aiperf.common.models import Turn

DEFAULT_ROLE = "user"


@RequestConverterFactory.register(EndpointType.OPENAI_CHAT_COMPLETIONS)
class OpenAIChatCompletionRequestConverter(RequestConverterProtocol[dict[str, Any]]):
    """Request converter for OpenAI chat completion requests."""

    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    async def format_payload(
        self,
        model_endpoint: ModelEndpointInfo,
        turn: Turn,
    ) -> dict[str, Any]:
        """Format payload for a chat completion request."""

        message = await self._create_message(turn)

        payload = {
            "messages": [message],
            "model": model_endpoint.primary_model_name,
            "stream": model_endpoint.endpoint.streaming,
        }

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        self.logger.debug("Formatted payload: %s", payload)
        return payload

    def _create_message(self, turn: Turn) -> dict[Any, Any]:
        message = {
            "role": turn.role or DEFAULT_ROLE,
            "content": [],
        }
        for text in turn.texts:
            for content in text.contents:
                if not content:
                    continue
                message["content"].append({"type": "text", "text": content})

        for image in turn.images:
            for content in image.contents:
                if not content:
                    continue
                message["content"].append(
                    {"type": "image_url", "image_url": {"url": content}}
                )

        for audio in turn.audios:
            for content in audio.contents:
                if not content:
                    continue
                if "," not in content:
                    raise ValueError(
                        "Audio content must be in the format 'format,b64_audio'."
                    )
                format, b64_audio = content.split(",", 1)
                message["content"].append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": b64_audio,
                            "format": format,
                        },
                    }
                )

        return message
