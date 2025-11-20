# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import EndpointType
from aiperf.common.factories import EndpointFactory
from aiperf.common.models import (
    BaseResponseData,
    ParsedResponse,
    Turn,
)
from aiperf.common.models.metadata import EndpointMetadata
from aiperf.common.models.record_models import ReasoningResponseData, RequestInfo
from aiperf.common.protocols import EndpointProtocol, InferenceServerResponse
from aiperf.common.types import JsonObject
from aiperf.endpoints.base_endpoint import BaseEndpoint

_DEFAULT_ROLE: str = "user"


@implements_protocol(EndpointProtocol)
@EndpointFactory.register(EndpointType.CHAT)
class ChatEndpoint(BaseEndpoint):
    """OpenAI Chat Completions endpoint.

    Supports multi-modal inputs (text, images, audio, video) and both
    streaming and non-streaming responses.
    """

    @classmethod
    def metadata(cls) -> EndpointMetadata:
        """Return Chat Completions endpoint metadata."""
        return EndpointMetadata(
            endpoint_path="/v1/chat/completions",
            supports_streaming=True,
            produces_tokens=True,
            tokenizes_input=True,
            supports_audio=True,
            supports_images=True,
            supports_videos=True,
            metrics_title="LLM Metrics",
        )

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format OpenAI Chat Completions request payload from RequestInfo.

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            OpenAI Chat Completions API payload
        """
        if not request_info.turns:
            raise ValueError("Chat endpoint requires at least one turn.")

        turns = request_info.turns
        model_endpoint = request_info.model_endpoint
        messages = self._create_messages(turns)

        payload = {
            "messages": messages,
            "model": turns[-1].model or model_endpoint.primary_model_name,
            "stream": model_endpoint.endpoint.streaming,
        }

        if turns[-1].max_tokens is not None:
            token_field = (
                "max_tokens"
                if model_endpoint.endpoint.use_legacy_max_tokens
                else "max_completion_tokens"
            )
            payload[token_field] = turns[-1].max_tokens

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        self.debug(lambda: f"Formatted payload: {payload}")
        return payload

    def _create_messages(self, turns: list[Turn]) -> list[dict[str, Any]]:
        """Create messages from turns for OpenAI Chat Completions."""
        messages = []
        for turn in turns:
            message = {
                "role": turn.role or _DEFAULT_ROLE,
            }
            self._set_message_content(message, turn)
            messages.append(message)
        return messages

    def _set_message_content(self, message: dict[str, Any], turn: Turn) -> None:
        """Create message content from turn for OpenAI Chat Completions."""
        if (
            len(turn.texts) == 1
            and len(turn.texts[0].contents) == 1
            and len(turn.images) == 0
            and len(turn.audios) == 0
            and len(turn.videos) == 0
        ):
            # Hotfix for Dynamo API which does not yet support a list of messages
            message["name"] = turn.texts[0].name
            message["content"] = (
                turn.texts[0].contents[0] if turn.texts[0].contents else ""
            )
            return

        message_content: list[dict[str, Any]] = []

        for text in turn.texts:
            for content in text.contents:
                if not content:
                    continue
                message_content.append({"type": "text", "text": content})

        for image in turn.images:
            for content in image.contents:
                if not content:
                    continue
                message_content.append(
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
                message_content.append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": b64_audio,
                            "format": format,
                        },
                    }
                )
        for video in turn.videos:
            for content in video.contents:
                if not content:
                    continue
                message_content.append(
                    {"type": "video_url", "video_url": {"url": content}}
                )

        message["content"] = message_content

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse OpenAI Chat Completions response.

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with extracted text/reasoning content and usage data
        """
        json_obj = response.get_json()
        if not json_obj:
            return None

        data = self.extract_chat_response_data(json_obj)
        usage = json_obj.get("usage") or None

        if data or usage:
            return ParsedResponse(perf_ns=response.perf_ns, data=data, usage=usage)

        return None

    def extract_chat_response_data(
        self, json_obj: JsonObject
    ) -> BaseResponseData | None:
        """Extract content from OpenAI JSON response.

        Handles both streaming (chat.completion.chunk) and non-streaming
        (chat.completion) formats using pattern matching.

        Args:
            json_obj: Deserialized OpenAI response

        Returns:
            Extracted response data or None if no content
        """
        match json_obj.get("object"):
            case "chat.completion":
                data_key = "message"
            case "chat.completion.chunk":
                data_key = "delta"
            case _:
                object_type = json_obj.get("object")
                raise ValueError(f"Unsupported OpenAI object type: {object_type!r}")

        choices = json_obj.get("choices")
        if not choices:
            self.debug(lambda: f"No choices found in response: {json_obj}")
            return None

        data = choices[0].get(data_key)
        if not data:
            self.debug(lambda: f"No data found in response: {json_obj}")
            return None

        content = data.get("content")
        reasoning = data.get("reasoning_content") or data.get("reasoning")
        if not content and not reasoning:
            return None

        if not reasoning:
            return self.make_text_response_data(content)

        return ReasoningResponseData(
            content=content,
            reasoning=reasoning,
        )
