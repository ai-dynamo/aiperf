# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.common.models import (
    BaseResponseData,
    InferenceServerResponse,
    ParsedResponse,
    ReasoningResponseData,
    RequestInfo,
    Turn,
)
from aiperf.common.types import JsonObject
from aiperf.endpoints.base_endpoint import BaseEndpoint

_DEFAULT_ROLE: str = "user"
_FAST_PARSE_FALLBACK = object()


class ChatEndpoint(BaseEndpoint):
    _FAST_PARSE_FALLBACK = _FAST_PARSE_FALLBACK
    """OpenAI Chat Completions endpoint.

    Supports multi-modal inputs (text, images, audio, video) and both
    streaming and non-streaming responses.
    """

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
        model_endpoint = self.run.cfg

        if turns[-1].raw_messages is not None:
            messages = turns[-1].raw_messages
        else:
            messages = self._create_messages(
                turns, request_info.system_message, request_info.user_context_message
            )

        payload = {
            "messages": messages,
            "model": turns[-1].model or model_endpoint.get_model_names()[0],
            "stream": model_endpoint.endpoint.streaming,
        }

        if turns[-1].raw_tools is not None:
            payload["tools"] = turns[-1].raw_tools

        if turns[-1].max_tokens is not None:
            token_field = (
                "max_tokens"
                if model_endpoint.endpoint.use_legacy_max_tokens
                else "max_completion_tokens"
            )
            payload[token_field] = turns[-1].max_tokens

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        if (
            model_endpoint.endpoint.streaming
            and model_endpoint.endpoint.use_server_token_count
        ):
            # Automatically set stream_options to include usage when using server token counts
            if "stream_options" not in payload:
                payload["stream_options"] = {"include_usage": True}
            elif (
                isinstance(payload["stream_options"], dict)
                and "include_usage" not in payload["stream_options"]
            ):
                payload["stream_options"]["include_usage"] = True

        self.trace(lambda: f"Formatted payload: {payload}")
        return payload

    def _create_messages(
        self,
        turns: list[Turn],
        system_message: str | None,
        user_context_message: str | None,
    ) -> list[dict[str, Any]]:
        """Create messages from turns for OpenAI Chat Completions.

        Args:
            turns: List of turns in the request
            system_message: Optional shared system message to prepend
            user_context_message: Optional per-conversation user context to prepend

        Returns:
            List of formatted message dicts for OpenAI Chat Completions API
        """
        messages = []

        # Prepend system_message and user_context_message if present
        if system_message:
            messages.append(
                {
                    "role": "system",
                    "content": system_message,
                }
            )

        if user_context_message:
            messages.append(
                {
                    "role": "user",
                    "content": user_context_message,
                }
            )

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
            message["content"] = (
                turn.texts[0].contents[0] if turn.texts[0].contents else ""
            )
            return

        message_content: list[dict[str, Any]] = []
        self._append_text_parts(message_content, turn)
        self._append_image_parts(message_content, turn)
        self._append_audio_parts(message_content, turn)
        self._append_video_parts(message_content, turn)
        message["content"] = message_content

    @staticmethod
    def _append_text_parts(parts: list[dict[str, Any]], turn: Turn) -> None:
        for text in turn.texts:
            for content in text.contents:
                if not content:
                    continue
                parts.append({"type": "text", "text": content})

    @staticmethod
    def _append_image_parts(parts: list[dict[str, Any]], turn: Turn) -> None:
        for image in turn.images:
            for content in image.contents:
                if not content:
                    continue
                parts.append({"type": "image_url", "image_url": {"url": content}})

    @staticmethod
    def _append_audio_parts(parts: list[dict[str, Any]], turn: Turn) -> None:
        for audio in turn.audios:
            for content in audio.contents:
                if not content:
                    continue
                if "," not in content:
                    raise ValueError(
                        "Audio content must be in the format 'format,b64_audio'."
                    )
                format, b64_audio = content.split(",", 1)
                parts.append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": b64_audio,
                            "format": format,
                        },
                    }
                )

    @staticmethod
    def _append_video_parts(parts: list[dict[str, Any]], turn: Turn) -> None:
        for video in turn.videos:
            for content in video.contents:
                if not content:
                    continue
                parts.append({"type": "video_url", "video_url": {"url": content}})

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

        fast_parsed = self._fast_parse_response(json_obj, response.perf_ns)
        if fast_parsed is not self._FAST_PARSE_FALLBACK:
            return fast_parsed

        data = self.extract_chat_response_data(json_obj)
        usage = json_obj.get("usage") or None

        if data or usage:
            return ParsedResponse(perf_ns=response.perf_ns, data=data, usage=usage)

        return None

    def _fast_parse_response(
        self,
        json_obj: JsonObject,
        perf_ns: int,
    ) -> ParsedResponse | None | object:
        """Fast-path the common OpenAI chat shapes and fall back for anything unusual."""
        try:
            data_key = self._fast_parse_data_key(json_obj)
            if data_key is None:
                return self._FAST_PARSE_FALLBACK

            choices = json_obj.get("choices")
            if not choices:
                return self._usage_only_response(json_obj, perf_ns)

            first_choice = choices[0]
            if not isinstance(first_choice, dict):
                return self._FAST_PARSE_FALLBACK

            data = first_choice.get(data_key)
            if not isinstance(data, dict):
                return None

            return self._build_fast_parsed(data, json_obj, perf_ns)
        except (IndexError, KeyError, TypeError):
            return self._FAST_PARSE_FALLBACK

    @staticmethod
    def _fast_parse_data_key(json_obj: JsonObject) -> str | None:
        object_type = json_obj.get("object")
        if object_type == "chat.completion":
            return "message"
        if object_type == "chat.completion.chunk":
            return "delta"
        return None

    @staticmethod
    def _usage_only_response(
        json_obj: JsonObject, perf_ns: int
    ) -> ParsedResponse | None:
        # Final usage-only chunk (stream_options.include_usage=true)
        # has empty choices but carries the cumulative usage totals.
        usage = json_obj.get("usage") or None
        if usage is not None:
            return ParsedResponse(perf_ns=perf_ns, data=None, usage=usage)
        return None

    def _build_fast_parsed(
        self, data: dict[str, Any], json_obj: JsonObject, perf_ns: int
    ) -> ParsedResponse | None:
        content = data.get("content")
        reasoning = data.get("reasoning_content") or data.get("reasoning")
        usage = json_obj.get("usage") or None

        if not content and not reasoning and not usage:
            return None

        if reasoning:
            response_data: BaseResponseData | None = ReasoningResponseData(
                content=content,
                reasoning=reasoning,
            )
        else:
            response_data = self.make_text_response_data(content)

        if response_data or usage:
            return ParsedResponse(perf_ns=perf_ns, data=response_data, usage=usage)
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
