# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from aiperf.common.models import (
    InferenceServerResponse,
    ParsedResponse,
    ReasoningResponseData,
    RequestInfo,
    TextResponseData,
    Turn,
)
from aiperf.common.types import JsonObject
from aiperf.endpoints.base_endpoint import BaseEndpoint

_DEFAULT_ROLE: str = "user"


class ResponsesEndpoint(BaseEndpoint):
    """OpenAI Responses API endpoint.

    Supports multi-modal inputs (text, images, audio) and both
    streaming and non-streaming responses.
    """

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format OpenAI Responses API request payload from RequestInfo.

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            OpenAI Responses API payload
        """
        if not request_info.turns:
            raise ValueError("Responses endpoint requires at least one turn.")

        turns = request_info.turns
        model_endpoint = request_info.model_endpoint

        input_items = self._create_input_items(turns, request_info.user_context_message)

        payload: dict[str, Any] = {
            "input": input_items,
            "model": turns[-1].model or model_endpoint.primary_model_name,
            "stream": model_endpoint.endpoint.streaming,
        }

        if request_info.system_message:
            payload["instructions"] = request_info.system_message

        if turns[-1].max_tokens is not None:
            payload["max_output_tokens"] = turns[-1].max_tokens

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        if (
            model_endpoint.endpoint.streaming
            and model_endpoint.endpoint.use_server_token_count
        ):
            if "stream_options" not in payload or not isinstance(
                payload["stream_options"], dict
            ):
                payload["stream_options"] = {"include_usage": True}
            elif "include_usage" not in payload["stream_options"]:
                payload["stream_options"]["include_usage"] = True

        self.trace(lambda: f"Formatted payload: {payload}")
        return payload

    def _create_input_items(
        self,
        turns: list[Turn],
        user_context_message: str | None,
    ) -> list[dict[str, Any]]:
        """Create input items from turns for OpenAI Responses API.

        Args:
            turns: List of turns in the request
            user_context_message: Optional per-conversation user context to prepend

        Returns:
            List of formatted input item dicts for OpenAI Responses API
        """
        items: list[dict[str, Any]] = []

        if user_context_message:
            items.append(
                {
                    "role": _DEFAULT_ROLE,
                    "content": user_context_message,
                }
            )

        for turn in turns:
            item: dict[str, Any] = {
                "role": turn.role or _DEFAULT_ROLE,
            }
            self._set_item_content(item, turn)
            items.append(item)
        return items

    def _set_item_content(self, item: dict[str, Any], turn: Turn) -> None:
        """Create input item content from turn for OpenAI Responses API."""
        if (
            len(turn.texts) == 1
            and len(turn.texts[0].contents) == 1
            and len(turn.images) == 0
            and len(turn.audios) == 0
            and len(turn.videos) == 0
        ):
            item["content"] = turn.texts[0].contents[0]
            return

        content: list[dict[str, Any]] = []

        for text in turn.texts:
            for c in text.contents:
                if not c:
                    continue
                content.append({"type": "input_text", "text": c})

        for image in turn.images:
            for c in image.contents:
                if not c:
                    continue
                content.append({"type": "input_image", "image_url": c})

        for audio in turn.audios:
            for c in audio.contents:
                if not c:
                    continue
                if "," not in c:
                    raise ValueError(
                        "Audio content must be in the format 'format,b64_audio'."
                    )
                fmt, b64_audio = c.split(",", 1)
                content.append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": b64_audio,
                            "format": fmt,
                        },
                    }
                )

        item["content"] = content

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse OpenAI Responses API response.

        Handles both streaming SSE events (with ``"type"`` field) and
        non-streaming responses (with ``"object": "response"``).

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with extracted text/reasoning content and usage data
        """
        json_obj = response.get_json()
        if not json_obj:
            return None

        # Streaming: events have a "type" field
        if "type" in json_obj:
            return self._parse_streaming_event(json_obj, response.perf_ns)

        # Non-streaming: full response object
        if json_obj.get("object") == "response":
            return self._parse_full_response(json_obj, response.perf_ns)

        return None

    def _parse_streaming_event(
        self, json_obj: JsonObject, perf_ns: int
    ) -> ParsedResponse | None:
        """Parse a streaming SSE event from the Responses API.

        Args:
            json_obj: Deserialized event JSON
            perf_ns: Performance timestamp

        Returns:
            Parsed response or None if the event carries no content
        """
        event_type = json_obj.get("type")

        if event_type == "response.output_text.delta":
            delta = json_obj.get("delta")
            if delta:
                return ParsedResponse(
                    perf_ns=perf_ns,
                    data=TextResponseData(text=delta),
                )
            return None

        if event_type == "response.reasoning_text.delta":
            delta = json_obj.get("delta")
            if delta:
                return ParsedResponse(
                    perf_ns=perf_ns,
                    data=ReasoningResponseData(reasoning=delta),
                )
            return None

        if event_type == "response.output_text.done":
            text = json_obj.get("text")
            if text:
                return ParsedResponse(
                    perf_ns=perf_ns,
                    data=TextResponseData(text=text),
                )
            return None

        if event_type == "response.completed":
            resp = json_obj.get("response") or {}
            usage = resp.get("usage") or None
            if usage:
                return ParsedResponse(perf_ns=perf_ns, data=None, usage=usage)
            return None

        # All other events (response.created, response.in_progress, etc.)
        return None

    def _parse_full_response(
        self, json_obj: JsonObject, perf_ns: int
    ) -> ParsedResponse | None:
        """Parse a non-streaming full response from the Responses API.

        Args:
            json_obj: Deserialized response JSON
            perf_ns: Performance timestamp

        Returns:
            Parsed response or None if no content found
        """
        usage = json_obj.get("usage") or None
        data = self._extract_response_content(json_obj)

        if data or usage:
            return ParsedResponse(perf_ns=perf_ns, data=data, usage=usage)

        return None

    def _extract_response_content(
        self, json_obj: JsonObject
    ) -> TextResponseData | ReasoningResponseData | None:
        """Extract content from a non-streaming Responses API response.

        Looks for output items with ``type: "message"`` (text content) and
        ``type: "reasoning"`` (summary text). Falls back to the top-level
        ``output_text`` convenience field.

        Args:
            json_obj: Deserialized response JSON

        Returns:
            Extracted response data or None
        """
        output = json_obj.get("output")
        if isinstance(output, list):
            text_parts: list[str] = []
            reasoning_parts: list[str] = []

            for item in output:
                if not isinstance(item, dict):
                    continue

                item_type = item.get("type")

                if item_type == "reasoning":
                    for part in item.get("summary", []):
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") == "summary_text" and part.get("text"):
                            reasoning_parts.append(part["text"])

                elif item_type == "message":
                    for part in item.get("content", []):
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") == "output_text" and part.get("text"):
                            text_parts.append(part["text"])

            if reasoning_parts:
                return ReasoningResponseData(
                    content="".join(text_parts) or None,
                    reasoning="".join(reasoning_parts),
                )
            if text_parts:
                return TextResponseData(text="".join(text_parts))

        # Fallback: top-level output_text convenience field
        output_text = json_obj.get("output_text")
        if output_text:
            return TextResponseData(text=output_text)

        return None
