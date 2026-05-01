# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, ClassVar

from aiperf.common.enums import MediaType
from aiperf.common.models import (
    InferenceServerResponse,
    ParsedResponse,
    ReasoningResponseData,
    RequestInfo,
    TextResponseData,
)
from aiperf.common.types import JsonObject
from aiperf.endpoints.base_endpoint import BaseEndpoint


class ResponsesEndpoint(BaseEndpoint):
    """OpenAI Responses API endpoint.

    Message-array construction reuses the generic
    ``BaseEndpoint.build_messages`` flow. Only the content-part type names
    differ from chat (``input_text`` vs ``text``, ``input_image`` vs
    ``image_url``), so we override those hooks and leave the iteration /
    raw-messages pass-through skeleton alone.

    The shared ``system_message`` lives on the top-level ``instructions``
    field rather than inside the ``input`` array (Responses API contract),
    and the per-conversation ``user_context_message`` is prepended as a
    leading user item.
    """

    # Responses API content-part type names. ``BaseEndpoint.extract_payload_inputs``
    # walks the payload once and dispatches every part against this map —
    # text parts contribute to the tokenisable text list, media parts
    # bump their respective counts.
    PART_TYPES: ClassVar[dict[MediaType, set[str]]] = {
        MediaType.TEXT: {"input_text"},
        MediaType.IMAGE: {"input_image"},
        MediaType.AUDIO: {"input_audio"},
        # Responses API does not currently support video input.
        MediaType.VIDEO: set(),
    }

    def extract_payload_inputs(self, payload: dict[str, Any]):
        """Responses-API single-pass extraction.

        Inherits the base-class walk (which dispatches content parts via
        ``PART_TYPES``) and additionally prepends ``instructions`` — the
        Responses-API equivalent of a system prompt that lives at the
        top level of the payload rather than inside ``input``.
        """
        result = super().extract_payload_inputs(payload)
        instructions = payload.get("instructions")
        if isinstance(instructions, str):
            result.texts.insert(0, instructions)
            if result.messages is not None:
                result.messages.insert(0, {"role": "system", "content": instructions})
        return result

    # --- Content-part hooks (override only the type names) -------------------

    def _render_text_part(self, text: str) -> dict[str, Any]:
        return {"type": "input_text", "text": text}

    def _render_image_part(self, url_or_data_uri: str) -> dict[str, Any]:
        # Responses API takes ``image_url`` as a plain string, not nested.
        return {"type": "input_image", "image_url": url_or_data_uri}

    def _render_audio_part(self, format_and_b64: str) -> dict[str, Any]:
        if "," not in format_and_b64:
            raise ValueError("Audio content must be in the format 'format,b64_audio'.")
        fmt, b64 = format_and_b64.split(",", 1)
        return {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}}

    # NOTE: Responses API does not currently support video input.
    # ``_render_video_part`` inherits the chat default and would only fire
    # if a caller authored video turns against a Responses endpoint — the
    # default output shape is structurally valid but the server will reject
    # it. Leave the default so misuse surfaces loudly rather than silently.

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format OpenAI Responses API request payload from RequestInfo."""
        if not request_info.turns:
            raise ValueError("Responses endpoint requires at least one turn.")

        turns = request_info.turns
        model_endpoint = request_info.model_endpoint

        # Responses API doesn't nest the system prompt into ``input``; it
        # lives in top-level ``instructions``. The per-conversation
        # ``user_context_message`` is prepended as a leading user item.
        input_items: list[dict[str, Any]] = []
        if request_info.user_context_message:
            input_items.append(
                {
                    "role": self.DEFAULT_TURN_ROLE,
                    "content": request_info.user_context_message,
                }
            )
        input_items.extend(self.build_messages(turns))

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
