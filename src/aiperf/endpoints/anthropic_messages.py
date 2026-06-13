# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, ClassVar

import orjson

from aiperf.common.enums import MediaType
from aiperf.common.models import (
    ExtractedPayload,
    InferenceServerResponse,
    ParsedResponse,
    ReasoningResponseData,
    RequestInfo,
    TextResponseData,
    ToolCallResponseData,
)
from aiperf.common.types import JsonObject
from aiperf.endpoints.base_endpoint import BaseEndpoint


class MessagesEndpoint(BaseEndpoint):
    """Anthropic Messages API endpoint (``/v1/messages``).

    The role/content message array reuses the generic
    ``BaseEndpoint.build_messages`` flow - Anthropic messages share the
    ``{"role": ..., "content": ...}`` shape with OpenAI chat. Only the
    image content-part shape differs (Anthropic nests a ``source`` object
    keyed by ``base64``/``url`` rather than OpenAI's ``image_url``), so we
    override that hook and leave the iteration skeleton alone.

    Two contract differences from chat drive the rest of this class:

    - The shared system prompt lives in the top-level ``system`` field,
      never as a ``system``-role message inside ``messages``.
    - ``max_tokens`` is required by the API; absence falls back to
      ``DEFAULT_MAX_TOKENS`` rather than being omitted.

    Audio and video inputs are not supported by the Messages API; those
    content parts raise at format time instead of letting the server 4xx.
    """

    DEFAULT_MAX_TOKENS: ClassVar[int] = 16384
    """Fallback when a turn carries no ``max_tokens``. The Messages API
    rejects requests without ``max_tokens``, so unlike chat we always emit
    a value."""

    # Anthropic content-part type names. ``text`` matches the chat default;
    # ``image`` replaces chat's ``image_url``. Audio/video are unsupported,
    # so their sets are empty and ISL accounting never expects them.
    PART_TYPES: ClassVar[dict[MediaType, set[str]]] = {
        MediaType.TEXT: {"text"},
        MediaType.IMAGE: {"image"},
        MediaType.AUDIO: set(),
        MediaType.VIDEO: set(),
    }

    def extract_payload_inputs(self, payload: dict[str, Any]) -> ExtractedPayload:
        """Messages-API single-pass extraction.

        Inherits the base-class walk (content parts dispatched via
        ``PART_TYPES``) and additionally prepends the top-level ``system``
        prompt - the Messages-API equivalent of a system message that lives
        outside ``messages`` and would otherwise be missed by ISL
        tokenisation. Accepts both the string and list-of-blocks shapes the
        API permits for ``system``.
        """
        result = super().extract_payload_inputs(payload)
        system = payload.get("system")
        if isinstance(system, str):
            result.texts.insert(0, system)
        elif isinstance(system, list):
            collected: list[str] = []
            for block in system:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str) and text:
                        collected.append(text)
                elif isinstance(block, str) and block:
                    collected.append(block)
            for text in reversed(collected):
                result.texts.insert(0, text)
        return result

    # --- Content-part hooks ---------------------------------------------------

    def _render_image_part(self, url_or_data_uri: str) -> dict[str, Any]:
        """Render one image as an Anthropic content block.

        Data URIs (``data:image/png;base64,<b64>``) become a ``base64``
        source with the parsed ``media_type``; everything else is treated
        as a remote ``url`` source.
        """
        if url_or_data_uri.startswith("data:"):
            header, _, b64 = url_or_data_uri.partition(",")
            media_type = header[len("data:") :].split(";", 1)[0] or "image/png"
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64,
                },
            }
        return {"type": "image", "source": {"type": "url", "url": url_or_data_uri}}

    def _render_audio_part(self, format_and_b64: str) -> dict[str, Any]:
        raise NotImplementedError(
            "Anthropic Messages API does not support audio input. "
            "Use endpoint=chat for audio turns, or remove the audio content."
        )

    def _render_video_part(self, url_or_data_uri: str) -> dict[str, Any]:
        raise NotImplementedError(
            "Anthropic Messages API does not support video input. "
            "Use endpoint=chat for video turns, or remove the video content."
        )

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format an Anthropic Messages API request payload from RequestInfo."""
        if not request_info.turns:
            raise ValueError("Messages endpoint requires at least one turn.")

        turns = request_info.turns
        model_endpoint = request_info.model_endpoint

        # The per-conversation user context is prepended as a leading user
        # message; the shared system prompt is top-level, not a message.
        messages: list[dict[str, Any]] = []
        if request_info.user_context_message:
            messages.append(
                {
                    "role": self.DEFAULT_TURN_ROLE,
                    "content": request_info.user_context_message,
                }
            )
        messages.extend(self.build_messages(turns))

        # Conversation-level fields walk turns from the end so FORK-mode
        # children whose final turn lacks model/tools still inherit the
        # parent's intent. Per-request overrides stay scoped to the turn.
        model_name = turns[-1].model
        max_tokens = turns[-1].max_tokens
        extra_body = turns[-1].extra_body
        raw_tools = self._latest_turn_attr(turns, "raw_tools")

        payload: dict[str, Any] = {
            "messages": messages,
            "model": model_name or model_endpoint.primary_model_name,
            "max_tokens": max_tokens
            if max_tokens is not None
            else self.DEFAULT_MAX_TOKENS,
            "stream": model_endpoint.endpoint.streaming,
        }
        if request_info.system_message:
            payload["system"] = request_info.system_message
        if raw_tools is not None:
            payload["tools"] = raw_tools

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)
        if extra_body:
            payload.update(extra_body)

        self.trace(lambda: f"Formatted payload: {payload}")
        return payload

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse an Anthropic Messages API response.

        Handles both the non-streaming full message (``type: message``) and
        the streaming SSE events (``message_start``, ``content_block_delta``,
        ``message_delta``, etc.). Streaming splits usage across two events:
        ``message_start`` carries ``input_tokens`` and ``message_delta``
        carries the final ``output_tokens``; both are surfaced as
        usage-only ``ParsedResponse`` objects.

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with extracted content and/or usage, or None.
        """
        json_obj = response.get_json()
        if not json_obj:
            return None

        if json_obj.get("type") == "message":
            return self._parse_full_response(json_obj, response.perf_ns)

        return self._parse_streaming_event(json_obj, response.perf_ns)

    def _parse_full_response(
        self, json_obj: JsonObject, perf_ns: int
    ) -> ParsedResponse | None:
        """Parse a non-streaming ``type: message`` response object."""
        data = self._extract_content_blocks(json_obj.get("content"))
        usage = json_obj.get("usage") or None

        if data is None and not usage:
            return None

        return ParsedResponse(perf_ns=perf_ns, data=data, usage=usage)

    @staticmethod
    def _extract_content_blocks(
        content: Any,
    ) -> TextResponseData | ReasoningResponseData | ToolCallResponseData | None:
        """Extract model-generated tokens from a ``content[]`` block list.

        ``text`` blocks contribute prose, ``thinking`` blocks contribute
        reasoning, and ``tool_use`` blocks contribute their ``name`` plus
        serialised ``input`` (tokens the model generated and the server's
        ``usage.output_tokens`` already counts). Precedence mirrors the
        chat and Responses endpoints: ``reasoning > text > tool``.
        """
        if not isinstance(content, list):
            return None

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_call_parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                MessagesEndpoint._collect_content_block(
                    block, text_parts, reasoning_parts, tool_call_parts
                )

        if reasoning_parts:
            return ReasoningResponseData(
                content="".join(text_parts) or None,
                reasoning="".join(reasoning_parts),
            )
        if text_parts:
            return TextResponseData(text="".join(text_parts))
        if tool_call_parts:
            return ToolCallResponseData(tool_call_text="".join(tool_call_parts))
        return None

    @staticmethod
    def _collect_content_block(
        block: dict[str, Any],
        text_parts: list[str],
        reasoning_parts: list[str],
        tool_call_parts: list[str],
    ) -> None:
        """Append one ``content[]`` block's tokens to the matching accumulator."""
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                text_parts.append(text)
        elif block_type == "thinking":
            thinking = block.get("thinking")
            if isinstance(thinking, str) and thinking:
                reasoning_parts.append(thinking)
        elif block_type == "tool_use":
            name = block.get("name")
            if isinstance(name, str) and name:
                tool_call_parts.append(name)
            tool_input = block.get("input")
            if tool_input:
                tool_call_parts.append(orjson.dumps(tool_input).decode())

    def _parse_streaming_event(
        self, json_obj: JsonObject, perf_ns: int
    ) -> ParsedResponse | None:
        """Parse a single Anthropic streaming SSE event."""
        event_type = json_obj.get("type")

        if event_type == "content_block_delta":
            data = self._streaming_delta_data(json_obj.get("delta") or {})
            return ParsedResponse(perf_ns=perf_ns, data=data) if data else None

        if event_type == "message_start":
            usage = (json_obj.get("message") or {}).get("usage") or None
            return (
                ParsedResponse(perf_ns=perf_ns, data=None, usage=usage)
                if usage
                else None
            )

        if event_type == "message_delta":
            usage = json_obj.get("usage") or None
            return (
                ParsedResponse(perf_ns=perf_ns, data=None, usage=usage)
                if usage
                else None
            )

        # content_block_start/stop, message_stop, ping: structural envelopes
        # with no replayable token content.
        return None

    @staticmethod
    def _streaming_delta_data(
        delta: dict[str, Any],
    ) -> TextResponseData | ReasoningResponseData | ToolCallResponseData | None:
        """Map a ``content_block_delta`` ``delta`` to its response-data shape."""
        delta_type = delta.get("type")
        if delta_type == "text_delta":
            text = delta.get("text")
            return TextResponseData(text=text) if text else None
        if delta_type == "thinking_delta":
            thinking = delta.get("thinking")
            return ReasoningResponseData(reasoning=thinking) if thinking else None
        if delta_type == "input_json_delta":
            partial = delta.get("partial_json")
            return ToolCallResponseData(tool_call_text=partial) if partial else None
        return None
