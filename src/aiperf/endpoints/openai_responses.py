# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, ClassVar

from aiperf.common.enums import MediaType
from aiperf.common.models import (
    ExtractedPayload,
    InferenceServerResponse,
    ParsedResponse,
    ReasoningResponseData,
    RequestInfo,
    RequestRecord,
    TextResponseData,
    ToolCallResponseData,
    Turn,
)
from aiperf.common.types import JsonObject
from aiperf.endpoints import _openai_responses_replay as _replay
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
    # walks the payload once and dispatches every part against this map -
    # text parts contribute to the tokenisable text list, media parts
    # bump their respective counts.
    PART_TYPES: ClassVar[dict[MediaType, set[str]]] = {
        MediaType.TEXT: {"input_text"},
        MediaType.IMAGE: {"input_image"},
        MediaType.AUDIO: {"input_audio"},
        # Responses API does not currently support video input.
        MediaType.VIDEO: set(),
    }

    def extract_payload_inputs(self, payload: dict[str, Any]) -> ExtractedPayload:
        """Responses-API single-pass extraction.

        Inherits the base-class walk (which dispatches content parts via
        ``PART_TYPES``) and additionally prepends ``instructions`` - the
        Responses-API equivalent of a system prompt that lives at the
        top level of the payload rather than inside ``input``.
        Accepts both string and list-of-content-parts shapes for
        ``instructions`` (some Responses-API variants emit either).
        """
        result = super().extract_payload_inputs(payload)
        instructions = payload.get("instructions")
        if isinstance(instructions, str):
            result.texts.insert(0, instructions)
        elif isinstance(instructions, list):
            collected: list[str] = []
            for part in instructions:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str) and text:
                        collected.append(text)
                elif isinstance(part, str) and part:
                    collected.append(part)
            for text in reversed(collected):
                result.texts.insert(0, text)
        return result

    # --- Content-part hooks (override only the type names) -------------------

    def _render_text_part(self, text: str) -> dict[str, Any]:
        return {"type": "input_text", "text": text}

    def _render_image_part(self, url_or_data_uri: str) -> dict[str, Any]:
        # Responses API takes ``image_url`` as a plain string, not nested.
        return {"type": "input_image", "image_url": url_or_data_uri}

    # _render_audio_part: inherited — Responses API uses the same
    # {"type": "input_audio", "input_audio": {...}} shape as chat.

    def _render_video_part(self, url_or_data_uri: str) -> dict[str, Any]:
        """Reject video parts at format time rather than letting the server 4xx.

        The Responses API does not accept video input as of this writing.
        Inheriting the chat default would emit ``{"type": "video_url", ...}``
        which the server rejects with an opaque schema error - and
        ``PART_TYPES[VIDEO]`` is empty, so ISL accounting silently
        undercounts. Surface the misconfiguration immediately.
        """
        raise NotImplementedError(
            "Responses API does not support video input. "
            "Use endpoint=chat for video turns, or remove the video content."
        )

    # ------------------------------------------------------------------
    # raw_messages filtering on replay
    # ------------------------------------------------------------------
    #
    # ``ResponsesEndpoint.build_assistant_turn`` captures the parent's
    # full ``output[]`` into ``raw_messages``. The Responses API accepts
    # most of those item types in ``input`` on the next request, but a
    # handful are *only* valid as input when paired with the corresponding
    # tool config or with ``previous_response_id`` / ``encrypted_content``:
    #
    # - ``web_search_call`` / ``file_search_call`` need ``tools=[{"type":
    #   "web_search"}]`` / ``file_search`` configured on the next request.
    # - ``image_generation_call`` / ``code_interpreter_call`` /
    #   ``computer_call`` likewise require their tool config.
    # - ``reasoning`` items need ``store=False`` + ``encrypted_content``,
    #   or the conversation continued via ``previous_response_id``.
    #
    # When a FORK-mode DAG child inherits the parent's history without
    # those tools/flags configured (the common case), splicing these
    # items back into ``input`` 400s with an opaque schema error.
    #
    # Skip them. ``message`` items (the actual assistant text) and
    # ``function_call`` items (user-defined tools, valid as input
    # alongside their ``function_call_output``) round-trip cleanly.

    _REPLAY_UNSAFE_OUTPUT_ITEM_TYPES: ClassVar[frozenset[str]] = frozenset(
        {
            "web_search_call",
            "file_search_call",
            "image_generation_call",
            "code_interpreter_call",
            "computer_call",
            "reasoning",
        }
    )

    def build_messages(self, turns: list[Turn]) -> list[dict[str, Any]]:
        """Filter Responses-API replay-unsafe items out of raw_messages.

        Same flatten-and-merge skeleton as ``BaseEndpoint.build_messages``
        but drops output-only item types from each turn's
        ``raw_messages`` before splicing. See
        ``_REPLAY_UNSAFE_OUTPUT_ITEM_TYPES``.
        """
        messages: list[dict[str, Any]] = []
        for turn in turns:
            if turn.raw_messages:
                for item in turn.raw_messages:
                    if (
                        isinstance(item, dict)
                        and item.get("type") in self._REPLAY_UNSAFE_OUTPUT_ITEM_TYPES
                    ):
                        continue
                    messages.append(item)
                continue
            message = self._render_turn_message(turn)
            message["type"] = "message"
            messages.append(message)
        return messages

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
                    "type": "message",
                    "role": self.DEFAULT_TURN_ROLE,
                    "content": request_info.user_context_message,
                }
            )
        input_items.extend(self.build_messages(turns))

        # Conversation-level fields walk turns from the end so FORK-mode
        # children whose final turn lacks model/tools still inherit the parent's
        # intent. Per-request overrides stay scoped to the dispatching turn.
        model_name = turns[-1].model
        max_tokens = turns[-1].max_tokens
        extra_body = turns[-1].extra_body

        payload: dict[str, Any] = {
            "input": input_items,
            "model": model_name or model_endpoint.primary_model_name,
            "stream": model_endpoint.endpoint.streaming,
        }
        for key, value in (
            ("instructions", request_info.system_message or None),
            ("max_output_tokens", max_tokens),
            ("tools", self._latest_turn_attr(turns, "raw_tools")),
        ):
            if value is not None:
                payload[key] = value

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)
        if extra_body:
            payload.update(extra_body)

        self._maybe_enable_usage_stream_options(payload, model_endpoint)

        self.trace(lambda: f"Formatted payload: {payload}")
        return payload

    @staticmethod
    def _maybe_enable_usage_stream_options(
        payload: dict[str, Any], model_endpoint: Any
    ) -> None:
        """Set ``stream_options.include_usage=True`` for streaming runs with
        server-side token counts, preserving any caller-supplied mapping."""
        ep = model_endpoint.endpoint
        if not (ep.streaming and ep.use_server_token_count):
            return
        so = payload.get("stream_options")
        if not isinstance(so, dict):
            payload["stream_options"] = {"include_usage": True}
        elif "include_usage" not in so:
            so["include_usage"] = True

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

        Surfaces ``response.function_call_arguments.delta`` as a
        ``ToolCallResponseData`` - without this, ~64% of streaming turns
        in real agentic traffic have NO data-bearing event, so the
        worker's first-token callback never fires and client-side OSL is
        undercounted by every tool-using turn. The arguments JSON is what
        the model generated on the wire, so the existing tokeniser treats
        it like any other generated text.

        ``response.function_call_arguments.done`` carries no replayable
        delta content (the assembled arguments are already captured by
        the chained deltas plus the final ``response.completed`` /
        ``response.output_item.done`` events that ``build_assistant_turn``
        consumes), so we let it fall through to the structural-envelope
        branch below.

        Args:
            json_obj: Deserialized event JSON
            perf_ns: Performance timestamp

        Returns:
            Parsed response or None if the event carries no content
        """
        event_type = json_obj.get("type")

        data = self._streaming_event_data(event_type, json_obj)
        if data is not None:
            return ParsedResponse(perf_ns=perf_ns, data=data)

        if event_type == "response.completed":
            resp = json_obj.get("response") or {}
            usage = resp.get("usage") or None
            if usage:
                return ParsedResponse(perf_ns=perf_ns, data=None, usage=usage)
            return None

        # All other events (response.created, response.in_progress,
        # response.output_item.added/done, content_part.added/done,
        # response.function_call_arguments.done, etc.) carry no replayable
        # token content - they're structural envelopes.
        return None

    @staticmethod
    def _streaming_event_data(
        event_type: Any, json_obj: JsonObject
    ) -> TextResponseData | ReasoningResponseData | ToolCallResponseData | None:
        """Map a content-bearing SSE event to its response-data shape.

        Returns ``None`` for events that carry no payload-relevant content
        (including the missing-delta variants of the data events) so the
        caller can fall through to non-data branches like
        ``response.completed``.
        """
        if event_type == "response.output_text.delta":
            delta = json_obj.get("delta")
            return TextResponseData(text=delta) if delta else None

        if event_type == "response.reasoning_text.delta":
            delta = json_obj.get("delta")
            return ReasoningResponseData(reasoning=delta) if delta else None

        if event_type == "response.output_text.done":
            text = json_obj.get("text")
            return TextResponseData(text=text) if text else None

        if event_type == "response.function_call_arguments.delta":
            delta = json_obj.get("delta")
            return ToolCallResponseData(tool_call_text=delta) if delta else None

        return None

    def _parse_full_response(
        self, json_obj: JsonObject, perf_ns: int
    ) -> ParsedResponse | None:
        """Parse a non-streaming full response object.

        Args:
            json_obj: Deserialized response JSON with "object": "response"
            perf_ns: Performance timestamp

        Returns:
            Parsed response with extracted content and usage
        """
        data = self._extract_response_content(json_obj)
        usage = json_obj.get("usage") or None

        if data is None and not usage:
            return None

        return ParsedResponse(perf_ns=perf_ns, data=data, usage=usage)

    def _extract_response_content(
        self, json_obj: JsonObject
    ) -> TextResponseData | ReasoningResponseData | ToolCallResponseData | None:
        """Extract content from a non-streaming Responses API response.

        Walks ``output[]`` for every item type that carries model-generated
        tokens:

        - ``message`` items contribute their ``output_text`` parts.
        - ``reasoning`` items contribute their ``summary_text`` parts.
        - ``function_call`` items contribute ``name`` + ``arguments`` -
          the model generated those tokens, and the server's
          ``usage.completion_tokens`` already counts them, so client-side
          OSL must too.

        Precedence mirrors ``ChatEndpoint.extract_chat_response_data``
        (PR #804): ``reasoning > message > function_call``. The first
        non-empty source wins; the others are dropped from this single
        ``ParsedResponse``. The full structured ``output[]`` is still
        captured by ``build_assistant_turn`` for fork-mode replay.

        Falls back to the top-level ``output_text`` convenience field when
        ``output[]`` is absent.

        Args:
            json_obj: Deserialized response JSON

        Returns:
            Extracted response data or None if no content found
        """
        output = json_obj.get("output")
        if isinstance(output, list):
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_call_parts: list[str] = []
            for item in output:
                if isinstance(item, dict):
                    self._collect_output_item(
                        item, text_parts, reasoning_parts, tool_call_parts
                    )

            if reasoning_parts:
                return ReasoningResponseData(
                    content="".join(text_parts) or None,
                    reasoning="".join(reasoning_parts),
                )
            if text_parts:
                return TextResponseData(text="".join(text_parts))
            if tool_call_parts:
                return ToolCallResponseData(
                    tool_call_text="".join(tool_call_parts),
                    content=None,
                )

        # Fallback: top-level output_text convenience field
        output_text = json_obj.get("output_text")
        if isinstance(output_text, str) and output_text:
            return TextResponseData(text=output_text)

        return None

    @staticmethod
    def _collect_output_item(
        item: dict[str, Any],
        text_parts: list[str],
        reasoning_parts: list[str],
        tool_call_parts: list[str],
    ) -> None:
        """Append model-generated tokens from one ``output[]`` item to the
        appropriate accumulator. Unknown item types are silently ignored.
        """
        item_type = item.get("type")
        if item_type == "reasoning":
            ResponsesEndpoint._collect_reasoning_summary(item, reasoning_parts)
        elif item_type == "message":
            ResponsesEndpoint._collect_message_content(item, text_parts)
        elif item_type == "function_call":
            ResponsesEndpoint._collect_function_call(item, tool_call_parts)

    @staticmethod
    def _collect_reasoning_summary(
        item: dict[str, Any], reasoning_parts: list[str]
    ) -> None:
        """Append non-empty ``summary_text`` strings from a reasoning item."""
        summary = item.get("summary")
        if not isinstance(summary, list):
            return
        for part in summary:
            if isinstance(part, dict) and part.get("type") == "summary_text":
                text = part.get("text")
                if text:
                    reasoning_parts.append(text)

    @staticmethod
    def _collect_message_content(item: dict[str, Any], text_parts: list[str]) -> None:
        """Append non-empty ``output_text`` strings from a message item."""
        content = item.get("content")
        if not isinstance(content, list):
            return
        for part in content:
            if isinstance(part, dict) and part.get("type") == "output_text":
                text = part.get("text")
                if text:
                    text_parts.append(text)

    @staticmethod
    def _collect_function_call(
        item: dict[str, Any], tool_call_parts: list[str]
    ) -> None:
        """Append ``name`` and ``arguments`` strings from a function_call item."""
        name = item.get("name")
        if isinstance(name, str) and name:
            tool_call_parts.append(name)
        arguments = item.get("arguments")
        if isinstance(arguments, str) and arguments:
            tool_call_parts.append(arguments)

    def build_assistant_turn(self, record: RequestRecord) -> Turn | None:
        """Capture every output item — message, function_call,
        web_search_call, image_generation_call, reasoning, etc. — for
        FORK-replay. See ``_openai_responses_replay`` for the assembly
        logic and the rationale for the dedup-by-id union of
        ``response.completed.response.output[]`` and
        ``response.output_item.done`` events.

        Falls back to the base text-only behaviour when no items are
        recoverable, so callers without tool-using workloads see no change.
        """
        items_by_key: dict[str, dict[str, Any]] = {}
        done_items: list[dict[str, Any]] = []

        for response in record.responses:
            json_obj = response.get_json()
            if not json_obj:
                continue
            if _replay.is_failure_event(json_obj):
                return super().build_assistant_turn(record)
            _replay.collect_response_items(json_obj, items_by_key, done_items)

        for item in done_items:
            _replay.merge_item(items_by_key, item)

        if not items_by_key:
            return super().build_assistant_turn(record)

        return Turn(role="assistant", raw_messages=list(items_by_key.values()))
