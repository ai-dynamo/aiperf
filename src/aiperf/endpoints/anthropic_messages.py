# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from typing import Any, ClassVar

import orjson

from aiperf.common.enums import CaseInsensitiveStrEnum, MediaType
from aiperf.common.models import (
    BaseResponseData,
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
from aiperf.endpoints.base_endpoint import BaseEndpoint

_ANTHROPIC_VERSION: str = "2023-06-01"


class ContentBlockType(CaseInsensitiveStrEnum):
    """Content block types in Anthropic Messages API requests and responses.

    ``tool_result`` appears only in requests (user-role history blocks);
    the rest appear in assistant responses.
    """

    TEXT = "text"
    THINKING = "thinking"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"


class DeltaType(CaseInsensitiveStrEnum):
    """Delta types within content_block_delta SSE events."""

    TEXT_DELTA = "text_delta"
    THINKING_DELTA = "thinking_delta"
    INPUT_JSON_DELTA = "input_json_delta"
    SIGNATURE_DELTA = "signature_delta"


class EventType(CaseInsensitiveStrEnum):
    """Payload type values in Anthropic Messages API responses."""

    MESSAGE = "message"
    MESSAGE_START = "message_start"
    CONTENT_BLOCK_START = "content_block_start"
    CONTENT_BLOCK_DELTA = "content_block_delta"
    CONTENT_BLOCK_STOP = "content_block_stop"
    MESSAGE_DELTA = "message_delta"
    MESSAGE_STOP = "message_stop"
    PING = "ping"
    ERROR = "error"


# --- Payload-input walk helpers (Anthropic shapes the base walk misses) ------


def _walk_system(payload: dict[str, Any], result: ExtractedPayload) -> None:
    """Prepend the top-level ``system`` field to ``result.texts``.

    Accepts both string and list-of-content-parts shapes (the Anthropic
    spec permits either). List form items must be ``{"type":"text","text":...}``;
    other types are skipped (the spec reserves them for future use).
    """
    system = payload.get("system")
    if isinstance(system, str):
        if system:
            result.texts.insert(0, system)
        return
    if not isinstance(system, list):
        return
    collected: list[str] = []
    for part in system:
        if isinstance(part, dict) and part.get("type") == ContentBlockType.TEXT:
            text = part.get("text")
            if isinstance(text, str) and text:
                collected.append(text)
        elif isinstance(part, str) and part:
            collected.append(part)
    for text in reversed(collected):
        result.texts.insert(0, text)


def _walk_tool_schemas(payload: dict[str, Any], result: ExtractedPayload) -> None:
    """Collect ``input_schema`` text from top-level Anthropic tools.

    The base ``_walk_tools_schema`` (called by the inherited
    ``extract_payload_inputs``) already harvests ``name`` and
    ``description`` fields from each tool dict. Anthropic's tool schema
    field is named ``input_schema`` (not OpenAI's ``parameters``), so we
    serialise it ourselves here. Without this the tokeniser undercounts
    every agentic request that declares tools.
    """
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        input_schema = tool.get("input_schema")
        if isinstance(input_schema, dict):
            with contextlib.suppress(TypeError):
                result.texts.append(orjson.dumps(input_schema).decode())


def _walk_tool_blocks(payload: dict[str, Any], result: ExtractedPayload) -> None:
    """Collect tokenisable text from ``tool_use`` and ``tool_result`` content blocks.

    The base content-part walk dispatches via ``PART_TYPES`` and only
    knows about media (text/image/audio/video). Anthropic's agentic
    history replay also includes:

    - ``{"type":"tool_use","id":...,"name":...,"input":{...}}`` -
      assistant blocks. Server tokenises ``name`` and the serialised
      ``input`` JSON.
    - ``{"type":"tool_result","tool_use_id":...,"content":...}`` -
      user-role blocks containing the tool output the model previously
      saw. ``content`` is either a string or a list of
      ``{"type":"text","text":...}`` blocks.

    Without this walk, agent-history replays silently undercount ISL by
    everything inside these blocks.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == ContentBlockType.TOOL_USE:
                _collect_tool_use(part, result)
            elif part_type == ContentBlockType.TOOL_RESULT:
                _collect_tool_result(part, result)


def _collect_tool_use(part: dict[str, Any], result: ExtractedPayload) -> None:
    """Collect ``name`` and serialised ``input`` from one tool_use block."""
    name = part.get("name")
    if isinstance(name, str) and name:
        result.texts.append(name)
    input_value = part.get("input")
    if isinstance(input_value, dict):
        with contextlib.suppress(TypeError):
            result.texts.append(orjson.dumps(input_value).decode())


def _collect_tool_result(part: dict[str, Any], result: ExtractedPayload) -> None:
    """Collect text from one tool_result block.

    ``content`` is either a string (legacy shorthand) or a list of
    ``{"type":"text","text":...}`` blocks. Other block types (image,
    etc.) are skipped here - image content already counts via the base
    walk's image branch when it encounters the part's ``type``.
    """
    content = part.get("content")
    if isinstance(content, str):
        if content:
            result.texts.append(content)
        return
    if not isinstance(content, list):
        return
    for sub in content:
        if not isinstance(sub, dict):
            continue
        if sub.get("type") == ContentBlockType.TEXT:
            text = sub.get("text")
            if isinstance(text, str) and text:
                result.texts.append(text)


# --- Assistant-replay accumulators (streaming + non-streaming responses) -----


def _absorb_event(
    json_obj: JsonObject,
    text_parts: list[str],
    thinking_blocks_by_index: dict[int, dict[str, Any]],
    tool_uses_by_index: dict[int, dict[str, Any]],
) -> None:
    """Fold one Anthropic response payload (streaming SSE or non-streaming
    ``message``) into the running assistant accumulators.

    Non-streaming ``type=message`` responses already carry the full
    ``content`` array; streaming responses arrive as a sequence of
    ``content_block_start`` (with the empty block envelope, including
    ``index``) and ``content_block_delta`` (with ``text_delta``,
    ``thinking_delta``, ``signature_delta``, or ``input_json_delta``
    fragments) events that must be reassembled.

    Thinking blocks carry an opaque ``signature`` Anthropic emits
    alongside the text; both must round-trip together for the server to
    accept the block on FORK-mode replay.
    """
    event_type = json_obj.get("type")
    if event_type == EventType.MESSAGE:
        _absorb_message(
            json_obj, text_parts, thinking_blocks_by_index, tool_uses_by_index
        )
    elif event_type == EventType.CONTENT_BLOCK_START:
        _absorb_content_block_start(
            json_obj, thinking_blocks_by_index, tool_uses_by_index
        )
    elif event_type == EventType.CONTENT_BLOCK_DELTA:
        _absorb_content_block_delta(
            json_obj, text_parts, thinking_blocks_by_index, tool_uses_by_index
        )


def _absorb_message(
    json_obj: JsonObject,
    text_parts: list[str],
    thinking_blocks_by_index: dict[int, dict[str, Any]],
    tool_uses_by_index: dict[int, dict[str, Any]],
) -> None:
    """Non-streaming ``type=message``: walk the full ``content`` array."""
    for block in json_obj.get("content") or []:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == ContentBlockType.TEXT:
            text = block.get("text")
            if isinstance(text, str):
                text_parts.append(text)
        elif block_type == ContentBlockType.THINKING:
            idx = len(thinking_blocks_by_index)
            # Preserve the full block (thinking, signature, anything else
            # the server emits) so _finalise_thinking round-trips it.
            thinking_blocks_by_index[idx] = {
                k: v for k, v in block.items() if k != "type"
            }
        elif block_type == ContentBlockType.TOOL_USE:
            idx = len(tool_uses_by_index)
            # Preserve every field the server emits (id, name, input,
            # ``caller`` for Claude Code agentic dispatch, future fields).
            tool_uses_by_index[idx] = {k: v for k, v in block.items() if k != "type"}


def _absorb_content_block_start(
    json_obj: JsonObject,
    thinking_blocks_by_index: dict[int, dict[str, Any]],
    tool_uses_by_index: dict[int, dict[str, Any]],
) -> None:
    """Streaming ``content_block_start``: open thinking / tool_use accumulators.

    Thinking blocks open with empty ``thinking`` and ``signature`` strings;
    tool_use blocks open with empty ``input`` (filled via input_json_delta).
    Other block types (``text``) need no per-index slot - text deltas
    accumulate into the flat ``text_parts`` list.
    """
    block = json_obj.get("content_block") or {}
    block_type = block.get("type")
    idx = json_obj.get("index")
    if block_type == ContentBlockType.THINKING:
        slot = idx if idx is not None else len(thinking_blocks_by_index)
        thinking_blocks_by_index[slot] = {k: v for k, v in block.items() if k != "type"}
        # Ensure the accumulator fields exist so deltas can append safely.
        thinking_blocks_by_index[slot].setdefault("thinking", "")
        thinking_blocks_by_index[slot].setdefault("signature", "")
        return
    if block_type != ContentBlockType.TOOL_USE:
        return
    slot = idx if idx is not None else len(tool_uses_by_index)
    accumulator = {k: v for k, v in block.items() if k != "type"}
    # Input streams in as JSON fragments via input_json_delta; accumulate
    # the raw string here, parse once at finalise.
    accumulator["_input_json"] = ""
    tool_uses_by_index[slot] = accumulator


def _absorb_content_block_delta(
    json_obj: JsonObject,
    text_parts: list[str],
    thinking_blocks_by_index: dict[int, dict[str, Any]],
    tool_uses_by_index: dict[int, dict[str, Any]],
) -> None:
    """Streaming ``content_block_delta``: dispatch by ``delta.type``.

    ``text_delta`` -> text_parts; ``thinking_delta`` and ``signature_delta``
    -> thinking accumulator at this index; ``input_json_delta`` -> tool_use
    accumulator at this index. Unknown delta types are dropped.
    """
    delta = json_obj.get("delta") or {}
    delta_type = delta.get("type")
    if delta_type == DeltaType.TEXT_DELTA:
        text = delta.get("text")
        if isinstance(text, str):
            text_parts.append(text)
        return

    idx = json_obj.get("index")
    if idx is None:
        return

    if delta_type == DeltaType.THINKING_DELTA:
        _append_to_indexed(
            thinking_blocks_by_index, idx, "thinking", delta.get("thinking")
        )
    elif delta_type == DeltaType.SIGNATURE_DELTA:
        _append_to_indexed(
            thinking_blocks_by_index, idx, "signature", delta.get("signature")
        )
    elif delta_type == DeltaType.INPUT_JSON_DELTA:
        _append_to_indexed(
            tool_uses_by_index, idx, "_input_json", delta.get("partial_json") or ""
        )


def _append_to_indexed(
    blocks_by_index: dict[int, dict[str, Any]],
    idx: int,
    field: str,
    fragment: Any,
) -> None:
    """Append a string ``fragment`` to ``blocks_by_index[idx][field]``.

    No-op when the index has no open accumulator (deltas can arrive
    before content_block_start in pathological streams) or the fragment
    is not a string (defensive: server contract is string deltas).
    """
    accum = blocks_by_index.get(idx)
    if accum is None or not isinstance(fragment, str):
        return
    accum[field] = accum.get(field, "") + fragment


def _finalise_thinking(accumulator: dict[str, Any]) -> dict[str, Any]:
    """Convert a streaming/non-streaming thinking accumulator into a wire block.

    Anthropic requires ``signature`` to round-trip thinking blocks on
    FORK-mode replay; both fields are preserved verbatim. Empty
    ``thinking`` / ``signature`` are kept (server emits them as empty
    strings on open) rather than stripped, so the resulting block matches
    the wire shape of the original.
    """
    block = {"type": "thinking"}
    block.update({k: v for k, v in accumulator.items() if v is not None})
    return block


def _finalise_tool_use(accumulator: dict[str, Any]) -> dict[str, Any]:
    """Convert a streaming/non-streaming tool_use accumulator into a wire block.

    Streaming accumulators carry ``_input_json`` (raw concatenated
    fragments from ``input_json_delta`` chunks); we parse once at the
    end and drop the raw string so the resulting block round-trips
    through ``build_messages`` unchanged. Malformed JSON is preserved as
    a string under ``input`` so the request still serialises (the server
    will reject it loudly rather than us silently dropping data).
    """
    if "_input_json" in accumulator:
        raw = accumulator.pop("_input_json")
        if raw:
            try:
                accumulator["input"] = orjson.loads(raw)
            except orjson.JSONDecodeError:
                accumulator["input"] = raw
        else:
            accumulator.setdefault("input", {})
    block = {"type": "tool_use"}
    block.update({k: v for k, v in accumulator.items() if v is not None})
    return block


def _accumulate_content_block(
    block: dict[str, Any],
    text_parts: list[str],
    thinking_parts: list[str],
    tool_call_parts: list[str],
) -> None:
    """Sort one non-streaming content block into the per-kind accumulators.

    ``tool_use`` blocks contribute their ``name`` plus serialised ``input`` —
    tokens the model generated that ``usage.output_tokens`` counts.
    """
    block_type = block.get("type")
    if block_type == ContentBlockType.TEXT:
        text_val = block.get("text")
        if text_val:
            text_parts.append(text_val)
    elif block_type == ContentBlockType.THINKING:
        thinking_val = block.get("thinking")
        if thinking_val:
            thinking_parts.append(thinking_val)
    elif block_type == ContentBlockType.TOOL_USE:
        name = block.get("name")
        if isinstance(name, str) and name:
            tool_call_parts.append(name)
        tool_input = block.get("input")
        if tool_input:
            tool_call_parts.append(orjson.dumps(tool_input).decode())


class MessagesEndpoint(BaseEndpoint):
    """Anthropic Messages endpoint.

    Supports text content, tool use, extended thinking, and both
    streaming and non-streaming responses via /v1/messages.

    Message-array construction reuses the generic
    ``BaseEndpoint.build_messages`` flow. Anthropic's wire shape differs
    from OpenAI chat in two places:

    - the system prompt lives at the top level of the payload (not in
      ``messages`` as a ``role: system`` entry);
    - the image content part is ``{"type": "image", "source": {...}}``
      rather than ``{"type": "image_url", ...}``.

    Audio and video content blocks are not part of the Anthropic Messages
    API; ``_render_audio_part`` / ``_render_video_part`` raise immediately
    so misuse fails at format-time rather than producing an opaque server
    4xx.
    """

    # Anthropic content-part type names: ``text`` (same as default) and
    # bare ``image`` (Anthropic uses ``{"type": "image", "source": {...}}``,
    # not OpenAI's ``image_url``). Audio/video are unsupported - empty
    # sets prevent ``extract_payload_inputs`` from miscounting parts that
    # happen to share OpenAI's type names.
    PART_TYPES: ClassVar[dict[MediaType, set[str]]] = {
        MediaType.TEXT: {"text"},
        MediaType.IMAGE: {"image"},
        MediaType.AUDIO: set(),
        MediaType.VIDEO: set(),
    }

    def get_endpoint_headers(self, request_info: RequestInfo) -> dict[str, str]:
        """Get Anthropic-specific headers using x-api-key auth."""
        cfg = self.model_endpoint.endpoint
        headers: dict[str, str] = {"content-type": "application/json"}
        if cfg.headers:
            headers.update(cfg.headers)
        if cfg.api_key:
            headers["x-api-key"] = cfg.api_key
        headers.setdefault("anthropic-version", _ANTHROPIC_VERSION)
        return headers

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        """Format Anthropic Messages API request payload.

        Args:
            request_info: Request context including model endpoint, metadata, and turns

        Returns:
            Anthropic Messages API payload
        """
        if not request_info.turns:
            raise ValueError("Anthropic Messages endpoint requires at least one turn.")

        turns = request_info.turns
        model_endpoint = request_info.model_endpoint

        messages: list[dict[str, Any]] = []
        if request_info.user_context_message:
            messages.append(
                {"role": "user", "content": request_info.user_context_message}
            )
        messages.extend(self.build_messages(turns))

        # Conversation-level fields (raw_tools, raw_system) walk turns from
        # the end so FORK-mode children whose final turn does not redeclare
        # them still inherit the parent's intent. Per-request overrides
        # (model, max_tokens, extra_body) stay scoped to the dispatching turn,
        # matching the repo-wide per-turn payload contract.
        raw_tools = self._latest_turn_attr(turns, "raw_tools")
        raw_system = self._latest_turn_attr(turns, "raw_system")
        max_tokens = turns[-1].max_tokens
        extra_body = turns[-1].extra_body
        model_name = turns[-1].model

        payload: dict[str, Any] = {
            "model": model_name or model_endpoint.primary_model_name,
            "messages": messages,
            # Anthropic requires max_tokens; default mirrors the API's
            # historical minimum-friendly value when no per-turn cap is set.
            "max_tokens": max_tokens if max_tokens is not None else 1024,
        }
        # Real Claude Code clients omit ``stream`` entirely on non-streaming
        # requests; only set it when streaming is enabled to match that wire shape.
        if model_endpoint.endpoint.streaming:
            payload["stream"] = True

        # raw_system (Turn-level list-of-blocks) wins over the
        # conversation-level system_message string. Lets callers attach
        # cache_control / Anthropic-specific extensions per-block.
        if raw_system is not None:
            payload["system"] = raw_system
        elif request_info.system_message:
            payload["system"] = request_info.system_message

        if raw_tools is not None:
            payload["tools"] = raw_tools

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        if extra_body:
            payload.update(extra_body)

        self.trace(lambda: f"Formatted payload: {payload}")
        return payload

    # --- Content-part hooks (override only the Anthropic-specific shapes) ----

    def _render_image_part(self, url_or_data_uri: str) -> dict[str, Any]:
        """Render one image as an Anthropic content block.

        Data URIs (``data:image/png;base64,<b64>``) become a ``base64``
        source with the parsed ``media_type`` — the shape aiperf's synthetic
        image generator emits; everything else is treated as a remote
        ``url`` source.
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
        return {
            "type": "image",
            "source": {"type": "url", "url": url_or_data_uri},
        }

    def _render_audio_part(self, format_and_b64: str) -> dict[str, Any]:
        """Anthropic Messages API does not accept audio content blocks.

        Raise immediately so misuse fails at ``format_payload`` rather than
        producing an opaque server 4xx after the request is dispatched.
        """
        raise NotImplementedError(
            "Anthropic Messages API does not support audio input. "
            "Use a different endpoint, or remove audio content from the turn."
        )

    def _render_video_part(self, url_or_data_uri: str) -> dict[str, Any]:
        """Anthropic Messages API does not accept video content blocks.

        Raise immediately so misuse fails at ``format_payload`` rather than
        producing an opaque server 4xx after the request is dispatched.
        """
        raise NotImplementedError(
            "Anthropic Messages API does not support video input. "
            "Use a different endpoint, or remove video content from the turn."
        )

    # --- Payload -> inputs extraction ----------------------------------------

    def extract_payload_inputs(self, payload: dict[str, Any]) -> ExtractedPayload:
        """Anthropic single-pass extraction.

        Inherits the base-class walk for ``messages`` (which dispatches
        content parts via ``PART_TYPES``) and additionally:

        - prepends top-level ``system`` (string OR list of
          ``{"type":"text","text":...}`` blocks);
        - collects ``input_schema`` for top-level ``tools``
          (Anthropic's equivalent of OpenAI's ``parameters``); ``name``
          and ``description`` are already harvested by the base walk;
        - collects ``name``/``input`` from ``tool_use`` content blocks
          and the ``content`` text of ``tool_result`` blocks - parts the
          server tokenises on agentic-history replay that the base walk
          would otherwise drop because they are not in ``PART_TYPES``.
        """
        result = super().extract_payload_inputs(payload)
        _walk_system(payload, result)
        _walk_tool_schemas(payload, result)
        _walk_tool_blocks(payload, result)
        return result

    def extract_response_data(self, record: RequestRecord) -> list[ParsedResponse]:
        """Extract parsed responses, merging Anthropic's split streaming usage.

        The Messages API splits streaming usage across two events:
        ``message_start`` carries ``input_tokens`` (and cache counts) while
        ``message_delta`` carries the final ``output_tokens`` — per the
        documented contract, ``message_delta`` may omit the input fields
        entirely. The record layer merges streaming usage with
        last-non-empty-chunk-wins semantics (``find_last_non_empty_usage``),
        which would drop ``input_tokens`` for servers that follow the
        documented shape. Fold keys from earlier usage-bearing responses
        into the final one (existing keys win, so cumulative servers that
        already repeat the input fields are untouched) so the record-level
        merge always sees a complete usage dict.
        """
        parsed = super().extract_response_data(record)
        usages = [p.usage for p in parsed if p.usage]
        if len(usages) >= 2:
            final = usages[-1]
            for earlier in usages[:-1]:
                for key, value in earlier.items():
                    final.setdefault(key, value)
        return parsed

    def build_assistant_turn(self, record: RequestRecord) -> Turn | None:
        """Capture text + thinking + ``tool_use`` blocks from an Anthropic
        response for replay.

        Walks the raw responses on ``record``, accumulating text deltas,
        reassembling streaming ``thinking`` blocks (``thinking_delta`` +
        ``signature_delta`` per index) and ``tool_use`` blocks
        (``input_json_delta`` fragments per index), then returns a Turn
        whose ``raw_messages`` re-renders as an assistant message carrying
        the full content array - thinking, then text, then tool_use - so a
        FORK-mode DAG child inheriting the parent's history sees the
        parent's complete reply, not just the text.

        Falls back to the base text-only behaviour when neither thinking
        nor tool_use blocks are present, so callers that don't use either
        feature see no behavioural change.
        """
        text_parts: list[str] = []
        thinking_blocks_by_index: dict[int, dict[str, Any]] = {}
        tool_uses_by_index: dict[int, dict[str, Any]] = {}

        for response in record.responses:
            json_obj = response.get_json()
            if not json_obj:
                continue
            _absorb_event(
                json_obj,
                text_parts,
                thinking_blocks_by_index,
                tool_uses_by_index,
            )

        if not thinking_blocks_by_index and not tool_uses_by_index:
            return super().build_assistant_turn(record)

        content_blocks: list[dict[str, Any]] = []
        # Anthropic emits thinking before text/tool_use; preserve that order
        # so the wire shape matches a fresh assistant reply.
        for idx in sorted(thinking_blocks_by_index):
            content_blocks.append(_finalise_thinking(thinking_blocks_by_index[idx]))
        text = "".join(text_parts)
        if text:
            content_blocks.append({"type": "text", "text": text})
        for idx in sorted(tool_uses_by_index):
            content_blocks.append(_finalise_tool_use(tool_uses_by_index[idx]))

        assistant_msg = {"role": "assistant", "content": content_blocks}
        return Turn(role="assistant", raw_messages=[assistant_msg])

    def _render_text_part(self, text: str) -> dict[str, Any]:
        """Anthropic text part shape: ``{"type": "text", "text": ...}``.

        Identical to the chat default; named explicitly here so the file
        documents the full Anthropic content-part shape contract in one place.
        """
        return {"type": "text", "text": text}

    def parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse Anthropic Messages response.

        Handles both streaming SSE events and non-streaming JSON responses.
        Uses the ``type`` field present in all Anthropic payloads to dispatch:
        ``"message"`` for non-streaming, streaming event types otherwise.

        Args:
            response: Raw response from inference server

        Returns:
            Parsed response with extracted text/reasoning content and usage data
        """
        json_obj = response.get_json()
        if not json_obj:
            return None

        event_type = json_obj.get("type")
        if event_type == EventType.MESSAGE:
            return self._parse_non_streaming(response, json_obj)
        if event_type is not None:
            return self._parse_streaming_event(response, json_obj, event_type)
        return None

    def _parse_non_streaming(
        self, response: InferenceServerResponse, json_obj: JsonObject
    ) -> ParsedResponse | None:
        """Parse non-streaming Anthropic Messages response."""
        data = self._extract_content_data(json_obj)
        usage = json_obj.get("usage")

        if data or usage:
            return ParsedResponse(perf_ns=response.perf_ns, data=data, usage=usage)
        return None

    def _extract_content_data(self, json_obj: JsonObject) -> BaseResponseData | None:
        """Extract content from Anthropic non-streaming response content array.

        ``tool_use`` blocks contribute their ``name`` plus serialised ``input``
        — tokens the model generated that ``usage.output_tokens`` counts.
        Precedence matches the chat endpoint: ``reasoning > text+tool > tool
        > text``; a response that carries both prose and a tool dispatch (the
        standard agentic shape) returns a ``ToolCallResponseData`` with both
        fields set so client-side OSL counts both portions.
        """
        content_blocks = json_obj.get("content")
        if not content_blocks or not isinstance(content_blocks, list):
            return None

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_call_parts: list[str] = []

        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            _accumulate_content_block(
                block, text_parts, thinking_parts, tool_call_parts
            )

        text = "".join(text_parts) or None
        thinking = "".join(thinking_parts) or None
        tool_call_text = "".join(tool_call_parts)

        if thinking:
            return ReasoningResponseData(content=text, reasoning=thinking)
        if tool_call_text:
            return ToolCallResponseData(tool_call_text=tool_call_text, content=text)
        return self.make_text_response_data(text)

    def _parse_streaming_event(
        self,
        response: InferenceServerResponse,
        json_obj: JsonObject,
        event_type: str,
    ) -> ParsedResponse | None:
        """Parse a streaming SSE event from the Anthropic Messages API."""
        match event_type:
            case EventType.MESSAGE_START:
                message = json_obj.get("message", {})
                usage = message.get("usage")
                if usage:
                    return ParsedResponse(perf_ns=response.perf_ns, usage=usage)
                return None

            case EventType.CONTENT_BLOCK_DELTA:
                return self._parse_content_block_delta(response, json_obj)

            case EventType.MESSAGE_DELTA:
                usage = json_obj.get("usage")
                if usage:
                    return ParsedResponse(perf_ns=response.perf_ns, usage=usage)
                return None

            case (
                EventType.PING
                | EventType.CONTENT_BLOCK_START
                | EventType.CONTENT_BLOCK_STOP
                | EventType.MESSAGE_STOP
            ):
                return None

            case EventType.ERROR:
                error_detail = json_obj.get("error", {})
                self.warning(
                    lambda: (
                        f"Anthropic streaming error: "
                        f"type={error_detail.get('type')}, "
                        f"message={error_detail.get('message')}"
                    )
                )
                return None

            case _:
                self.debug(lambda: f"Unknown Anthropic SSE event type: {event_type!r}")
                return None

    def _parse_content_block_delta(
        self, response: InferenceServerResponse, json_obj: JsonObject
    ) -> ParsedResponse | None:
        """Parse a ``content_block_delta`` SSE event.

        Split out of ``_parse_streaming_event`` so that method stays under
        the cyclomatic-complexity guardrail; the delta has its own
        sub-dispatch on ``delta.type``.
        """
        delta = json_obj.get("delta", {})
        delta_type = delta.get("type")

        if delta_type == DeltaType.TEXT_DELTA:
            text = delta.get("text")
            if text:
                return ParsedResponse(
                    perf_ns=response.perf_ns,
                    data=TextResponseData(text=text),
                )
            return None

        if delta_type == DeltaType.THINKING_DELTA:
            thinking = delta.get("thinking")
            if thinking:
                return ParsedResponse(
                    perf_ns=response.perf_ns,
                    data=ReasoningResponseData(reasoning=thinking),
                )
            return None

        if delta_type == DeltaType.INPUT_JSON_DELTA:
            # Tool-call argument fragments are model-generated tokens that
            # usage.output_tokens counts; surface them so client-side OSL
            # matches the non-streaming parse of the same content.
            partial = delta.get("partial_json")
            if partial:
                return ParsedResponse(
                    perf_ns=response.perf_ns,
                    data=ToolCallResponseData(tool_call_text=partial),
                )
            return None

        # signature_delta / unknown -> drop silently; signature material is
        # not tokenizable output.
        return None
