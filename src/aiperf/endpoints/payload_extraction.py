# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Payload-input extraction helpers shared by all chat-like endpoints.

Lives separately from ``base_endpoint.py`` because the walk logic is
sizeable and self-contained: it is exercised exclusively through
``BaseEndpoint.extract_payload_inputs``, which composes these primitives
against the endpoint's ``PART_TYPES`` map.
"""

from __future__ import annotations

import contextlib
from typing import Any

import orjson

from aiperf.common.enums import MediaType
from aiperf.common.models import ExtractedPayload


def extract_inputs(
    payload: dict[str, Any],
    part_types: dict[MediaType, set[str]],
) -> ExtractedPayload:
    """Single O(n) walk yielding tokenisable text + per-modality counts.

    Covers chat ``messages``, Responses ``input``, completions ``prompt``,
    embeddings ``input``, rankings ``query`` + ``passages``, and HuggingFace
    ``inputs``. Endpoints with a non-standard payload shape override
    ``BaseEndpoint.extract_payload_inputs`` directly.
    """
    result = ExtractedPayload()
    # Reverse index the part-type set: ``{"text": MediaType.TEXT,
    # "image_url": MediaType.IMAGE, ...}``. Built per-call - the map is
    # small and per-part lookup is O(1).
    type_to_media: dict[str, MediaType] = {
        type_name: media_type
        for media_type, type_names in part_types.items()
        for type_name in type_names
    }

    found_items_shape = _walk_items_arrays(payload, result, type_to_media)
    _walk_tools_schema(payload, result)

    if found_items_shape:
        return result

    _walk_flat_fallbacks(payload, result)
    return result


def _walk_items_arrays(
    payload: dict[str, Any],
    result: ExtractedPayload,
    type_to_media: dict[str, MediaType],
) -> bool:
    """Walk chat ``messages`` / Responses ``input`` arrays.

    Returns True iff a recognisable items-array was found (so flat-field
    fallbacks should not also fire).
    """
    found = False
    for items_field in ("messages", "input"):
        items = payload.get(items_field)
        if not isinstance(items, list) or not items:
            continue
        # Disambiguate Responses/chat message arrays from embeddings
        # ``input: [str, ...]``: the former always carries dicts with a
        # ``role`` key OR a Responses-style item ``type`` key
        # (``function_call``, ``function_call_output``, ``message``,
        # ``reasoning``, ...). Pure embedding strings have neither.
        if not any(isinstance(i, dict) and ("role" in i or "type" in i) for i in items):
            continue
        found = True
        for item in items:
            if isinstance(item, dict):
                _walk_item(item, result, type_to_media)
    return found


def _walk_item(
    item: dict[str, Any],
    result: ExtractedPayload,
    type_to_media: dict[str, MediaType],
) -> None:
    """Walk one chat/Responses item: content, tool_calls, function_call(_output)."""
    _walk_item_content(item, result, type_to_media)
    _walk_item_tool_calls(item, result)
    _walk_item_function_call(item, result)


def _walk_item_content(
    item: dict[str, Any],
    result: ExtractedPayload,
    type_to_media: dict[str, MediaType],
) -> None:
    """Chat-shape ``content`` (string or content-parts array)."""
    content = item.get("content")
    if isinstance(content, str):
        result.texts.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                _walk_content_part(part, result, type_to_media)


def _walk_content_part(
    part: dict[str, Any],
    result: ExtractedPayload,
    type_to_media: dict[str, MediaType],
) -> None:
    """Dispatch one content part against the media-type map."""
    media = type_to_media.get(part.get("type"))
    if media is MediaType.TEXT:
        text = part.get("text")
        if isinstance(text, str):
            result.texts.append(text)
    elif media is MediaType.IMAGE:
        result.image_count += 1
    elif media is MediaType.AUDIO:
        result.audio_count += 1
    elif media is MediaType.VIDEO:
        result.video_count += 1


def _walk_item_tool_calls(item: dict[str, Any], result: ExtractedPayload) -> None:
    """Chat-shape assistant message replaying earlier ``tool_calls``.

    Each call's ``function.name`` and ``function.arguments`` are tokens the
    model previously generated, and the server tokenises them on input
    replay. Without this the ISL of agent-history replays is undercounted
    by everything in those calls.
    """
    tool_calls = item.get("tool_calls")
    if not isinstance(tool_calls, list):
        return
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        if isinstance(fn, dict):
            _collect_str_fields(fn, ("name", "arguments"), result.texts)


def _walk_item_function_call(item: dict[str, Any], result: ExtractedPayload) -> None:
    """Responses-shape replayed ``function_call`` and ``function_call_output``."""
    item_type = item.get("type")
    if item_type == "function_call":
        _collect_str_fields(item, ("name", "arguments"), result.texts)
    elif item_type == "function_call_output":
        output_text = item.get("output")
        if isinstance(output_text, str) and output_text:
            result.texts.append(output_text)


def _walk_tools_schema(payload: dict[str, Any], result: ExtractedPayload) -> None:
    """Walk top-level ``tools`` schemas.

    Schema text the server tokenises into the prefix of every request:
    ``function.name``, ``function.description``, ``function.parameters``
    (serialised), plus any future tool shape's ``name``/``description``.
    Counts toward ISL.
    """
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") or {}
        # Chat-shape: nested under ``function``. Responses-shape: fields
        # can sit at the top level. Walk both.
        for source in (fn, tool):
            if isinstance(source, dict):
                _collect_tool_source(source, result)


def _collect_tool_source(source: dict[str, Any], result: ExtractedPayload) -> None:
    """Collect ``name``/``description`` strings and serialised parameters."""
    _collect_str_fields(source, ("name", "description"), result.texts)
    parameters = source.get("parameters")
    if isinstance(parameters, dict):
        # Serialise once; the tokeniser sees the same JSON the server
        # would when prepending the tool schema to the prompt.
        with contextlib.suppress(TypeError):
            result.texts.append(orjson.dumps(parameters).decode())


def _collect_str_fields(
    source: dict[str, Any], keys: tuple[str, ...], out: list[str]
) -> None:
    """Append every non-empty string ``source[key]`` to ``out``."""
    for key in keys:
        value = source.get(key)
        if isinstance(value, str) and value:
            out.append(value)


def _walk_flat_fallbacks(payload: dict[str, Any], result: ExtractedPayload) -> None:
    """Flat-field fallback shapes (completions / embeddings / rankings / HF).

    Only consulted when no items-array was found so embeddings
    ``input: [str, ...]`` doesn't get double-counted with the chat/Responses
    walk. Each shape early-returns so a plugin that accidentally emitted
    two shapes doesn't silently double-count.
    """
    if _append_string_or_list(payload, "prompt", result):
        return
    if _append_string_or_list(payload, "input", result):
        return
    if _append_query_passages(payload, result):
        return

    hf = payload.get("inputs")
    if isinstance(hf, str):
        result.texts.append(hf)


def _append_string_or_list(
    payload: dict[str, Any], key: str, result: ExtractedPayload
) -> bool:
    """Append payload[key] if it's a string or list-of-strings; True if matched.

    Also handles pre-tokenised embeddings shapes (``list[int]`` and
    ``list[list[int]]``): instead of pushing strings to ``texts`` (which
    the tokeniser would re-process), we sum the int-list lengths into
    ``pretokenised_token_count`` so consumers can add the count to ISL
    directly. This catches the OpenAI embeddings ``input: token_ids``
    form that would otherwise silently zero-count.
    """
    value = payload.get(key)
    if isinstance(value, str):
        result.texts.append(value)
        return True
    if isinstance(value, list):
        if all(isinstance(s, str) for s in value):
            result.texts.extend(value)
            return True
        if value and all(isinstance(s, int) for s in value):
            # Single pre-tokenised sequence: list[int].
            result.pretokenised_token_count += len(value)
            return True
        if value and all(
            isinstance(s, list) and all(isinstance(t, int) for t in s) for s in value
        ):
            # Batch of pre-tokenised sequences: list[list[int]].
            result.pretokenised_token_count += sum(len(s) for s in value)
            return True
    return False


def _append_query_passages(payload: dict[str, Any], result: ExtractedPayload) -> bool:
    """Rankings shape: ``query`` + ``passages`` (strings or {"text": ...} dicts)."""
    query = payload.get("query")
    passages = payload.get("passages")
    if not (isinstance(query, str) and isinstance(passages, list)):
        return False
    result.texts.append(query)
    for p in passages:
        if isinstance(p, str):
            result.texts.append(p)
        elif isinstance(p, dict) and isinstance(p.get("text"), str):
            result.texts.append(p["text"])
    return True
