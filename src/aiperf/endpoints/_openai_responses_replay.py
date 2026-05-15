# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FORK-replay capture helpers for the OpenAI Responses endpoint.

Pulls the dedup-by-id assembly of ``output[]`` items out of
``ResponsesEndpoint`` so the main file stays under the file-size cap.
The Responses API accepts the same item shapes in ``input`` that it
emits in ``output``, so the captured items go into ``raw_messages``
and ``build_messages`` extends them onto the next request's ``input``
array verbatim — a FORK-mode DAG child therefore sees the parent's
full output (including tool/function calls), not just its text.

Capture is a **union** of two sources, deduplicated by item ``id``:

- ``response.completed.response.output[]`` — the assembled list
  ordering, canonical when present.
- ``response.output_item.done.item`` events — each carries one
  fully-assembled output item.

Real-world traces show both sources can drop items the other captured
(~3% of streaming turns either way), so the union with id-keyed dedup
is what makes the capture lossless.
"""

from __future__ import annotations

from typing import Any

import orjson

from aiperf.common.types import JsonObject

# Stream-end failure types: do NOT splice partial output items into a
# FORK child's history when the parent's stream ended in failure — the
# child would treat partials as the parent's authoritative reply.
_FAILURE_EVENT_TYPES = frozenset(
    {"response.failed", "response.incomplete", "response.error", "error"}
)


def is_failure_event(json_obj: JsonObject) -> bool:
    """True when ``json_obj`` is a Responses-API stream-end failure event."""
    return json_obj.get("type") in _FAILURE_EVENT_TYPES


def collect_response_items(
    json_obj: JsonObject,
    items_by_key: dict[str, dict[str, Any]],
    done_items: list[dict[str, Any]],
) -> None:
    """Pull output items from a single response payload.

    Non-streaming full-response objects and streaming
    ``response.completed`` events both contribute directly to
    ``items_by_key`` (canonical ordering). Streaming
    ``response.output_item.done`` events buffer into ``done_items`` so
    completed-ordering wins when both sources are present.
    """
    if json_obj.get("object") == "response":
        merge_output_list(items_by_key, json_obj.get("output"))
        return

    event_type = json_obj.get("type")

    if event_type == "response.completed":
        resp = json_obj.get("response") or {}
        merge_output_list(items_by_key, resp.get("output"))
        return

    if event_type == "response.output_item.done":
        item = json_obj.get("item")
        if isinstance(item, dict):
            done_items.append(item)


def merge_output_list(
    items_by_key: dict[str, dict[str, Any]],
    output: Any,
) -> None:
    """Merge every dict in an ``output[]`` list into ``items_by_key``."""
    if not isinstance(output, list):
        return
    for item in output:
        if isinstance(item, dict):
            merge_item(items_by_key, item)


def merge_item(items_by_key: dict[str, dict[str, Any]], item: dict[str, Any]) -> None:
    """Insert ``item`` into ``items_by_key`` if its id is novel.

    Dedup key precedence: ``id`` > ``call_id`` > ``item_id``. Items
    that carry none of these three (rare but possible for a
    not-yet-typed future item shape) get a synthesised key from
    ``(type, hash(json))`` so structurally-identical duplicates still
    collapse to one but distinct items don't collide.
    """
    key = item.get("id") or item.get("call_id") or item.get("item_id")
    if not key:
        try:
            payload_hash = hash(orjson.dumps(item, option=orjson.OPT_SORT_KEYS))
        except TypeError:
            payload_hash = id(item)
        key = f"{item.get('type', '?')}::{payload_hash}"
    items_by_key.setdefault(key, item)
