# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared payload formatting logic for dataset processing.

Provides a generator that creates formatted API request payloads from
conversations using an endpoint protocol. Used by both the dataset manager
(inputs.json generation) and the custom composer (payload pre-formatting).
Also provides the chunked encoder the dataset manager streams inputs.json
through.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import orjson

from aiperf.common.constants import BYTES_PER_MIB
from aiperf.common.enums import CreditPhase
from aiperf.common.models import Conversation, InputsFile, SessionPayloads
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.common.models.record_models import RequestInfo
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType


def format_conversation_payloads(
    conversations: Iterable[Conversation],
    model_endpoint: ModelEndpointInfo,
) -> Iterator[tuple[str, int, dict[str, Any]]]:
    """Yield formatted payloads for each turn in the given conversations.

    Creates an endpoint instance and iterates over all turns, producing
    (session_id, turn_index, payload) tuples.

    Args:
        conversations: Conversations to format payloads for.
        model_endpoint: Endpoint configuration for payload formatting.

    Yields:
        Tuples of (session_id, turn_index, formatted_payload_dict).

    Raises:
        NotImplementedError: If the endpoint does not support format_payload.
    """
    EndpointClass = plugins.get_class(PluginType.ENDPOINT, model_endpoint.endpoint.type)
    endpoint = EndpointClass(model_endpoint=model_endpoint)

    for conversation in conversations:
        for i, turn in enumerate(conversation.turns):
            if turn.raw_payload is not None:
                yield conversation.session_id, i, turn.raw_payload
                continue
            request_info = RequestInfo(
                model_endpoint=model_endpoint,
                turns=[turn],
                turn_index=i,
                credit_num=i,
                credit_phase=CreditPhase.PROFILING,
                x_request_id="",
                x_correlation_id="",
                conversation_id=conversation.session_id,
                system_message=conversation.system_message,
                user_context_message=conversation.user_context_message,
            )
            request_info.endpoint_headers = endpoint.get_endpoint_headers(request_info)
            request_info.endpoint_params = endpoint.get_endpoint_params(request_info)
            yield conversation.session_id, i, endpoint.format_payload(request_info)


# Flush threshold for the streamed inputs.json encoder. Bounds the bytes held
# between writes without paying an aiofiles thread hop per payload.
INPUTS_JSON_WRITE_CHUNK_BYTES = BYTES_PER_MIB

_INPUTS_JSON_EMPTY = b'{\n  "data": []\n}'
_INPUTS_JSON_HEAD = b'{\n  "data": ['
_INPUTS_JSON_TAIL = b"\n  ]\n}"
# OPT_INDENT_2 nesting: session objects sit under the "data" array, payload
# objects under each session's "payloads" array.
_SESSION_INDENT = b"\n    "
_PAYLOAD_INDENT = b"\n        "


def _iter_session_pieces(session: SessionPayloads) -> Iterator[bytes]:
    """Yield one session exactly as OPT_INDENT_2 would print it, one payload per piece."""
    dumped = session.model_dump(exclude_none=True, mode="json")
    payloads = dumped.pop("payloads")
    head = b"{"
    if "session_id" in dumped:
        head += b'\n      "session_id": ' + orjson.dumps(dumped["session_id"]) + b","
    if not payloads:
        yield head + b'\n      "payloads": []\n    }'
        return
    yield head + b'\n      "payloads": ['
    for index, payload in enumerate(payloads):
        piece = b"," if index else b""
        yield (
            piece
            + _PAYLOAD_INDENT
            + orjson.dumps(payload, option=orjson.OPT_INDENT_2).replace(
                b"\n", _PAYLOAD_INDENT
            )
        )
    yield b"\n      ]\n    }"


def iter_inputs_json_chunks(
    inputs: InputsFile, chunk_bytes: int = INPUTS_JSON_WRITE_CHUNK_BYTES
) -> Iterator[bytes]:
    """Yield the inputs.json document in chunks flushed at payload boundaries.

    Every chunk but the last is at least ``chunk_bytes`` and none exceeds it by
    more than one encoded payload. orjson escapes control characters inside
    strings, so re-indenting on raw newlines is exact and the concatenated
    chunks are byte-identical to encoding the whole ``InputsFile`` with
    ``OPT_INDENT_2`` in one call.
    """
    if not inputs.data:
        yield _INPUTS_JSON_EMPTY
        return

    buffer = bytearray(_INPUTS_JSON_HEAD)
    for index, session in enumerate(inputs.data):
        if index:
            buffer += b","
        buffer += _SESSION_INDENT
        for piece in _iter_session_pieces(session):
            buffer += piece
            if len(buffer) >= chunk_bytes:
                yield bytes(buffer)
                buffer.clear()
    buffer += _INPUTS_JSON_TAIL
    yield bytes(buffer)
