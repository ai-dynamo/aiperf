# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire-encode/decode dispatch between Pydantic ``Message`` and ``msgspec.Struct``.

Phase 3 of the records-pipeline msgspec migration ships heterogeneous message
families on the same ZMQ channels: control-plane messages stay Pydantic (JSON
on the wire), the hot-path records-pipeline messages become ``msgspec.Struct``
and ride the wire as ``msgspec.msgpack`` (binary, ~25-40% smaller than JSON for
typical records-pipeline payloads, and faster to encode/decode).

This module is the single chokepoint that picks the right codec for each
direction.

Outbound (encode):
* Pydantic ``Message`` subclasses keep using :meth:`Message.to_json_bytes` --
  JSON on the wire.
* ``msgspec.Struct`` instances use a shared :class:`msgspec.msgpack.Encoder`
  (encoders cache their schema, so one instance is fine for all Structs) --
  msgpack on the wire.

Inbound (decode):
* Sniff the first byte: ``{`` (0x7b) marks JSON (Pydantic path), anything else
  is treated as msgpack (msgspec path).
* On the msgpack path, use the cached tagged-union
  :class:`msgspec.msgpack.Decoder` built from registered Struct subclasses.
* On the JSON path, fall through to :meth:`Message.from_json` (the existing
  Pydantic ``AutoRoutedModel`` discriminated dispatch).

Modules that own a msgspec-backed message register it once at import time
via :func:`register_msgspec_message`. The registry is keyed by the message
type string (``str(MessageType.X)``).
"""

from __future__ import annotations

from functools import reduce
from operator import or_
from typing import Any

import msgspec

from aiperf.common.messages.base_messages import Message
from aiperf.common.types import MessageTypeT

# JSON envelopes always start with ``{`` (0x7b) because every Pydantic message
# serializes as a top-level object. msgpack maps for our Structs always start
# in the fixmap range (0x80-0x8f) or the map16/map32 prefixes (0xde/0xdf), so a
# leading 0x7b is an unambiguous JSON sentinel.
_JSON_LEAD_BYTE = 0x7B

# Shared encoder for every msgspec.Struct message. msgspec.msgpack.Encoder is
# schema-agnostic; one instance handles all types and caches schemas by class.
_MSGSPEC_ENCODER = msgspec.msgpack.Encoder()

# Reverse map: message-type string -> registered ``msgspec.Struct`` class.
_MSGSPEC_CLASSES: dict[str, type[msgspec.Struct]] = {}

# Cached decoder for the union of every registered msgspec message. Rebuilt on
# registration so inbound msgpack decodes directly into its tagged Struct class.
_MSGSPEC_UNION_DECODER: msgspec.msgpack.Decoder[Any] | None = None


def _rebuild_msgspec_union_decoder() -> None:
    """Rebuild the tagged-union decoder from registered msgspec message classes."""
    global _MSGSPEC_UNION_DECODER
    classes = tuple(_MSGSPEC_CLASSES.values())
    if not classes:
        _MSGSPEC_UNION_DECODER = None
        return
    union_type = reduce(or_, classes)
    _MSGSPEC_UNION_DECODER = msgspec.msgpack.Decoder(union_type)


def register_msgspec_message(
    message_type: MessageTypeT, cls: type[msgspec.Struct]
) -> None:
    """Register a ``msgspec.Struct`` class for wire dispatch.

    Idempotent on identical re-registration; raises on conflicting class for
    the same ``message_type`` (catches accidental double-registration in tests
    or import cycles).
    """
    key = str(message_type)
    existing = _MSGSPEC_CLASSES.get(key)
    if existing is cls:
        return
    if existing is not None:
        raise ValueError(
            f"msgspec wire registry conflict for {key}: {existing!r} vs {cls!r}"
        )
    _MSGSPEC_CLASSES[key] = cls
    _rebuild_msgspec_union_decoder()


def encode_message(message: Message | msgspec.Struct) -> bytes:
    """Encode a message for the wire, dispatching by encoder family.

    msgspec.Struct -> msgpack bytes. Pydantic Message -> JSON bytes.
    """
    if isinstance(message, msgspec.Struct):
        return _MSGSPEC_ENCODER.encode(message)
    return message.to_json_bytes()


def decode_message(message_bytes: bytes) -> Message | msgspec.Struct:
    """Decode wire bytes into a message instance.

    Sniffs the first byte to distinguish JSON (Pydantic) from msgpack (msgspec):
    JSON envelopes lead with ``{`` (0x7b); msgpack maps lead with a fixmap
    (0x80-0x8f) or map16/map32 prefix (0xde/0xdf).

    On the msgpack path, decodes through the cached tagged-union decoder built
    from registered msgspec-backed message classes. On the JSON path, falls back
    to the Pydantic ``AutoRoutedModel`` dispatch.
    """
    if (
        message_bytes
        and message_bytes[0] != _JSON_LEAD_BYTE
        and _MSGSPEC_UNION_DECODER is not None
    ):
        try:
            return _MSGSPEC_UNION_DECODER.decode(message_bytes)
        except msgspec.DecodeError:
            pass
    return Message.from_json(message_bytes)
