# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable message codecs for transport clients.

PUSH/PULL historically assumed JSON bytes carrying Pydantic ``Message`` models.
This module adds a small codec abstraction so individual channels can opt into
MessagePack/msgspec without forking the transport client implementations.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import msgspec

from aiperf.common.messages import Message


@runtime_checkable
class MessageCodecProtocol(Protocol):
    """Codec interface for transport clients."""

    cache_key: str

    def encode(self, message: Any) -> bytes:
        """Serialize a message to bytes."""

    def decode(self, data: bytes) -> Any:
        """Deserialize bytes into a message object."""


class JsonMessageCodec:
    """Default JSON codec using the existing Pydantic message path."""

    cache_key = "json-message"

    def encode(self, message: Message) -> bytes:
        return message.to_json_bytes()

    def decode(self, data: bytes) -> Message:
        return Message.from_json(data)


class PydanticMsgpackCodec:
    """MessagePack codec that still rehydrates into routed Pydantic messages."""

    def __init__(
        self,
        *,
        cache_key: str,
        message_base_type: type[Message] = Message,
    ) -> None:
        self.cache_key = cache_key
        self._message_base_type = message_base_type
        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder()

    def encode(self, message: Message) -> bytes:
        payload = message.model_dump(exclude_none=True, mode="json")
        return self._encoder.encode(payload)

    def decode(self, data: bytes) -> Message:
        payload = self._decoder.decode(data)
        return self._message_base_type.from_json(payload)


class MsgspecStructCodec:
    """Typed MessagePack codec for msgspec struct messages."""

    def __init__(
        self,
        *,
        decode_type: Any,
        cache_key: str,
    ) -> None:
        self.cache_key = cache_key
        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder(type=decode_type)

    def encode(self, message: Any) -> bytes:
        return self._encoder.encode(message)

    def decode(self, data: bytes) -> Any:
        return self._decoder.decode(data)


JSON_MESSAGE_CODEC = JsonMessageCodec()


def codec_cache_key(codec: MessageCodecProtocol | None) -> str:
    """Return a stable cache token for a codec."""

    return codec.cache_key if codec is not None else JSON_MESSAGE_CODEC.cache_key
