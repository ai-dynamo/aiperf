# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable message codecs for transport clients."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import msgspec
from pydantic import BaseModel

from aiperf.common.messages import Message
from aiperf.common.models.base_models import _msgspec_dec_hook, _msgspec_enc_hook


def _enc_hook(obj: Any) -> Any:
    """Encoder fallback for legacy Pydantic fields embedded in msgspec structs.

    After P2 this hook is unreachable for the records path — kept for channels
    still passing Pydantic export models. P3 deletes this.
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json", exclude_none=True)
    raise TypeError(f"Object of type {type(obj).__name__} is not msgpack-encodable")


def _dec_hook(t: type, obj: Any) -> Any:
    """Decoder fallback symmetric to ``_enc_hook`` — retired in P3."""
    if isinstance(obj, dict) and isinstance(t, type) and issubclass(t, BaseModel):
        return t.model_validate(obj)
    raise NotImplementedError(
        f"Unsupported msgspec decode target {t!r} for value {type(obj).__name__}"
    )


@runtime_checkable
class MessageCodecProtocol(Protocol):
    """Codec interface for transport clients."""

    cache_key: str

    def encode(self, message: Any) -> bytes: ...
    def decode(self, data: bytes) -> Any: ...


class JsonMessageCodec:
    """JSON codec (msgspec-backed). Wire-equivalent to the prior Pydantic path."""

    cache_key = "json-message"

    def __init__(self) -> None:
        self._encoder = msgspec.json.Encoder(enc_hook=_msgspec_enc_hook)
        self._decoder: msgspec.json.Decoder | None = None

    def _get_decoder(self) -> msgspec.json.Decoder:
        if self._decoder is None:
            self._decoder = msgspec.json.Decoder(
                type=Message._union_type(), dec_hook=_msgspec_dec_hook
            )
        return self._decoder

    def encode(self, message: Message) -> bytes:
        return self._encoder.encode(message)

    def decode(self, data: bytes) -> Message:
        return self._get_decoder().decode(data)


class PydanticMsgpackCodec:
    """Transitional: msgpack encode of the tagged ``Message`` union.

    Kept alive through P2 so channels that explicitly opted into this codec
    don't need a simultaneous rewrite. P3 deletes this and collapses all
    traffic onto ``MsgspecStructCodec(decode_type=Message)``.
    """

    def __init__(
        self,
        *,
        cache_key: str,
        message_base_type: type[Message] = Message,
    ) -> None:
        self.cache_key = cache_key
        self._message_base_type = message_base_type
        self._encoder = msgspec.msgpack.Encoder(enc_hook=_msgspec_enc_hook)
        self._decoder: msgspec.msgpack.Decoder | None = None

    def _get_decoder(self) -> msgspec.msgpack.Decoder:
        if self._decoder is None:
            self._decoder = msgspec.msgpack.Decoder(
                type=self._message_base_type._union_type(),
                dec_hook=_msgspec_dec_hook,
            )
        return self._decoder

    def encode(self, message: Message) -> bytes:
        return self._encoder.encode(message)

    def decode(self, data: bytes) -> Message:
        return self._get_decoder().decode(data)


class MsgspecStructCodec:
    """Typed msgpack codec — primary codec for records/raw-inference channels."""

    def __init__(self, *, decode_type: Any, cache_key: str) -> None:
        self.cache_key = cache_key
        self._encoder = msgspec.msgpack.Encoder(enc_hook=_enc_hook)
        self._decoder = msgspec.msgpack.Decoder(type=decode_type, dec_hook=_dec_hook)

    def encode(self, message: Any) -> bytes:
        return self._encoder.encode(message)

    def decode(self, data: bytes) -> Any:
        return self._decoder.decode(data)


JSON_MESSAGE_CODEC = JsonMessageCodec()


def codec_cache_key(codec: MessageCodecProtocol | None) -> str:
    """Return a stable cache token for a codec."""
    return codec.cache_key if codec is not None else JSON_MESSAGE_CODEC.cache_key
