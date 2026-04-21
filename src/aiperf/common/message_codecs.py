# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single msgpack codec for inter-service transport.

After P3 every ZMQ channel that carried Pydantic envelopes in the
transitional phase now carries msgspec structs directly. The Pydantic
fallback hooks are gone; ``MsgspecStructCodec`` is the only codec.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import msgspec

from aiperf.common.messages import Message
from aiperf.common.models.base_models import _msgspec_dec_hook, _msgspec_enc_hook


@runtime_checkable
class MessageCodecProtocol(Protocol):
    """Codec interface for transport clients."""

    cache_key: str

    def encode(self, message: Any) -> bytes: ...
    def decode(self, data: bytes) -> Any: ...


class MsgspecStructCodec:
    """Typed msgpack codec for msgspec struct payloads.

    ``_msgspec_enc_hook`` / ``_msgspec_dec_hook`` are wired in so embedded
    ``ExtensibleStrEnum`` fields (``MessageType``, ``TimingMode``,
    ``ArrivalPattern``, ...) round-trip correctly — msgspec's native enum
    handling doesn't recognise the custom metaclass. See
    ``gotcha_msgspec_extensible_str_enum``.
    """

    def __init__(self, *, decode_type: Any, cache_key: str) -> None:
        self.cache_key = cache_key
        self._encoder = msgspec.msgpack.Encoder(enc_hook=_msgspec_enc_hook)
        self._decoder = msgspec.msgpack.Decoder(
            type=decode_type, dec_hook=_msgspec_dec_hook
        )

    def encode(self, message: Any) -> bytes:
        return self._encoder.encode(message)

    def decode(self, data: bytes) -> Any:
        return self._decoder.decode(data)


def _build_message_codec() -> MsgspecStructCodec:
    """Lazy singleton — cannot evaluate ``Message._union_type()`` at import
    time because the concrete subclass registry is built up during module
    imports of the various message modules.
    """
    return MsgspecStructCodec(
        decode_type=Message._union_type(),
        cache_key="msgspec-message",
    )


_MESSAGE_CODEC: MsgspecStructCodec | None = None


def get_message_codec() -> MsgspecStructCodec:
    """Return the shared default codec for ``Message`` envelopes.

    Instantiated on first access so every Message subclass is registered
    before ``Message._union_type()`` snapshots the tagged-union types.
    """
    global _MESSAGE_CODEC
    if _MESSAGE_CODEC is None:
        _MESSAGE_CODEC = _build_message_codec()
    return _MESSAGE_CODEC


def codec_cache_key(codec: MessageCodecProtocol | None) -> str:
    """Return a stable cache token for a codec."""
    return codec.cache_key if codec is not None else "msgspec-message"
