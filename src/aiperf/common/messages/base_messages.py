# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, ClassVar

import msgspec
import orjson
from typing_extensions import Self

from aiperf.common.enums import MessageType
from aiperf.common.models.base_models import _msgspec_dec_hook, _msgspec_enc_hook
from aiperf.common.models.error_models import ErrorDetails

_JSON_ENCODER = msgspec.json.Encoder(enc_hook=_msgspec_enc_hook)
_MSGPACK_ENCODER = msgspec.msgpack.Encoder(enc_hook=_msgspec_enc_hook)


def _known_message_tags() -> set[str]:
    """Return the set of every registered ``MessageType`` string value."""
    return {m.value for m in MessageType}


class Message(
    msgspec.Struct,
    tag_field="message_type",
    kw_only=True,
    omit_defaults=True,
):
    """Base message class — msgspec.Struct tagged union on ``message_type``.

    Subclasses register their tag via ``class Foo(Message, tag=MessageType.FOO.value):``.
    msgspec resolves the concrete subclass on decode with zero registry machinery.

    Compatibility shims (``from_json``, ``to_json_bytes``, ``model_dump_json``,
    ``model_dump``, ``__str__``) preserve the prior Pydantic-backed API so
    transport clients, codecs, and tests need no call-site changes.
    """

    request_ns: int | None = None
    request_id: str | None = None

    _json_decoder_cache: ClassVar[msgspec.json.Decoder | None] = None
    _msgpack_decoder_cache: ClassVar[msgspec.msgpack.Decoder | None] = None

    @classmethod
    def _all_tagged_subclasses(cls) -> list[type]:
        """Collect every concrete tagged subclass of ``cls`` (recursively).

        msgspec auto-assigns ``tag=<ClassName>`` when ``tag_field`` is set and
        no explicit ``tag=`` is supplied on the subclass. Filter those abstract
        intermediaries out by keeping only classes whose tag matches a valid
        ``MessageType`` value.
        """
        seen: set[type] = set()
        stack: list[type] = list(cls.__subclasses__())
        while stack:
            sub = stack.pop()
            if sub in seen:
                continue
            seen.add(sub)
            stack.extend(sub.__subclasses__())
        # Avoid a hard circular import — enum members are loaded lazily.
        known_tags = _known_message_tags()
        tagged = [
            s for s in seen if getattr(s.__struct_config__, "tag", None) in known_tags
        ]
        return tagged

    @classmethod
    def _union_type(cls) -> Any:
        """Build a Union type of all concrete tagged Message subclasses.

        For a concrete subclass (i.e. one whose own tag matches a real
        ``MessageType``), we return the class itself so ``from_json``
        round-trips a specific envelope without collapsing into a union.
        """
        known_tags = _known_message_tags()
        cls_tag = getattr(cls.__struct_config__, "tag", None)
        if cls_tag in known_tags:
            return cls
        tagged = cls._all_tagged_subclasses()
        if not tagged:
            return cls
        result: Any = tagged[0]
        for t in tagged[1:]:
            result = result | t
        return result

    @classmethod
    def _json_decoder(cls) -> msgspec.json.Decoder:
        cached: msgspec.json.Decoder | None = cls.__dict__.get("_json_decoder_cache")
        if cached is None:
            cached = msgspec.json.Decoder(
                type=cls._union_type(), dec_hook=_msgspec_dec_hook
            )
            # Set on the specific subclass (not on Message) so each concrete
            # type gets its own decoder.
            cls._json_decoder_cache = cached
        return cached

    @classmethod
    def _msgpack_decoder(cls) -> msgspec.msgpack.Decoder:
        cached: msgspec.msgpack.Decoder | None = cls.__dict__.get(
            "_msgpack_decoder_cache"
        )
        if cached is None:
            cached = msgspec.msgpack.Decoder(
                type=cls._union_type(), dec_hook=_msgspec_dec_hook
            )
            cls._msgpack_decoder_cache = cached
        return cached

    @classmethod
    def from_json(cls, json_or_dict: str | bytes | bytearray | dict[str, Any]) -> Self:
        """Decode bytes/str/dict into the correct tagged-union subclass."""
        if isinstance(json_or_dict, dict):
            # msgspec picks the tagged-union branch from the discriminator BEFORE
            # dec_hook runs, so coerce the message_type value to a raw str up
            # front in case callers passed a ``MessageType`` enum instance.
            payload = json_or_dict
            mt = payload.get("message_type")
            if mt is not None and not isinstance(mt, str):
                payload = {**payload, "message_type": str(mt)}
            elif mt is not None and type(mt) is not str:
                # ExtensibleStrEnum IS a str subclass but has a custom metaclass;
                # normalize to plain str so msgspec's tag matcher accepts it.
                payload = {**payload, "message_type": str.__str__(mt)}
            return msgspec.convert(
                payload, cls._union_type(), strict=False, dec_hook=_msgspec_dec_hook
            )
        return cls._json_decoder().decode(json_or_dict)

    def to_json_bytes(self) -> bytes:
        """Serialize to JSON bytes (wire-compatible with prior Pydantic path)."""
        return _JSON_ENCODER.encode(self)

    def to_msgpack_bytes(self) -> bytes:
        """Serialize to msgpack bytes — used by the P3 single-codec path."""
        return _MSGPACK_ENCODER.encode(self)

    def model_dump(
        self,
        *,
        exclude_none: bool = False,
        mode: str | None = None,
        by_alias: bool = False,
    ) -> dict[str, Any]:
        """Pydantic-compat shim — ``mode`` and ``by_alias`` are accepted but ignored.

        ``omit_defaults=True`` on the Struct already drops None defaults, so the
        ``exclude_none`` branch is a no-op for the common path. Explicit non-None
        values are preserved.
        """
        data = msgspec.to_builtins(self, enc_hook=_msgspec_enc_hook)
        if exclude_none:
            return {k: v for k, v in data.items() if v is not None}
        return data

    def model_dump_json(
        self, *, exclude_none: bool = True, indent: int | None = None
    ) -> str:
        """Pydantic-compat shim returning a JSON string."""
        encoded = _JSON_ENCODER.encode(self)
        if indent is not None:
            return orjson.dumps(
                orjson.loads(encoded), option=orjson.OPT_INDENT_2
            ).decode()
        return encoded.decode()

    def __str__(self) -> str:
        return self.to_json_bytes().decode()

    @property
    def message_type(self) -> MessageType:
        """Expose the msgspec tag as the legacy ``message_type`` attribute."""
        return self.__struct_config__.tag  # type: ignore[return-value]


class ErrorMessage(Message, kw_only=True, tag=MessageType.ERROR.value):
    """Envelope carrying an ErrorDetails payload."""

    error: ErrorDetails
