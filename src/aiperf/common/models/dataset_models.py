# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, ClassVar

import msgspec

from aiperf.common.enums import ConversationContextMode, MediaType
from aiperf.common.models.base_models import PydanticStructMixin
from aiperf.common.types import MediaTypeT
from aiperf.plugin.enums import DatasetClientStoreType, DatasetSamplingStrategy


class DatasetClientMetadata(
    PydanticStructMixin,
    msgspec.Struct,
    tag_field="client_type",
    frozen=True,
    kw_only=True,
    omit_defaults=True,
):
    """Base class for dataset client access metadata.

    Discriminated union keyed on ``client_type`` — msgspec routes dicts to
    the correct subclass on decode. Every subclass must declare ``tag=...``
    or it becomes unreachable via the union decoder.
    """

    @property
    def client_type(self) -> str:
        """String tag that identifies the concrete client store type.

        Mirrored on the encoded payload by msgspec; exposed as an attribute
        so existing consumers (plugin lookup, log messages) keep working.
        """
        return type(self).__struct_config__.tag

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: Any,
    ) -> Any:
        # Override the default PydanticStructMixin schema so that validation
        # against the base class dispatches across every tagged subclass.
        # msgspec.convert requires a Union type when decoding via tag; the
        # bare base class does not route.
        from pydantic_core import core_schema as _core_schema

        from aiperf.common.models.base_models import (
            _msgspec_dec_hook,
            _msgspec_enc_hook,
        )

        def _iter_subclasses() -> list[type]:
            seen: list[type] = []
            stack: list[type] = list(cls.__subclasses__())
            while stack:
                sub = stack.pop()
                if sub in seen:
                    continue
                seen.append(sub)
                stack.extend(sub.__subclasses__())
            return seen

        def _union_target() -> Any:
            subs = _iter_subclasses()
            if not subs:
                return cls
            if len(subs) == 1:
                return subs[0]
            import typing as _typing

            return _typing.Union[tuple(subs)]  # noqa: UP007

        def _validate(value: Any) -> Any:
            if isinstance(value, cls):
                return value
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected dict or {cls.__name__} instance, got {type(value).__name__}"
                )
            return msgspec.convert(value, _union_target(), dec_hook=_msgspec_dec_hook)

        def _serialize(value: Any) -> Any:
            return msgspec.to_builtins(value, enc_hook=_msgspec_enc_hook)

        return _core_schema.no_info_plain_validator_function(
            _validate,
            serialization=_core_schema.plain_serializer_function_ser_schema(
                _serialize,
                return_schema=_core_schema.any_schema(),
                when_used="always",
            ),
        )


class MemoryMapClientMetadata(
    DatasetClientMetadata,
    tag=DatasetClientStoreType.MEMORY_MAP.value,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
):
    """Client metadata for memory-mapped dataset access.

    Contains paths to mmap files that workers use for zero-copy,
    O(1) conversation lookups. For Kubernetes deployments, also includes
    paths to pre-compressed files for efficient network transfer.
    """

    data_file_path: Path
    index_file_path: Path
    conversation_count: int = 0
    total_size_bytes: int = 0
    compressed: bool = False
    compressed_size_bytes: int = 0


# Hot-path dataset models use msgspec.Struct for ~3-4x faster encode/decode/construct
# vs Pydantic v2. These are instantiated per-turn per-request at high QPS.
# The PydanticStructMixin lets these structs appear as fields on Pydantic
# envelopes (e.g. ConversationResponseMessage) without bespoke serialization.


class Media(
    PydanticStructMixin,
    msgspec.Struct,
    kw_only=True,
    omit_defaults=False,
):
    """Base class for all media fields. Contains name and contents of the media data."""

    name: str = ""
    contents: list[str] = msgspec.field(default_factory=list)


class Text(Media):
    """Media that contains text/prompt data."""

    media_type: ClassVar[MediaTypeT] = MediaType.TEXT


class Image(Media):
    """Media that contains image data."""

    media_type: ClassVar[MediaTypeT] = MediaType.IMAGE


class Audio(Media):
    """Media that contains audio data."""

    media_type: ClassVar[MediaTypeT] = MediaType.AUDIO


class Video(Media):
    """Media that contains video data."""

    media_type: ClassVar[MediaTypeT] = MediaType.VIDEO


class TurnMetadata(
    PydanticStructMixin,
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
):
    """Metadata of a turn: absolute timestamp (ms) and/or delay (ms)."""

    timestamp_ms: int | float | None = None
    delay_ms: int | float | None = None


class Turn(
    PydanticStructMixin,
    msgspec.Struct,
    kw_only=True,
    omit_defaults=False,
):
    """A dataset representation of a single turn within a conversation.

    A turn is a single interaction between a user and an AI assistant,
    and it contains timestamp, delay, and raw data that user sends in each turn.
    """

    model: str | None = None
    role: str | None = None
    timestamp: int | float | None = None
    delay: int | float | None = None
    max_tokens: int | None = None
    raw_messages: list[dict[str, Any]] | None = None
    raw_tools: list[dict[str, Any]] | None = None
    texts: list[Text] = msgspec.field(default_factory=list)
    images: list[Image] = msgspec.field(default_factory=list)
    audios: list[Audio] = msgspec.field(default_factory=list)
    videos: list[Video] = msgspec.field(default_factory=list)

    def metadata(self) -> "TurnMetadata":
        """Get the metadata of the turn."""
        return TurnMetadata(
            timestamp_ms=self.timestamp,
            delay_ms=self.delay,
        )

    def copy_with_stripped_media(self) -> "Turn":
        """Create a copy of this turn with multimodal data replaced by placeholders.

        This preserves text data (needed for tokenization) and raw messages/tools
        (needed for API payload reconstruction) but replaces potentially large
        image/audio/video contents with small placeholder strings. This is
        more efficient than a full deep copy followed by stripping.

        Returns:
            A new Turn with stripped multimodal contents and messages.
        """
        return Turn(
            model=self.model,
            role=self.role,
            timestamp=self.timestamp,
            delay=self.delay,
            max_tokens=self.max_tokens,
            raw_messages=list(self.raw_messages)
            if self.raw_messages is not None
            else None,
            raw_tools=list(self.raw_tools) if self.raw_tools is not None else None,
            texts=[Text(name=t.name, contents=list(t.contents)) for t in self.texts],
            images=[
                Image(
                    name=img.name,
                    contents=[f"image_{i}" for i in range(len(img.contents))],
                )
                for img in self.images
            ],
            audios=[
                Audio(
                    name=aud.name,
                    contents=[f"audio_{i}" for i in range(len(aud.contents))],
                )
                for aud in self.audios
            ],
            videos=[
                Video(
                    name=vid.name,
                    contents=[f"video_{i}" for i in range(len(vid.contents))],
                )
                for vid in self.videos
            ],
        )


class ConversationMetadata(
    PydanticStructMixin,
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
):
    """Metadata of a conversation."""

    conversation_id: str
    context_mode: ConversationContextMode | None = None
    turns: list[TurnMetadata] = msgspec.field(default_factory=list)


class DatasetMetadata(
    PydanticStructMixin,
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
):
    """Metadata of a dataset's structure.

    Contains dataset structure information (conversations, timing) used by
    timing strategies to schedule requests. Does NOT contain data access
    metadata - that's in DatasetClientMetadata (sent separately in
    DatasetConfiguredNotification).
    """

    sampling_strategy: DatasetSamplingStrategy
    conversations: list[ConversationMetadata] = msgspec.field(default_factory=list)
    has_timing_data: bool = False
    # Dataset-level default for how prior turns are accumulated. Set by the
    # loader based on dataset format semantics. Individual conversations can
    # override this via their own context_mode field.
    default_context_mode: ConversationContextMode | None = None

    def __post_init__(self) -> None:
        if (
            self.default_context_mode
            == ConversationContextMode.MESSAGE_ARRAY_WITHOUT_RESPONSES
        ):
            raise ValueError(
                f"{ConversationContextMode.MESSAGE_ARRAY_WITHOUT_RESPONSES} is not yet supported"
            )

    @property
    def total_turn_count(self) -> int:
        """Get the total number of turns in the dataset."""
        return sum(len(conversation.turns) for conversation in self.conversations)

    @property
    def average_turn_count(self) -> float:
        """Get the average number of turns across all conversations in the dataset."""
        if len(self.conversations) == 0:
            return 0
        return self.total_turn_count / len(self.conversations)


class Conversation(
    PydanticStructMixin,
    msgspec.Struct,
    kw_only=True,
    omit_defaults=False,
):
    """A dataset representation of a full conversation.

    A conversation is a sequence of turns between a user and an endpoint,
    and it contains the session ID and all the turns that consists the conversation.
    """

    session_id: str = ""
    context_mode: ConversationContextMode | None = None
    turns: list[Turn] = msgspec.field(default_factory=list)
    system_message: str | None = None
    user_context_message: str | None = None

    def __post_init__(self) -> None:
        # Parity with the former Pydantic field_validator on context_mode.
        if self.context_mode == ConversationContextMode.MESSAGE_ARRAY_WITHOUT_RESPONSES:
            raise ValueError(
                f"{ConversationContextMode.MESSAGE_ARRAY_WITHOUT_RESPONSES} is not yet supported"
            )

    def metadata(self) -> ConversationMetadata:
        """Get the metadata of the conversation."""
        return ConversationMetadata(
            conversation_id=self.session_id,
            context_mode=self.context_mode,
            turns=[turn.metadata() for turn in self.turns],
        )


class SessionPayloads(
    PydanticStructMixin,
    msgspec.Struct,
    frozen=True,
    kw_only=True,
):
    """A single session, with its session ID and a list of formatted payloads (one per turn)."""

    session_id: str | None = None
    # Formatted payloads in the session (one per turn), already prepared for
    # the model and endpoint.
    payloads: list[dict[str, Any]] = msgspec.field(default_factory=list)


class InputsFile(
    PydanticStructMixin,
    msgspec.Struct,
    frozen=True,
    kw_only=True,
):
    """A list of all dataset sessions. Each session contains a list of formatted payloads (one per turn).
    This is similar to the format used by GenAI-Perf for the inputs.json file.

    Intentionally does not set ``omit_defaults=True``: the on-disk inputs.json
    schema contract always includes ``"data": [...]`` (the tutorials and
    downstream tools expect it), and SessionPayloads always includes
    ``session_id`` and ``payloads``.
    """

    data: list[SessionPayloads] = msgspec.field(default_factory=list)
