# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import msgspec
from pydantic import ConfigDict

from aiperf.common.enums import (
    ConversationBranchMode,
    ConversationContextMode,
    MediaType,
)
from aiperf.common.models.branch import ConversationBranchInfo
from aiperf.common.models.prerequisites import TurnPrerequisite
from aiperf.common.types import MediaTypeT
from aiperf.plugin.enums import DatasetClientStoreType, DatasetSamplingStrategy


def _iter_dataset_client_subclasses(cls: type) -> list[type]:
    seen: list[type] = []
    stack: list[type] = list(cls.__subclasses__())
    while stack:
        sub = stack.pop()
        if sub in seen:
            continue
        seen.append(sub)
        stack.extend(sub.__subclasses__())
    return seen


def _dataset_client_union_target(cls: type) -> Any:
    subs = _iter_dataset_client_subclasses(cls)
    if not subs:
        return cls
    if len(subs) == 1:
        return subs[0]
    import typing as _typing

    return _typing.Union[tuple(subs)]  # noqa: UP007


def _make_dataset_client_metadata_validator(cls: type) -> Any:
    from aiperf.common.models.base_models import _msgspec_dec_hook

    def _validate(value: Any) -> Any:
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise ValueError(
                f"Expected dict or {cls.__name__} instance, got {type(value).__name__}"
            )
        return msgspec.convert(
            value, _dataset_client_union_target(cls), dec_hook=_msgspec_dec_hook
        )

    return _validate


def _serialize_dataset_client_metadata(value: Any) -> Any:
    from aiperf.common.models.base_models import _msgspec_enc_hook

    return msgspec.to_builtins(value, enc_hook=_msgspec_enc_hook)


class _PydanticCompatMixin:
    """Mixin providing Pydantic-style ``model_validate`` / ``model_dump``
    methods on msgspec.Struct subclasses. Used by hot-path dataset models
    so callers (tests, loaders) that still speak Pydantic vocabulary keep
    working after the perf restoration to msgspec.
    """

    @classmethod
    def model_validate(cls, value: Any) -> Any:
        if isinstance(value, cls):
            return value
        from aiperf.common.models.base_models import _msgspec_dec_hook

        return msgspec.convert(value, cls, dec_hook=_msgspec_dec_hook)

    @classmethod
    def model_validate_json(cls, value: str | bytes) -> Any:
        from aiperf.common.models.base_models import _msgspec_dec_hook

        return msgspec.json.decode(value, type=cls, dec_hook=_msgspec_dec_hook)

    def model_dump(self, **_: Any) -> dict[str, Any]:
        from aiperf.common.models.base_models import _msgspec_enc_hook

        return msgspec.to_builtins(self, enc_hook=_msgspec_enc_hook)

    def model_dump_json(self, **_: Any) -> str:
        from aiperf.common.models.base_models import _msgspec_enc_hook

        return msgspec.json.encode(self, enc_hook=_msgspec_enc_hook).decode("utf-8")


class DatasetClientMetadata(
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
        """String tag that identifies the concrete client store type."""
        return type(self).__struct_config__.tag

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: Any,
    ) -> Any:
        # Custom core schema for the tagged-union base class so validation
        # dispatches across every tagged subclass. msgspec.convert requires a
        # Union type when decoding via tag; the bare base class does not route.
        from pydantic_core import core_schema as _core_schema

        validate = _make_dataset_client_metadata_validator(cls)
        serialize = _serialize_dataset_client_metadata

        return _core_schema.no_info_plain_validator_function(
            validate,
            serialization=_core_schema.plain_serializer_function_ser_schema(
                serialize,
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
    # Pre-compressed files for Kubernetes HTTP transfer (optional)
    compressed_data_file_path: Path | None = None
    compressed_index_file_path: Path | None = None
    compressed: bool = False
    compressed_size_bytes: int = 0


# Hot-path dataset models use msgspec.Struct for ~3-4x faster encode/decode/construct
# vs Pydantic v2. These are instantiated per-turn per-request at high QPS.
# Media (and its Text/Image/Audio/Video subclasses) is a slotted dataclass so it
# works natively with both msgspec (Turn.texts / Turn.images / ...) and Pydantic
# (SingleTurn / RandomPool / MultiTurn dataset-loader schemas) without a
# compatibility shim. ``extra="forbid"`` keeps Pydantic's union discrimination
# honest when the loader unions a Media-subclass variant with another shape.


@dataclass(slots=True, kw_only=True)
class Media:
    """Base class for all media fields. Contains name and contents of the media data."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    name: str = ""
    contents: list[str] = field(default_factory=list)


@dataclass(slots=True, kw_only=True)
class Text(Media):
    """Media that contains text/prompt data."""

    media_type: ClassVar[MediaTypeT] = MediaType.TEXT


@dataclass(slots=True, kw_only=True)
class Image(Media):
    """Media that contains image data."""

    media_type: ClassVar[MediaTypeT] = MediaType.IMAGE


@dataclass(slots=True, kw_only=True)
class Audio(Media):
    """Media that contains audio data."""

    media_type: ClassVar[MediaTypeT] = MediaType.AUDIO


@dataclass(slots=True, kw_only=True)
class Video(Media):
    """Media that contains video data."""

    media_type: ClassVar[MediaTypeT] = MediaType.VIDEO


class TurnMetadata(
    _PydanticCompatMixin,
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
):
    """Metadata of a turn: timing + DAG projection fields."""

    timestamp_ms: int | float | None = None
    delay_ms: int | float | None = None
    branch_ids: list[str] = msgspec.field(default_factory=list)
    """Branch IDs declared on this turn (DAG projection). Mirrors
    ``Turn.branch_ids`` for ``ConversationMetadata`` consumers."""
    has_forks: bool = False
    """True if this turn triggers any FORK-mode branch. Stamped at load time
    by the dag_jsonl loader's topology walk so the sticky router can defer
    parent-session eviction until all forks have spawned."""
    prerequisites: list[TurnPrerequisite] = msgspec.field(default_factory=list)
    """Conditions gating dispatch of this turn (DAG projection)."""


class Turn(
    _PydanticCompatMixin,
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
    """Pre-formatted OpenAI-compatible messages array. When set, bypasses
    normal turn-based message construction in endpoints."""
    raw_tools: list[dict[str, Any]] | None = None
    """Pre-formatted OpenAI-compatible tool definitions. When set alongside
    raw_messages, injected into the API payload."""
    texts: list[Text] = msgspec.field(default_factory=list)
    images: list[Image] = msgspec.field(default_factory=list)
    audios: list[Audio] = msgspec.field(default_factory=list)
    videos: list[Video] = msgspec.field(default_factory=list)
    raw_payload: dict[str, Any] | None = None
    """Complete pre-built API request payload for verbatim replay. When set,
    bypasses all endpoint payload construction and sends this dict directly
    to the transport."""
    extra_body: dict[str, Any] | None = None
    """Non-native per-turn request-body fields merged into the top level of
    the chat-completions payload at dispatch time."""
    prerequisites: list[TurnPrerequisite] = msgspec.field(default_factory=list)
    """Conditions gating dispatch of this turn (DAG authoring)."""
    branch_ids: list[str] = msgspec.field(default_factory=list)
    """Branch IDs declared on this turn (DAG authoring)."""
    audio_duration_seconds: float | None = None
    """Duration of the audio content in seconds. Used by ASR-specific metrics
    like RTFx."""

    def metadata(self) -> TurnMetadata:
        """Get the metadata of the turn."""
        return TurnMetadata(
            timestamp_ms=self.timestamp,
            delay_ms=self.delay,
            branch_ids=list(self.branch_ids),
            prerequisites=list(self.prerequisites),
        )

    def copy_with_stripped_media(self) -> Turn:
        """Create a copy of this turn with multimodal data replaced by placeholders.

        This preserves text data (needed for tokenization) and raw messages/tools
        (needed for API payload reconstruction) but replaces potentially large
        image/audio/video contents with small placeholder strings.
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
            raw_payload=self.raw_payload,
            extra_body=dict(self.extra_body) if self.extra_body is not None else None,
            prerequisites=list(self.prerequisites),
            branch_ids=list(self.branch_ids),
            audio_duration_seconds=self.audio_duration_seconds,
        )


class ConversationMetadata(
    _PydanticCompatMixin,
    msgspec.Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
):
    """Metadata of a conversation."""

    conversation_id: str
    turns: list[TurnMetadata] = msgspec.field(default_factory=list)
    branches: list[ConversationBranchInfo] = msgspec.field(default_factory=list)
    """Branch descriptors (DAG projection); empty on non-DAG datasets."""
    is_root: bool = True
    """True for sampleable roots; False for fork/spawn children."""
    agent_depth: int = 0
    """DAG nesting level (0 = root). Mirrors Conversation.agent_depth."""
    parent_conversation_id: str | None = None
    """DAG child's parent conversation_id; None for roots."""
    context_mode: ConversationContextMode | None = None
    """Optional per-conversation context-mode override. Falls back to
    DatasetMetadata.default_context_mode when None."""
    accuracy_ground_truth: str | None = None
    """Ground-truth answer for this conversation (accuracy mode only)."""
    accuracy_task: str | None = None
    """Benchmark sub-task name for this conversation (accuracy mode only)."""


class DatasetMetadata(
    _PydanticCompatMixin,
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
        # perf: validation moved to msgspec/protocol boundary
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
    _PydanticCompatMixin,
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
    """Optional shared system message prepended to the first turn."""
    user_context_message: str | None = None
    """Optional per-conversation user context prepended to the first turn."""
    accuracy_ground_truth: str | None = None
    """Ground-truth answer for this conversation (accuracy mode only)."""
    accuracy_task: str | None = None
    """Benchmark sub-task name for this conversation (accuracy mode only)."""
    agent_depth: int = 0
    """Static DAG nesting level — 0 for sampleable roots,
    ``parent_depth + 1`` for fork-spawned descendants."""
    branches: list[ConversationBranchInfo] = msgspec.field(default_factory=list)
    """Branch descriptors (DAG authoring). Empty on non-DAG datasets."""
    is_root: bool = True
    """True for sampleable roots; False for fork/spawn children."""
    parent_conversation_id: str | None = None
    """DAG child's parent conversation_id; None for roots."""

    def __post_init__(self) -> None:
        # perf: validation moved to msgspec/protocol boundary
        if self.context_mode == ConversationContextMode.MESSAGE_ARRAY_WITHOUT_RESPONSES:
            raise ValueError(
                f"{ConversationContextMode.MESSAGE_ARRAY_WITHOUT_RESPONSES} is not yet supported"
            )

    def metadata(self) -> ConversationMetadata:
        """Project this Conversation into its DatasetMetadata form."""
        modes = {b.branch_id: b.mode for b in self.branches}
        turn_metas = [
            TurnMetadata(
                timestamp_ms=t.timestamp,
                delay_ms=t.delay,
                branch_ids=list(t.branch_ids),
                has_forks=any(
                    modes.get(bid) == ConversationBranchMode.FORK
                    for bid in t.branch_ids
                ),
                prerequisites=list(t.prerequisites),
            )
            for t in self.turns
        ]
        return ConversationMetadata(
            conversation_id=self.session_id,
            turns=turn_metas,
            branches=list(self.branches),
            is_root=self.is_root,
            agent_depth=self.agent_depth,
            parent_conversation_id=self.parent_conversation_id,
            accuracy_ground_truth=self.accuracy_ground_truth,
            accuracy_task=self.accuracy_task,
        )


class SessionPayloads(
    _PydanticCompatMixin,
    msgspec.Struct,
    frozen=True,
    kw_only=True,
):
    """A single session, with its session ID and a list of formatted payloads (one per turn)."""

    session_id: str | None = None
    payloads: list[dict[str, Any]] = msgspec.field(default_factory=list)


class InputsFile(
    _PydanticCompatMixin,
    msgspec.Struct,
    frozen=True,
    kw_only=True,
):
    """A list of all dataset sessions.

    Intentionally does not set ``omit_defaults=True``: the on-disk inputs.json
    schema contract always includes ``"data": [...]``.
    """

    data: list[SessionPayloads] = msgspec.field(default_factory=list)
