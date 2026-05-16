# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Index models and codec helpers for the memory-mapped dataset.

Kept separate from the writer/reader so both sides can import the index
schema without pulling in mmap/zstd code paths.
"""

import types

import msgspec
import orjson
from pydantic import Field, field_validator

from aiperf.common.models import AIPerfBaseModel, Conversation


class _ConversationOrjsonEncoder:
    """msgspec-API-compatible wrapper around orjson for Pydantic Conversations.

    Branch carried Conversation as ``msgspec.Struct`` so ``msgspec.json.Encoder``
    could serialize it directly. On main Conversation is a Pydantic model
    (kept for the field-description/validator surface), so we encode the JSON
    via orjson's default dispatch. Decoder still uses msgspec.json.Decoder so
    the workers' fast-path bytes-to-struct conversion stays intact — Pydantic
    will accept the dict msgspec materialized.
    """

    @staticmethod
    def encode(conversation: Conversation) -> bytes:
        return orjson.dumps(conversation.model_dump())


_CONVERSATION_ENCODER = _ConversationOrjsonEncoder()
class _ConversationOrjsonDecoder:
    """msgspec-API-compatible decoder for Pydantic Conversations."""

    @staticmethod
    def decode(data: bytes) -> Conversation:
        return Conversation.model_validate(orjson.loads(data))


_CONVERSATION_DECODER: _ConversationOrjsonDecoder | None = None


def _get_conversation_decoder() -> _ConversationOrjsonDecoder:
    global _CONVERSATION_DECODER
    if _CONVERSATION_DECODER is None:
        _CONVERSATION_DECODER = _ConversationOrjsonDecoder()
    return _CONVERSATION_DECODER


def _import_zstandard() -> types.ModuleType:
    """Lazy-import zstandard or raise a helpful error."""
    try:
        import zstandard

        return zstandard
    except ImportError as e:
        raise ImportError(
            "zstandard library required for compression. Install with: pip install zstandard"
        ) from e


class ConversationOffset(AIPerfBaseModel):
    """Offset information for a single conversation in the memory-mapped file."""

    offset: int = Field(ge=0, description="Byte offset where conversation data starts")
    size: int = Field(ge=0, description="Size of the conversation data in bytes")


class MemoryMapDatasetIndex(AIPerfBaseModel):
    """Index structure for the memory-mapped dataset.

    All data is stored as uncompressed JSON bytes serialized with orjson.
    """

    conversation_ids: list[str] = Field(
        default_factory=list, description="List of all conversation IDs in the dataset"
    )
    offsets: dict[str, ConversationOffset] = Field(
        default_factory=dict,
        description="Mapping of conversation IDs to their byte offsets and sizes",
    )
    total_size: int = Field(
        default=0, ge=0, description="Total size of the serialized dataset in bytes"
    )

    @field_validator("conversation_ids")
    @classmethod
    def validate_conversation_ids(cls, v: list[str]) -> list[str]:
        """Ensure conversation_ids are unique."""
        if len(v) != len(set(v)):
            raise ValueError("conversation_ids must contain unique values")
        return v
