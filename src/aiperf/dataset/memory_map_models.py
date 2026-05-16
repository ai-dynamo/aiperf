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
    """msgspec-API-compatible wrapper around msgspec.json for Conversation.

    Conversation is a msgspec.Struct (restored from the older performance
    branch), so we use msgspec.json.encode directly. The pydantic-compat
    ``model_dump`` shim is also available for callers that need a dict.
    """

    @staticmethod
    def encode(conversation: Conversation) -> bytes:
        from aiperf.common.models.base_models import _msgspec_enc_hook

        return msgspec.json.encode(conversation, enc_hook=_msgspec_enc_hook)


_CONVERSATION_ENCODER = _ConversationOrjsonEncoder()


class _ConversationOrjsonDecoder:
    """msgspec-API-compatible decoder for Conversation msgspec.Struct."""

    @staticmethod
    def decode(data: bytes) -> Conversation:
        from aiperf.common.models.base_models import _msgspec_dec_hook

        return msgspec.json.decode(data, type=Conversation, dec_hook=_msgspec_dec_hook)


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
