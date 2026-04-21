# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import CreditPhase, MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import (
    Conversation,
    DatasetMetadata,
    MemoryMapClientMetadata,
    Turn,
)

# msgspec tagged-union discrimination does not walk the subclass graph for a
# ``DatasetClientMetadata`` base annotation — the concrete union has to be
# spelt out as the field type. Today ``MemoryMapClientMetadata`` is the only
# concrete variant; when a second lands, widen this alias to ``A | B``.
DatasetClientMetadataUnion = MemoryMapClientMetadata


class ConversationRequestMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.CONVERSATION_REQUEST.value
):
    """Request a full conversation by ID."""

    conversation_id: str
    credit_phase: CreditPhase | None = None


class ConversationResponseMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.CONVERSATION_RESPONSE.value
):
    """Full conversation payload."""

    conversation: Conversation


class ConversationTurnRequestMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.CONVERSATION_TURN_REQUEST.value
):
    """Request a single turn by (conversation_id, turn_index)."""

    conversation_id: str
    turn_index: int


class ConversationTurnResponseMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.CONVERSATION_TURN_RESPONSE.value
):
    """Single turn payload."""

    turn: Turn


class DatasetConfiguredNotification(
    BaseServiceMessage,
    kw_only=True,
    tag=MessageType.DATASET_CONFIGURED_NOTIFICATION.value,
):
    """Broadcast that dataset configuration is complete."""

    metadata: DatasetMetadata
    client_metadata: DatasetClientMetadataUnion
    benchmark_generation: str
    dataset_generation: str


class DatasetDownloadedNotification(
    BaseServiceMessage,
    kw_only=True,
    tag=MessageType.DATASET_DOWNLOADED_NOTIFICATION.value,
):
    """Pod-scoped dataset download complete."""

    client_metadata: MemoryMapClientMetadata
    pod_index: str | None = None
    success: bool = True
    error_message: str | None = None
