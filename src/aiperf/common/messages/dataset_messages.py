# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import Field, SerializeAsAny, field_validator

from aiperf.common.enums import CreditPhase, MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import (
    Conversation,
    DatasetClientMetadata,
    DatasetMetadata,
    Turn,
)
from aiperf.common.types import MessageTypeT


class ConversationRequestMessage(BaseServiceMessage):
    """Message to request a full conversation by ID."""

    message_type: MessageTypeT = MessageType.CONVERSATION_REQUEST

    conversation_id: str = Field(..., description="The dataset conversation ID")
    credit_phase: CreditPhase | None = Field(
        default=None,
        description="The type of credit phase (either warmup or profiling). If not provided, the dataset manager will use the default credit phase.",
    )


class ConversationResponseMessage(BaseServiceMessage):
    """Message containing a full conversation."""

    message_type: MessageTypeT = MessageType.CONVERSATION_RESPONSE
    conversation: Conversation = Field(..., description="The conversation data")


class ConversationTurnRequestMessage(BaseServiceMessage):
    """Message to request a single turn from a conversation."""

    message_type: MessageTypeT = MessageType.CONVERSATION_TURN_REQUEST

    conversation_id: str = Field(
        ...,
        description="The ID of the conversation.",
    )
    turn_index: int = Field(
        ...,
        ge=0,
        description="The index of the turn in the conversation.",
    )


class ConversationTurnResponseMessage(BaseServiceMessage):
    """Message containing a single turn from a conversation."""

    message_type: MessageTypeT = MessageType.CONVERSATION_TURN_RESPONSE

    turn: Turn = Field(..., description="The turn data")


class DatasetDownloadedNotification(BaseServiceMessage):
    """Notification that a worker pod finished materializing the dataset locally.

    Kubernetes only. The DatasetManager's DatasetConfiguredNotification
    describes files on the controller pod; a worker pod cannot open those. The
    pod's WorkerGroupManager downloads them to its own emptyDir and publishes
    this so its sibling workers know when the pod-local files exist and where.
    Workers must not open the mmap before this arrives -- the download finishes
    tens of milliseconds after the configured notification, so an eager open
    reliably fails with "Data file not found".
    """

    message_type: MessageTypeT = MessageType.DATASET_DOWNLOADED_NOTIFICATION

    client_metadata: SerializeAsAny[DatasetClientMetadata] = Field(
        ...,
        description="Pod-local client access metadata: the mmap paths inside this "
        "pod, decompressed and ready to open.",
    )
    pod_index: str | None = Field(
        default=None,
        description="Index of the worker pod that downloaded the dataset. Workers "
        "ignore notifications from other pods, whose files they cannot see.",
    )
    success: bool = Field(
        default=True,
        description="False when the download failed; the paths are placeholders.",
    )
    error_message: str | None = Field(
        default=None,
        description="Failure detail when success is False.",
    )

    @field_validator("client_metadata", mode="before")
    @classmethod
    def route_client_metadata(cls, v: Any) -> DatasetClientMetadata:
        """Route the nested AutoRoutedModel field to its concrete subclass."""
        if isinstance(v, dict):
            return DatasetClientMetadata.from_json(v)
        return v


class DatasetConfiguredNotification(BaseServiceMessage):
    """Notification sent to notify other services that the dataset has been configured.

    Contains two separate pieces of information:
    - metadata: Dataset structure (conversations, sampling strategy) for timing strategies
    - client_metadata: Client access info (e.g., mmap paths) for workers to read data
    """

    message_type: MessageTypeT = MessageType.DATASET_CONFIGURED_NOTIFICATION

    metadata: DatasetMetadata = Field(
        ...,
        description="Dataset structure metadata (conversations, timing) for timing strategies.",
    )
    client_metadata: SerializeAsAny[DatasetClientMetadata] = Field(
        ...,
        description="Client access metadata (e.g., mmap file paths) for workers to read dataset.",
    )
    benchmark_generation: str | None = Field(
        default=None,
        description="Identity of the benchmark this dataset was built for. Worker pods "
        "and the API dataset router tag their state with it so a stale pod can be "
        "told apart from one serving the current benchmark.",
    )
    dataset_generation: str | None = Field(
        default=None,
        description="Identity of the dataset itself. Changes whenever the dataset is "
        "rebuilt, so a pod can tell whether the files it already downloaded are the "
        "ones this notification describes.",
    )

    @field_validator("client_metadata", mode="before")
    @classmethod
    def route_client_metadata(cls, v: Any) -> DatasetClientMetadata:
        """Route nested AutoRoutedModel field to correct subclass.

        Pydantic's nested model validation doesn't use AutoRoutedModel.from_json(),
        so we manually route dict inputs to the correct subclass based on client_type.
        """
        if isinstance(v, dict):
            return DatasetClientMetadata.from_json(v)
        return v


class DatasetConfigurationFailedNotification(BaseServiceMessage):
    """Notification published by DatasetManager when its PROFILE_CONFIGURE handler raises.

    Lets peer services (notably TimingManager, which awaits
    DatasetConfiguredNotification) abort their wait immediately instead of
    blocking on the dataset configuration timeout. The CommandErrorResponse
    path remains the authoritative failure signal for the SystemController;
    this notification is the broadcast equivalent for fan-out wakeups.
    """

    message_type: MessageTypeT = MessageType.DATASET_CONFIGURATION_FAILED

    error: str = Field(
        ...,
        description="Human-readable description of the dataset configuration failure.",
    )
