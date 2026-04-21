# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Round-trip tests for dataset envelopes carrying msgspec payload structs.

Exists because the dataset hot-path (Conversation/Turn/media) and metadata
(TurnMetadata/ConversationMetadata/DatasetMetadata) plus the
DatasetClientMetadata tagged union are msgspec.Struct, while the envelope
messages remain Pydantic during Phase 2.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import param

from aiperf.common.enums import ConversationContextMode
from aiperf.common.messages.dataset_messages import (
    ConversationRequestMessage,
    ConversationResponseMessage,
    ConversationTurnRequestMessage,
    ConversationTurnResponseMessage,
    DatasetConfiguredNotification,
    DatasetDownloadedNotification,
)
from aiperf.common.models import (
    Conversation,
    ConversationMetadata,
    DatasetMetadata,
    MemoryMapClientMetadata,
    Turn,
    TurnMetadata,
)
from aiperf.plugin.enums import DatasetSamplingStrategy


def _conversation() -> Conversation:
    return Conversation(
        session_id="s1",
        context_mode=ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES,
        turns=[Turn(role="user")],
    )


def _dataset_metadata() -> DatasetMetadata:
    return DatasetMetadata(
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        conversations=[
            ConversationMetadata(
                conversation_id="s1",
                turns=[TurnMetadata(timestamp_ms=0)],
            ),
        ],
        has_timing_data=True,
    )


def _client_metadata() -> MemoryMapClientMetadata:
    return MemoryMapClientMetadata(
        data_file_path=Path("/tmp/data.dat"),
        index_file_path=Path("/tmp/index.dat"),
        conversation_count=1,
    )


@pytest.mark.parametrize(
    "message_factory",
    [
        param(
            lambda: ConversationRequestMessage(service_id="dm", conversation_id="s1"),
            id="ConversationRequestMessage",
        ),
        param(
            lambda: ConversationResponseMessage(
                service_id="dm", conversation=_conversation()
            ),
            id="ConversationResponseMessage",
        ),
        param(
            lambda: ConversationTurnRequestMessage(
                service_id="dm", conversation_id="s1", turn_index=0
            ),
            id="ConversationTurnRequestMessage",
        ),
        param(
            lambda: ConversationTurnResponseMessage(
                service_id="dm", turn=Turn(role="user")
            ),
            id="ConversationTurnResponseMessage",
        ),
        param(
            lambda: DatasetConfiguredNotification(
                service_id="dm",
                metadata=_dataset_metadata(),
                client_metadata=_client_metadata(),
                benchmark_generation="b1",
                dataset_generation="d1",
            ),
            id="DatasetConfiguredNotification",
        ),
        param(
            lambda: DatasetDownloadedNotification(
                service_id="dm",
                client_metadata=_client_metadata(),
                pod_index="0",
            ),
            id="DatasetDownloadedNotification",
        ),
    ],
)
def test_dataset_envelope_roundtrips(message_factory) -> None:
    """Envelope with msgspec payload must round-trip through Pydantic JSON."""
    message = message_factory()

    payload = message.model_dump_json()
    decoded = type(message).model_validate_json(payload)

    assert decoded == message


def test_dataset_configured_routes_tagged_union_from_dict() -> None:
    """DatasetConfiguredNotification.client_metadata dispatches on client_type.

    The tagged-union base class overrides __get_pydantic_core_schema__ to
    build ``Union[<subclasses>]`` at validation time, so a dict with
    ``client_type`` routes to the correct concrete struct.
    """
    payload = {
        "service_id": "dm",
        "request_ns": 1,
        "message_type": "dataset_configured_notification",
        "metadata": {
            "sampling_strategy": "sequential",
            "conversations": [],
            "has_timing_data": False,
        },
        "client_metadata": {
            "client_type": "memory_map",
            "data_file_path": "/tmp/data.dat",
            "index_file_path": "/tmp/index.dat",
            "conversation_count": 1,
        },
        "benchmark_generation": "b1",
        "dataset_generation": "d1",
    }

    msg = DatasetConfiguredNotification.model_validate(payload)

    assert isinstance(msg.client_metadata, MemoryMapClientMetadata)
    assert msg.client_metadata.client_type == "memory_map"
    assert msg.client_metadata.data_file_path == Path("/tmp/data.dat")
