# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tagged-union round-trip tests for Message after P2 msgspec flip."""

import msgspec
import orjson

from aiperf.common.enums import LifecycleState, MessageType
from aiperf.common.messages import (
    HeartbeatMessage,
    Message,
    StatusMessage,
)
from aiperf.plugin.enums import ServiceType


def test_heartbeat_message_msgpack_roundtrip_via_base_decoder():
    """Encoding a HeartbeatMessage and decoding as Message routes to HeartbeatMessage."""
    msg = HeartbeatMessage(
        service_id="svc-1",
        service_type=ServiceType.WORKER,
        state=LifecycleState.RUNNING,
    )
    encoder = msgspec.msgpack.Encoder()
    decoder = msgspec.msgpack.Decoder(type=Message)

    restored = decoder.decode(encoder.encode(msg))

    assert isinstance(restored, HeartbeatMessage)
    assert restored.service_id == "svc-1"
    assert restored.service_type == ServiceType.WORKER
    assert restored.state == LifecycleState.RUNNING


def test_json_roundtrip_preserves_tag_field_name():
    """JSON emits the 'message_type' tag so external consumers stay compatible."""
    msg = StatusMessage(
        service_id="svc-2",
        service_type=ServiceType.TIMING_MANAGER,
        state=LifecycleState.STOPPED,
    )
    encoded = msgspec.json.encode(msg)
    as_dict = msgspec.json.decode(encoded)
    assert as_dict["message_type"] == MessageType.STATUS


def test_from_json_compat_wrapper_accepts_bytes_and_dict():
    """Message.from_json preserves AutoRoutedModel's dual-input API over msgspec."""
    heartbeat_dict = {
        "message_type": MessageType.HEARTBEAT,
        "service_id": "svc-3",
        "service_type": ServiceType.WORKER,
        "state": LifecycleState.RUNNING,
    }
    from_dict = Message.from_json(heartbeat_dict)
    from_bytes = Message.from_json(orjson.dumps(heartbeat_dict))
    assert isinstance(from_dict, HeartbeatMessage)
    assert isinstance(from_bytes, HeartbeatMessage)


def test_to_json_bytes_shim_emits_message_type_tag():
    """`to_json_bytes()` stays wire-compatible with the prior Pydantic path."""
    msg = HeartbeatMessage(
        service_id="svc-4",
        service_type=ServiceType.WORKER,
        state=LifecycleState.RUNNING,
    )
    decoded = orjson.loads(msg.to_json_bytes())
    assert decoded["message_type"] == MessageType.HEARTBEAT
    assert decoded["service_id"] == "svc-4"
    # request_id default (None) is excluded
    assert "request_id" not in decoded
