# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

from pydantic import Field

from aiperf.common.enums import MessageType, ServiceState, ServiceType
from aiperf.common.messages import Message, StatusMessage
from aiperf.common.pydantic_utils import exclude_if_none


@exclude_if_none(["b"])
class MockMessage(Message):
    a: int
    b: int | None = Field(default=None)


@exclude_if_none(["c"])
class MockMessageSubclass(MockMessage):
    c: int | None = Field(default=None)


def test_exclude_if_none():
    message = MockMessage(message_type=MessageType.UNKNOWN, a=1, b=None)
    assert message.model_dump() == {"message_type": MessageType.UNKNOWN, "a": 1}
    assert message.model_dump_json() == '{"message_type":"unknown","a":1}'

    message = MockMessage(message_type=MessageType.UNKNOWN, a=1, b=2)
    assert message.model_dump() == {"message_type": MessageType.UNKNOWN, "a": 1, "b": 2}
    assert message.model_dump_json() == '{"message_type":"unknown","a":1,"b":2}'

    message = MockMessage(message_type=MessageType.UNKNOWN, a=1)
    assert message.model_dump() == {"message_type": MessageType.UNKNOWN, "a": 1}
    assert message.model_dump_json() == '{"message_type":"unknown","a":1}'


def test_exclude_if_none_subclass():
    message = MockMessageSubclass(message_type=MessageType.UNKNOWN, a=1, b=None, c=None)
    assert message.model_dump() == {"message_type": MessageType.UNKNOWN, "a": 1}
    assert message.model_dump_json() == '{"message_type":"unknown","a":1}'

    message = MockMessageSubclass(message_type=MessageType.UNKNOWN, a=1, b=2, c=None)
    assert message.model_dump() == {"message_type": MessageType.UNKNOWN, "a": 1, "b": 2}
    assert message.model_dump_json() == '{"message_type":"unknown","a":1,"b":2}'

    message = MockMessageSubclass(message_type=MessageType.UNKNOWN, a=1, b=2, c=3)
    assert message.model_dump() == {
        "message_type": MessageType.UNKNOWN,
        "a": 1,
        "b": 2,
        "c": 3,
    }
    assert message.model_dump_json() == '{"message_type":"unknown","a":1,"b":2,"c":3}'


def test_exclude_if_none_decorator():
    @exclude_if_none(["some_field"])
    class ExampleMessage(Message):
        some_field: int | None = Field(default=None)

    message = ExampleMessage(message_type=MessageType.UNKNOWN, some_field=None)
    assert message.model_dump() == {"message_type": MessageType.UNKNOWN}
    assert message.model_dump_json() == '{"message_type":"unknown"}'

    message = ExampleMessage(message_type=MessageType.UNKNOWN, some_field=1)
    assert message.model_dump() == {
        "message_type": MessageType.UNKNOWN,
        "some_field": 1,
    }
    assert message.model_dump_json() == '{"message_type":"unknown","some_field":1}'


def test_status_message():
    message = StatusMessage(
        state=ServiceState.READY,
        service_id="test",
        service_type=ServiceType.WORKER,
        request_ns=1234567890,
        request_id="test",
    )
    assert message.model_dump() == {
        "message_type": MessageType.STATUS,
        "state": ServiceState.READY,
        "service_id": "test",
        "service_type": ServiceType.WORKER,
        "request_ns": 1234567890,
        "request_id": "test",
    }
    assert json.loads(message.model_dump_json()) == json.loads(
        '{"message_type":"status","state":"ready","service_id":"test","service_type":"worker","request_ns":1234567890,"request_id":"test"}'
    )

    message = StatusMessage(
        state=ServiceState.READY,
        request_ns=1234567890,
        request_id=None,
        service_id="test",
        service_type=ServiceType.WORKER,
    )
    assert message.model_dump() == {
        "message_type": MessageType.STATUS,
        "state": ServiceState.READY,
        "service_id": "test",
        "service_type": ServiceType.WORKER,
        "request_ns": 1234567890,
    }
    assert json.loads(message.model_dump_json()) == json.loads(
        '{"message_type":"status","state":"ready","service_id":"test","service_type":"worker","request_ns":1234567890}'
    )
