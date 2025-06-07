#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
import json

from pydantic import Field

from aiperf.common.enums import MessageType, ServiceState, ServiceType
from aiperf.common.messages import BaseMessage, StatusMessage, exclude_if_none


@exclude_if_none(["b"])
class _TestMessage(BaseMessage):
    a: int
    b: int | None = Field(default=None)


@exclude_if_none(["c"])
class _TestMessageSubclass(_TestMessage):
    c: int | None = Field(default=None)


def test_exclude_if_none():
    message = _TestMessage(a=1, b=None)
    assert message.model_dump() == {"a": 1}
    assert message.model_dump_json() == '{"a":1}'

    message = _TestMessage(a=1, b=2)
    assert message.model_dump() == {"a": 1, "b": 2}
    assert message.model_dump_json() == '{"a":1,"b":2}'

    message = _TestMessage(a=1)
    assert message.model_dump() == {"a": 1}
    assert message.model_dump_json() == '{"a":1}'


def test_exclude_if_none_subclass():
    message = _TestMessageSubclass(a=1, b=None, c=None)
    assert message.model_dump() == {"a": 1}
    assert message.model_dump_json() == '{"a":1}'

    message = _TestMessageSubclass(a=1, b=2, c=None)
    assert message.model_dump() == {"a": 1, "b": 2}
    assert message.model_dump_json() == '{"a":1,"b":2}'

    message = _TestMessageSubclass(a=1, b=2, c=3)
    assert message.model_dump() == {"a": 1, "b": 2, "c": 3}
    assert message.model_dump_json() == '{"a":1,"b":2,"c":3}'


def test_exclude_if_none_decorator():
    @exclude_if_none(["some_field"])
    class ExampleMessage(BaseMessage):
        some_field: int | None = Field(default=None)

    message = ExampleMessage(some_field=None)
    assert message.model_dump() == {}
    assert message.model_dump_json() == "{}"

    message = ExampleMessage(some_field=1)
    assert message.model_dump() == {"some_field": 1}
    assert message.model_dump_json() == '{"some_field":1}'


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
