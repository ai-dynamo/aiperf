# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for AutoRoutedModel-based routing.

The nested-discriminator cases were originally written against the pub/sub
command hierarchy (``Message`` -> ``CommandMessage`` -> ``SpawnWorkersCommand``
-> ``CommandSuccessResponse`` -> ``ProcessRecordsResponse``). That hierarchy is
gone with the pub/sub command plumbing, but multi-level routing is still a
capability of ``AutoRoutedModel``, so the cases are kept against a hierarchy
defined here. The single-level case still exercises a real message.
"""

import json
from typing import Any, ClassVar

import pytest

from aiperf.common.enums import LifecycleState, MessageType
from aiperf.common.messages import HeartbeatMessage, Message
from aiperf.common.models.base_models import AIPerfBaseModel


class Envelope(AIPerfBaseModel):
    """Level 1: routes on ``kind``."""

    discriminator_field: ClassVar[str] = "kind"

    kind: str


class RequestEnvelope(Envelope):
    """Level 2: routes on ``action``."""

    discriminator_field: ClassVar[str] = "action"

    kind: str = "request"
    action: str


class SpawnRequest(RequestEnvelope):
    action: str = "spawn"
    num_workers: int


class DrainRequest(RequestEnvelope):
    action: str = "drain"
    cancelled: bool = False


class ResponseEnvelope(Envelope):
    """Level 2: routes on ``status``."""

    discriminator_field: ClassVar[str] = "status"

    kind: str = "response"
    status: str


class FailureResponse(ResponseEnvelope):
    status: str = "failure"
    error: str


class SuccessResponse(ResponseEnvelope):
    """Level 3: success responses route further on ``action``."""

    discriminator_field: ClassVar[str] = "action"

    status: str = "success"
    action: str = ""
    data: Any | None = None


class SpawnSuccessResponse(SuccessResponse):
    action: str = "spawn"


def assert_routed_to(msg, expected_class, **expected_attrs):
    """Assert message routed to expected class with expected attributes."""
    assert isinstance(msg, expected_class), (
        f"Expected {expected_class.__name__}, got {type(msg).__name__}"
    )
    for attr, value in expected_attrs.items():
        assert getattr(msg, attr) == value, (
            f"Expected {attr}={value}, got {getattr(msg, attr)}"
        )


class TestAutoRoutedModel:
    """Test AutoRoutedModel routing behavior."""

    def test_single_level_routing_on_a_real_message(self):
        """The production path: message_type alone selects the class."""
        msg = Message.from_json(
            {
                "message_type": "heartbeat",
                "state": "running",
                "service_id": "test-service",
                "service_type": "worker",
            }
        )
        assert_routed_to(
            msg,
            HeartbeatMessage,
            message_type=MessageType.HEARTBEAT,
            state=LifecycleState.RUNNING,
        )

    @pytest.mark.parametrize(
        "data,expected_class,expected_attrs",
        [
            (
                {"kind": "request", "action": "spawn", "num_workers": 5},
                SpawnRequest,
                {"kind": "request", "action": "spawn", "num_workers": 5},
            ),
            (
                {"kind": "request", "action": "drain", "cancelled": True},
                DrainRequest,
                {"action": "drain", "cancelled": True},
            ),
            # Fallback to the intermediate class for an unregistered value.
            (
                {"kind": "request", "action": "unknown_action"},
                RequestEnvelope,
                {"action": "unknown_action"},
            ),
        ],
    )  # fmt: skip
    def test_two_level_routing(self, data, expected_class, expected_attrs):
        msg = Envelope.from_json(data)
        assert_routed_to(msg, expected_class, **expected_attrs)

    @pytest.mark.parametrize(
        "data,expected_class,expected_attrs",
        [
            (
                {"kind": "response", "status": "failure", "error": "Failed"},
                FailureResponse,
                {"status": "failure"},
            ),
            (
                {"kind": "response", "status": "success", "action": "spawn"},
                SpawnSuccessResponse,
                {"status": "success", "action": "spawn"},
            ),
        ],
    )  # fmt: skip
    def test_three_level_routing(self, data, expected_class, expected_attrs):
        """kind -> status -> action, resolved in a single parse."""
        msg = Envelope.from_json(data)
        assert_routed_to(msg, expected_class, **expected_attrs)

    def test_specialized_response_carries_its_payload(self, process_records_result):
        msg = Envelope.from_json(
            {
                "kind": "response",
                "status": "success",
                "action": "spawn",
                "data": process_records_result,
            }
        )
        assert_routed_to(msg, SpawnSuccessResponse, action="spawn")
        assert msg.data is not None

    def test_json_string_routing(self, base_message_data):
        """Test routing from JSON string (ensures single parse)."""
        data = {
            **base_message_data,
            "message_type": "heartbeat",
            "state": "running",
            "service_type": "worker",
        }
        msg = Message.from_json(json.dumps(data))
        assert_routed_to(msg, HeartbeatMessage, state=LifecycleState.RUNNING)

    @pytest.mark.parametrize(
        "model,data,match",
        [
            (Message, {"service_id": "test"}, "Missing discriminator 'message_type'"),
            (Envelope, {}, "Missing discriminator 'kind'"),
            (Envelope, {"kind": "request"}, "Missing discriminator 'action'"),
        ],
    )  # fmt: skip
    def test_missing_discriminator_error(self, model, data, match):
        """Test that missing discriminators raise ValueError."""
        with pytest.raises(ValueError, match=match):
            model.from_json(data)

    def test_unknown_discriminator_value_falls_back_to_base_class(self):
        """Test that unknown discriminator values fall back to base class validation."""
        # Unknown message type should still work with base Message class
        data = {
            "message_type": "unknown_type",
            "service_id": "test",
        }
        msg = Message.from_json(data)
        # Should be validated as base Message class
        assert msg.message_type == "unknown_type"
        assert msg.service_id == "test"

    @pytest.mark.parametrize(
        "input_transform,description",
        [
            (lambda d: d, "dict (no parsing)"),
            (lambda d: json.dumps(d), "JSON string"),
            (lambda d: json.dumps(d).encode("utf-8"), "bytes"),
            (lambda d: bytearray(json.dumps(d).encode("utf-8")), "bytearray"),
        ],
    )  # fmt: skip
    def test_from_json_input_types(self, input_transform, description):
        """Test that from_json accepts various input types: dict, str, bytes, bytearray."""
        data = {
            "message_type": "heartbeat",
            "state": "running",
            "service_id": "test-service",
            "service_type": "worker",
        }
        msg = Message.from_json(input_transform(data))
        assert_routed_to(msg, HeartbeatMessage, state=LifecycleState.RUNNING)
