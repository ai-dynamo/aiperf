# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for AutoRoutedModel-based message routing."""

import json

import pytest

from aiperf.common.enums import (
    CommandResponseStatus,
    CommandType,
    LifecycleState,
    MessageType,
)
from aiperf.common.messages import Message, StatusMessage
from aiperf.common.messages.command_messages import (
    CommandErrorResponse,
    CommandMessage,
    CommandSuccessResponse,
    ProcessRecordsCommand,
    ProcessRecordsResponse,
    SpawnWorkersCommand,
)


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

    @pytest.mark.parametrize(
        "data,expected_class,expected_attrs",
        [
            # Single-level routing
            (
                {
                    "message_type": "status",
                    "state": "running",
                    "service_id": "test-service",
                    "service_type": "worker",
                },
                StatusMessage,
                {"message_type": MessageType.STATUS, "state": LifecycleState.RUNNING},
            ),
            # Two-level routing
            (
                {
                    "message_type": "command",
                    "command": "spawn_workers",
                    "service_id": "controller",
                    "num_workers": 5,
                },
                SpawnWorkersCommand,
                {"message_type": MessageType.COMMAND, "command": CommandType.SPAWN_WORKERS, "num_workers": 5},
            ),
            (
                {
                    "message_type": "command",
                    "command": "process_records",
                    "service_id": "controller",
                    "cancelled": True,
                },
                ProcessRecordsCommand,
                {"command": CommandType.PROCESS_RECORDS, "cancelled": True},
            ),
            # Fallback to base class
            (
                {
                    "message_type": "command",
                    "command": "unknown_command",
                    "service_id": "controller",
                },
                CommandMessage,
                {"command": "unknown_command"},
            ),
        ],
    )  # fmt: skip
    def test_routing_levels(self, data, expected_class, expected_attrs):
        """Test routing at various nesting levels."""
        msg = Message.from_json(data)
        assert_routed_to(msg, expected_class, **expected_attrs)

    @pytest.mark.parametrize(
        "data,expected_class,expected_attrs",
        [
            # Error response (status routing)
            (
                {
                    "message_type": "command_response",
                    "status": "failure",
                    "command": "spawn_workers",
                    "command_id": "cmd-123",
                    "service_id": "worker",
                    "error": {"message": "Failed", "type": "Error"},
                },
                CommandErrorResponse,
                {"status": CommandResponseStatus.FAILURE},
            ),
            # Success response (status + command routing)
            (
                {
                    "message_type": "command_response",
                    "status": "success",
                    "command": "spawn_workers",
                    "command_id": "cmd-123",
                    "service_id": "worker",
                    "data": {"count": 5},
                },
                CommandSuccessResponse,
                {"status": CommandResponseStatus.SUCCESS},
            ),
        ],
    )  # fmt: skip
    def test_response_routing(self, data, expected_class, expected_attrs):
        """Test three-level routing for command responses."""
        msg = Message.from_json(data)
        assert_routed_to(msg, expected_class, **expected_attrs)

    def test_specialized_response_routing(self, process_records_result):
        """Test routing to specialized response class."""
        data = {
            "message_type": "command_response",
            "status": "success",
            "command": "process_records",
            "command_id": "cmd-456",
            "service_id": "records-manager",
            "data": process_records_result,
        }

        msg = Message.from_json(data)
        assert_routed_to(
            msg, ProcessRecordsResponse, command=CommandType.PROCESS_RECORDS
        )

    def test_json_string_routing(self, base_message_data):
        """Test routing from JSON string (ensures single parse)."""
        data = {
            **base_message_data,
            "message_type": "status",
            "state": "running",
            "service_type": "worker",
        }
        msg = Message.from_json(json.dumps(data))
        assert_routed_to(msg, StatusMessage, state=LifecycleState.RUNNING)

    @pytest.mark.parametrize(
        "data,match",
        [
            ({"service_id": "test"}, "Missing discriminator 'message_type'"),
            ({"message_type": "command", "service_id": "test"}, "Missing discriminator 'command'"),
        ],
    )  # fmt: skip
    def test_missing_discriminator_error(self, data, match):
        """Test that missing discriminators raise ValueError."""
        with pytest.raises(ValueError, match=match):
            Message.from_json(data)

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
            "message_type": "status",
            "state": "running",
            "service_id": "test-service",
            "service_type": "worker",
        }
        msg = Message.from_json(input_transform(data))
        assert_routed_to(msg, StatusMessage, state=LifecycleState.RUNNING)
